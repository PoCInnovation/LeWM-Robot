"""
Réglages matériels pour GPU NVIDIA (cible : RTX 4090, Ada Lovelace, 24 GB).

Tout ce qui dépend du device est centralisé ici, pour que les scripts n'aient
qu'à appeler `configure_backend()` en tête puis `resolve_amp_dtype()` :

    - TF32 pour les matmuls/convs fp32 (gratuit sur Ampere+ : ~2-3x sur les
      produits matriciels fp32, précision largement suffisante ici) ;
    - cuDNN benchmark (shapes fixes → autotune des kernels) ;
    - autocast bf16 si le GPU le supporte (Ampere+, donc 4090 : oui) ;
    - choix du device pour héberger les latents pré-encodés (VRAM si ça tient :
      sur 24 GB, un dataset DINOv3-small entier tient et supprime le goulot
      CPU → GPU du training) ;
    - nombre de workers DataLoader ;
    - AdamW fused, torch.compile opt-in ;
    - suivi de la VRAM (pic alloué) pour calibrer les batch sizes.

Reste 100 % fonctionnel sur CPU (tout devient no-op / fp32).
"""

from __future__ import annotations
import os
from typing import Dict, Iterable, Optional, Tuple, Union

import torch

DeviceLike = Union[str, torch.device]


def configure_backend(tf32: bool = True, cudnn_benchmark: bool = True,
                      verbose: bool = True) -> None:
    """Active TF32 + cuDNN benchmark sur CUDA. No-op sur CPU."""
    if not torch.cuda.is_available():
        return
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # "high" = TF32 autorisé pour les matmuls fp32 (Ampere+)
        torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if verbose:
        print(f"[Device] TF32={'on' if tf32 else 'off'}  "
              f"cudnn.benchmark={'on' if cudnn_benchmark else 'off'}")


def get_device(requested: str = "auto") -> torch.device:
    """'auto' → cuda si dispo, sinon cpu."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def bf16_supported() -> bool:
    """bf16 natif (Ampere+ ; la 4090 est Ada = oui)."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8


def resolve_amp_dtype(device: DeviceLike, requested: str = "auto"
                      ) -> Optional[torch.dtype]:
    """
    Dtype d'autocast à utiliser pour l'entraînement/l'inférence.

    requested : "auto" (bf16 si CUDA + support, sinon aucun), "bf16",
                "fp16", "fp32"/"none"/"off" (pas d'autocast).
    Renvoie None quand il ne faut PAS activer l'autocast.
    """
    dev = torch.device(device)
    req = (requested or "auto").lower()
    valid = ("auto", "fp32", "float32", "none", "off", "false",
             "bf16", "bfloat16", "fp16", "float16", "half")
    if req not in valid:
        raise ValueError(f"Précision inconnue : {requested!r} (attendu : "
                         f"auto / bf16 / fp16 / fp32)")
    if req in ("fp32", "float32", "none", "off", "false"):
        return None
    if dev.type != "cuda":
        return None                     # autocast CPU bf16 : lent, inutile ici
    if req == "auto":
        return torch.bfloat16 if bf16_supported() else None
    if req in ("bf16", "bfloat16"):
        if not bf16_supported():
            print("[Device] bf16 demandé mais non supporté → fp32.")
            return None
        return torch.bfloat16
    return torch.float16                # fp16 / float16 / half


def resolve_model_dtype(device: DeviceLike, requested: str = "auto") -> torch.dtype:
    """
    Dtype des POIDS d'un modèle figé (l'encodeur DINO) : bf16 sur GPU
    compatible, fp32 sinon. Divise par 2 la VRAM et ~x2-3 le débit d'encodage.
    """
    amp = resolve_amp_dtype(device, requested)
    return amp if amp is not None else torch.float32


def autocast_ctx(device: DeviceLike, amp_dtype: Optional[torch.dtype]):
    """Contexte autocast (désactivé si amp_dtype est None)."""
    return torch.autocast(device_type=torch.device(device).type,
                          dtype=amp_dtype or torch.float32,
                          enabled=amp_dtype is not None)


def resolve_num_workers(requested: Union[int, str, None] = "auto",
                        cap: int = 8) -> int:
    """
    Workers DataLoader. 'auto' = min(cap, cœurs - 2). Le décodage vidéo
    (encodage) est le goulot : sur une machine 4090 typique (8-16 cœurs),
    6-8 workers saturent le GPU en DINOv3-small.
    """
    if requested is None or requested == "auto":
        n = os.cpu_count() or 4
        return max(0, min(cap, n - 2))
    return int(requested)


def vram_free_gb(device: DeviceLike = "cuda") -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return free / 1e9


def peak_vram_gb(device: DeviceLike = "cuda") -> float:
    """Pic de mémoire allouée depuis le dernier reset (0 sur CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(torch.device(device)) / 1e9


def reset_peak_vram(device: DeviceLike = "cuda") -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def choose_data_device(nbytes: int, requested: str = "auto",
                       train_device: DeviceLike = "cuda",
                       max_fraction: float = 0.45) -> torch.device:
    """
    Où héberger les latents pré-encodés pendant le training.

    'auto' : en VRAM si `nbytes` <= max_fraction × VRAM libre (il faut laisser
    la place au modèle, aux activations et à l'optimiseur), sinon RAM CPU.
    'cuda' / 'cpu' : forcé.
    """
    train_device = torch.device(train_device)
    req = (requested or "auto").lower()
    if req == "cpu" or train_device.type != "cuda":
        return torch.device("cpu")
    if req in ("cuda", "gpu"):
        return train_device
    free = vram_free_gb(train_device) * 1e9
    if nbytes <= max_fraction * free:
        return train_device
    print(f"[Device] Latents ({nbytes / 1e9:.1f} GB) > {max_fraction:.0%} de la "
          f"VRAM libre ({free / 1e9:.1f} GB) → restent en RAM CPU.")
    return torch.device("cpu")


def place_tensors(tensors: Dict[str, torch.Tensor], requested: str,
                  train_device: DeviceLike, verbose: bool = True
                  ) -> Tuple[Dict[str, torch.Tensor], torch.device]:
    """
    Héberge un dict de tenseurs (latents pré-encodés) en VRAM si possible.

    Renvoie (tenseurs déplacés, device choisi). Avec les tenseurs en VRAM,
    l'indexation `z[idx]` d'un batch reste sur le GPU : zéro transfert PCIe
    pendant le training. Sinon ils restent en RAM (pinned si CUDA dispo)
    et chaque batch est transféré à la volée.
    """
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    target = choose_data_device(nbytes, requested, train_device)
    out = {}
    for k, t in tensors.items():
        if target.type == "cuda":
            out[k] = t.to(target, non_blocking=True)
        elif torch.cuda.is_available():
            out[k] = t.pin_memory()
        else:
            out[k] = t
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    if verbose:
        print(f"[Data] {nbytes / 1e9:.2f} GB de latents hébergés sur {target}"
              + (" — aucun transfert CPU→GPU pendant le training."
                 if target.type == "cuda" else ""))
    return out, target


def make_adamw(params: Iterable[torch.nn.Parameter], lr: float,
               weight_decay: float, device: DeviceLike) -> torch.optim.AdamW:
    """AdamW avec kernel fused sur CUDA (fallback foreach sinon)."""
    params = list(params)
    use_fused = torch.device(device).type == "cuda"
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                 fused=use_fused)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def maybe_compile(module: torch.nn.Module, enabled: bool,
                  mode: Optional[str] = None) -> torch.nn.Module:
    """
    torch.compile opt-in. Renvoie le module compilé (ou l'original si
    désactivé / indisponible). Sur 4090 avec des shapes fixes : ~1.3-1.8x sur
    le predictor. Le state_dict doit être pris sur le module ORIGINAL (le
    compilé préfixe les clés par `_orig_mod.`) — cf. unwrap().
    """
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        print("[Device] torch.compile indisponible (torch < 2.0) → ignoré.")
        return module
    try:
        compiled = torch.compile(module, mode=mode) if mode else torch.compile(module)
        print(f"[Device] torch.compile activé (mode={mode or 'default'}).")
        return compiled
    except Exception as e:  # pragma: no cover - dépend de l'environnement
        print(f"[Device] torch.compile a échoué ({type(e).__name__}: {e}) → "
              "module non compilé.")
        return module


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Module original derrière torch.compile (clés state_dict sans _orig_mod)."""
    return getattr(module, "_orig_mod", module)


def gpu_summary() -> str:
    """Une ligne décrivant le GPU (pour les logs)."""
    if not torch.cuda.is_available():
        return "aucun GPU (mode CPU)"
    p = torch.cuda.get_device_properties(0)
    return (f"{p.name} | {p.total_memory / 1e9:.0f} GB | sm_{p.major}{p.minor} | "
            f"CUDA {torch.version.cuda} | bf16={'oui' if bf16_supported() else 'non'}")
