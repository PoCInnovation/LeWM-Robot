"""
Réglages matériels pour GPU NVIDIA (cible : RTX 5090, Blackwell, 32 GB).

Tout ce qui dépend du device est centralisé ici, pour que les scripts n'aient
qu'à appeler `configure_backend()` en tête puis `resolve_amp_dtype()` :

    - TF32 pour les matmuls/convs fp32 (gratuit sur Ampere+ : ~2-3x sur les
      produits matriciels fp32, précision largement suffisante ici) ;
    - cuDNN benchmark (shapes fixes → autotune des kernels) ;
    - autocast bf16 si le GPU le supporte (Ampere+, donc 5090 : oui) ;
    - choix du device pour héberger les latents pré-encodés (VRAM si ça tient :
      selon la mémoire libre mesurée, avec une réserve pour le training) ;
    - nombre de workers DataLoader ;
    - AdamW fused, torch.compile opt-in ;
    - suivi de la VRAM (pic alloué) pour calibrer les batch sizes.
    - plafond de puissance NVIDIA vérifié avant calcul (80 % par défaut).

Reste 100 % fonctionnel sur CPU (tout devient no-op / fp32).
"""

from __future__ import annotations
import atexit
import os
import re
import shutil
import subprocess
from typing import Dict, Iterable, Optional, Tuple, Union

import torch

DeviceLike = Union[str, torch.device]

# Une même commande Python peut appeler setup_hardware() plusieurs fois. On ne
# doit ni réappliquer la limite ni enregistrer plusieurs restaurations atexit.
_POWER_LIMIT_STATES: Dict[int, Tuple[str, float]] = {}


def _find_nvidia_smi() -> Optional[str]:
    """Trouve nvidia-smi sur Linux/WSL et sur une installation Windows standard."""
    executable = shutil.which("nvidia-smi")
    if executable:
        return executable
    if os.name == "nt":
        program_files = os.environ.get("ProgramW6432", r"C:\Program Files")
        candidate = os.path.join(
            program_files, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _run_nvidia_smi(executable: str, args: list[str]) -> str:
    result = subprocess.run(
        [executable, *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "erreur inconnue").strip()
        raise RuntimeError(f"nvidia-smi a échoué : {detail}")
    return result.stdout.strip()


def _query_power_limits(executable: str, gpu_index: int) -> Tuple[float, ...]:
    output = _run_nvidia_smi(executable, [
        "-i", str(gpu_index),
        "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit",
        "--format=csv,noheader,nounits",
    ])
    line = next((line for line in output.splitlines() if line.strip()), "")
    try:
        values = tuple(float(value.strip()) for value in line.split(","))
    except ValueError as exc:
        raise RuntimeError(
            f"Limites de puissance illisibles pour le GPU {gpu_index} : {line!r}"
        ) from exc
    if len(values) != 4:
        raise RuntimeError(
            f"Réponse nvidia-smi inattendue pour le GPU {gpu_index} : {line!r}")
    return values


def _set_power_limit(executable: str, gpu_index: int, watts: float) -> None:
    value = f"{watts:.1f}".rstrip("0").rstrip(".")
    _run_nvidia_smi(executable, ["-i", str(gpu_index), "-pl", value])


def _restore_power_limit(executable: str, gpu_index: int, watts: float) -> None:
    try:
        _set_power_limit(executable, gpu_index, watts)
        print(f"[Device] Limite GPU {gpu_index} restaurée à {watts:.0f} W.")
    except Exception as exc:  # pragma: no cover - exécuté à la fermeture
        print(f"[Device] ATTENTION : restauration de la limite GPU impossible : {exc}")


def configure_power_limit(percent: float = 80.0, gpu_index: int = 0,
                          required: bool = True,
                          restore_at_exit: bool = True,
                          verbose: bool = True) -> Optional[float]:
    """Limite la puissance NVIDIA à ``percent`` du TGP par défaut.

    Le réglage est appliqué avec ``nvidia-smi`` et vérifié avant tout calcul
    lourd. Si ``required`` est vrai, l'entraînement est refusé quand la limite
    ne peut pas être garantie (PowerShell doit généralement être administrateur).
    La limite présente avant le lancement est restaurée à la sortie normale.
    """
    if not torch.cuda.is_available():
        return None
    if not 1 <= float(percent) <= 100:
        raise ValueError("hardware.power_limit_percent doit être entre 1 et 100")
    if gpu_index < 0:
        raise ValueError("hardware.gpu_index doit être positif")
    if gpu_index in _POWER_LIMIT_STATES:
        return _query_power_limits(_POWER_LIMIT_STATES[gpu_index][0], gpu_index)[0]

    try:
        executable = _find_nvidia_smi()
        if not executable:
            raise RuntimeError("nvidia-smi est introuvable")
        current, default, minimum, maximum = _query_power_limits(
            executable, gpu_index)
        target = round(default * float(percent) / 100.0, 1)
        if target < minimum:
            raise RuntimeError(
                f"{percent:g} % du TGP ({target:.1f} W) est inférieur à la "
                f"limite matérielle minimale ({minimum:.1f} W)")
        target = min(target, maximum)

        # Une limite déjà plus stricte est conservée.
        if current > target + 0.5:
            _set_power_limit(executable, gpu_index, target)
            verified, _, _, _ = _query_power_limits(executable, gpu_index)
            if verified > target + 0.5:
                raise RuntimeError(
                    f"limite demandée {target:.1f} W, limite active {verified:.1f} W")
            if restore_at_exit:
                _POWER_LIMIT_STATES[gpu_index] = (executable, current)
                atexit.register(_restore_power_limit, executable, gpu_index, current)
            current = verified

        if verbose:
            print(f"[Device] GPU {gpu_index} limité à {current:.0f} W "
                  f"({percent:g} % du TGP par défaut de {default:.0f} W).")
        return current
    except Exception as exc:
        message = (
            f"Impossible de garantir la limite GPU à {percent:g} % : {exc}. "
            "Sous Windows, lancer PowerShell en administrateur.")
        if required:
            raise RuntimeError(message) from exc
        print(f"[Device] ATTENTION : {message}")
        return None


def validate_cuda_runtime(require_cuda: bool = False) -> None:
    """Refuse un runtime trop ancien pour Blackwell avant de charger les modèles."""
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "GPU CUDA indisponible. Vérifier nvidia-smi et installer les "
                "roues PyTorch CUDA avec make install.")
        return
    if torch.version.hip is not None:
        return
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        return
    def version_pair(value):
        match = re.match(r"(\d+)\.(\d+)", str(value))
        return tuple(map(int, match.groups())) if match else (0, 0)

    if (version_pair(torch.__version__) < (2, 7)
            or version_pair(torch.version.cuda) < (12, 8)):
        raise RuntimeError(
            f"GPU Blackwell sm_{major}{minor} : PyTorch >= 2.7 compilé avec "
            f"CUDA >= 12.8 requis (installé : torch {torch.__version__}, "
            f"CUDA {torch.version.cuda}). Lancer make install ; "
            "les roues cu124/cu126 ne conviennent pas à la RTX 5090.")


def configure_backend(tf32: bool = True, cudnn_benchmark: bool = True,
                      verbose: bool = True) -> None:
    """Active TF32 + cuDNN benchmark sur CUDA. No-op sur CPU."""
    if not torch.cuda.is_available():
        return
    validate_cuda_runtime()
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
    """bf16 natif (Ampere+ ; la 5090 est Blackwell = oui)."""
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
    Workers DataLoader. 'auto' = min(cap, cœurs - 2).
    Ajuster selon le CPU et le stockage en mesurant le débit d’encodage.
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
    désactivé / indisponible). Le gain doit être mesuré sur la machine cible. Le state_dict doit être pris sur le module ORIGINAL (le
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
