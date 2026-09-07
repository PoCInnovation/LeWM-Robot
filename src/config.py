"""
Loader de configuration YAML + utilitaires de reproductibilité.

Usage:
    from src.config import load_config, set_seed

    cfg = load_config("configs/default.yaml")
    set_seed(cfg["seed"])

    encoder_size = cfg["encoder"]["size"]
    dataset_id   = cfg["dataset"]["hf_id"]
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Charge une config YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config introuvable : {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Merge deux configs récursivement (override prend la priorité)."""
    result = dict(base)
    for key, val in override.items():
        if (key in result and isinstance(result[key], dict)
                and isinstance(val, dict)):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = val
    return result


def set_seed(seed: int = 42) -> None:
    """Fixe les graines pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Config] Seed set to {seed}")


def get_device(device: str = "auto") -> torch.device:
    """Résout 'auto' en device disponible."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def log_environment() -> None:
    """Loggue l'environnement (utile pour reproductibilité)."""
    print("=" * 60)
    print("Environnement :")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"  GPU        : {p.name} (sm_{p.major}{p.minor}, "
              f"CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()})")
        print(f"  VRAM       : {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB libre")
        print(f"  bf16       : {torch.cuda.is_bf16_supported()}")
        print(f"  TF32       : matmul={torch.backends.cuda.matmul.allow_tf32} "
              f"cudnn={torch.backends.cudnn.allow_tf32}")
    else:
        print("  GPU        : (aucun, mode CPU)")
    print("=" * 60)


def hardware_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Section `hardware` du YAML avec des défauts sûrs (cible RTX 5090).

    Clés :
        tf32            : bool   — TF32 pour les matmuls fp32 (défaut True)
        cudnn_benchmark : bool   — autotune cuDNN (défaut True)
        precision       : str    — "auto" (bf16 sur GPU compatible) / "bf16" /
                                   "fp16" / "fp32"
        data_device     : str    — "auto" / "cuda" / "cpu" : où héberger les
                                   latents pré-encodés pendant le training
        num_workers     : int|"auto" — workers DataLoader (encodage)
        compile         : bool   — torch.compile du predictor (défaut False)
    """
    defaults = {
        "tf32": True,
        "cudnn_benchmark": True,
        "precision": "auto",
        "data_device": "auto",
        "num_workers": "auto",
        "compile": False,
    }
    hw = dict(defaults)
    hw.update(cfg.get("hardware") or {})
    return hw


def setup_hardware(cfg: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Applique la section `hardware` (TF32, cuDNN) et la renvoie."""
    from src.device import configure_backend
    hw = hardware_config(cfg)
    configure_backend(tf32=bool(hw["tf32"]),
                      cudnn_benchmark=bool(hw["cudnn_benchmark"]),
                      verbose=verbose)
    return hw
