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
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  GPU        : (aucun, mode CPU)")
    print("=" * 60)
