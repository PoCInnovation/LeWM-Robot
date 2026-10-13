"""
Checkpoints du world model : écriture atomique et rechargement complet.

L'écriture passe par un fichier temporaire puis un `os.replace` : une coupure
de la machine (location, spot) en pleine écriture laisse l'ancien fichier
intact au lieu d'un checkpoint tronqué.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

from src.fusion import make_fusion
from src.latents import ActionStats
from src.lora import LoRAConfig, inject_lora
from src.predictor import PredictorConfig, WorldModelPredictor


def atomic_save(obj, path: Union[str, Path]) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_world_model(ckpt_path: Union[str, Path], device: Union[str, torch.device]
                     ) -> Tuple[nn.Module, nn.Module, dict]:
    """
    Charge predictor (+ LoRA si présent) et fusion d'un checkpoint de 04 / 05.

    Returns:
        predictor, fusion (en eval, sur device), metadata (dont action_stats :
        ActionStats ou None pour un checkpoint antérieur à la normalisation)
    """
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    predictor = WorldModelPredictor(PredictorConfig(**ckpt["predictor_config"]))
    has_lora = "lora_config" in ckpt
    if has_lora:     # injecter d'abord : la state_dict contient les clés A et B
        predictor = inject_lora(predictor, LoRAConfig(**ckpt["lora_config"]), verbose=False)
    predictor.load_state_dict(ckpt["predictor_state_dict"])

    fusion = make_fusion(ckpt["fusion_name"], dim=ckpt["embed_dim"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])

    predictor.to(device).eval()
    fusion.to(device).eval()

    metadata = {
        "fusion_name":  ckpt["fusion_name"],
        "embed_dim":    ckpt["embed_dim"],
        "action_dim":   ckpt["action_dim"],
        "has_lora":     has_lora,
        "epoch":        ckpt.get("epoch", "?"),
        "val_loss":     ckpt.get("val_loss", "?"),
        "action_stats": (ActionStats.from_dict(ckpt["action_stats"])
                         if "action_stats" in ckpt else None),
    }
    return predictor, fusion, metadata
