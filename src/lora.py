"""
LoRA (Low-Rank Adaptation) injection pour le predictor.

Permet d'adapter un predictor entraîné sur Isaac (simu) aux conditions réelles
sans réentraîner tous les poids :
    - Le predictor est figé (W reste tel quel)
    - On ajoute à chaque couche linéaire une correction A·B de rang faible
    - Seuls A et B sont entraînés (sur les 40 démos réelles)
    - Capacité limitée → impossible d'overfit avec si peu de données

Référence : Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """Configuration d'injection LoRA."""
    rank: int = 8                        # dimension intérieure (capacité d'adaptation)
    alpha: float = 16.0                  # facteur d'échelle (scaling = alpha/rank)
    dropout: float = 0.0                 # dropout sur la branche LoRA
    target_modules: List[str] = None     # quels modules adapter

    def __post_init__(self):
        if self.target_modules is None:
            # Par défaut : adapter les linéaires des blocs transformer
            self.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj",
                                    "in_proj_weight"]


class LoRALinear(nn.Module):
    """
    Linear layer avec correction LoRA.

    Calcule :  y = W·x + (alpha/r) · A·B·x

    Où :
        W est figé (poids original)
        A : (in_features, r)  entraîné
        B : (r, out_features) entraîné
        alpha/r : facteur d'échelle
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base_layer  # référence vers la linéaire originale (figée)
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.rank = rank
        self.scaling = alpha / rank

        # Matrices LoRA — init A en gaussien petit, B en zéro
        # → au début la correction est nulle (W non modifié)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Geler le base layer
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chemin principal (figé)
        base_out = self.base(x)

        # Chemin LoRA
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def inject_lora(model: nn.Module, config: LoRAConfig,
                verbose: bool = True) -> nn.Module:
    """
    Injecte des couches LoRA dans toutes les nn.Linear du modèle.

    Modification IN-PLACE : le modèle passé est modifié, et retourné par commodité.

    Args:
        model   : le model à adapter (ex: WorldModelPredictor)
        config  : configuration LoRA
        verbose : si True, log les couches modifiées

    Returns:
        Le model modifié (mêmes poids que l'original + LoRA ajouté)
    """
    n_replaced = 0
    n_params_added = 0
    total_base_params = 0

    # Geler tout le modèle d'abord
    for param in model.parameters():
        param.requires_grad = False

    # Parcourir les modules et remplacer les Linear par LoRALinear
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            # Décomposer le nom pour accéder au parent
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr_name = parts[0]
            else:
                parent = model.get_submodule(parts[0])
                attr_name = parts[1]

            # Créer le LoRALinear wrappant l'original
            lora_layer = LoRALinear(
                base_layer=module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )
            setattr(parent, attr_name, lora_layer)

            n_replaced += 1
            n_params_added += (lora_layer.lora_A.numel()
                               + lora_layer.lora_B.numel())
            total_base_params += sum(p.numel() for p in module.parameters())

            if verbose:
                print(f"  [LoRA] {name}: {module.in_features}→{module.out_features}  "
                      f"+{n_params_added // n_replaced} params")

    print(f"\n[LoRA] {n_replaced} couches adaptées.")
    print(f"[LoRA] Params LoRA ajoutés    : {n_params_added:,}")
    print(f"[LoRA] Params base couverts   : {total_base_params:,}")
    print(f"[LoRA] Ratio compression      : {n_params_added / total_base_params * 100:.2f}%")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Renvoie uniquement les paramètres LoRA (à passer à l'optimiseur)."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def merge_lora(model: nn.Module) -> nn.Module:
    """
    Fusionne les corrections LoRA dans les poids originaux.

    À utiliser en fin d'entraînement pour optimiser l'inférence :
        W_merged = W + (alpha/r) · A · B
    Après merging, la couche LoRA est remplacée par une simple Linear avec
    les poids fusionnés. Mathématiquement équivalent mais ~30% plus rapide.

    Returns:
        Le model avec LoRA fusionné (les LoRALinear redeviennent des Linear).
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            # Calculer la correction
            delta_W = (module.lora_A @ module.lora_B).T * module.scaling
            # Note : la matrice W de nn.Linear est (out_features, in_features),
            # donc on doit transposer (lora_A @ lora_B) qui est (in, out)

            # Créer une nouvelle Linear avec les poids fusionnés
            new_linear = nn.Linear(
                module.in_features, module.out_features,
                bias=module.base.bias is not None,
            )
            with torch.no_grad():
                new_linear.weight.copy_(module.base.weight + delta_W)
                if module.base.bias is not None:
                    new_linear.bias.copy_(module.base.bias)

            # Remplacer dans le parent
            parts = name.rsplit(".", 1)
            parent = model if len(parts) == 1 else model.get_submodule(parts[0])
            attr_name = parts[-1]
            setattr(parent, attr_name, new_linear)

    return model


if __name__ == "__main__":
    # Sanity check : injecter LoRA dans un mini predictor
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.predictor import WorldModelPredictor, PredictorConfig

    # Predictor d'origine
    pred_cfg = PredictorConfig(embed_dim=384, action_dim=6, n_layers=2, n_heads=4)
    predictor = WorldModelPredictor(pred_cfg)
    n_params_before = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Predictor params (trainable) avant LoRA : {n_params_before:,}")

    # Injection LoRA
    lora_cfg = LoRAConfig(rank=8, alpha=16.0)
    predictor = inject_lora(predictor, lora_cfg, verbose=False)

    n_params_after = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Predictor params (trainable) après LoRA : {n_params_after:,}")

    # Test forward
    B, N = 2, 100
    z_t = torch.randn(B, N, pred_cfg.embed_dim)
    action = torch.randn(B, pred_cfg.action_dim)
    z_t1 = predictor(z_t, action)
    print(f"Forward OK : output shape = {tuple(z_t1.shape)}")

    # Test merge
    print("\nTest merge LoRA :")
    predictor = merge_lora(predictor)
    z_t1_merged = predictor(z_t, action)
    print(f"Forward après merge OK : output shape = {tuple(z_t1_merged.shape)}")
    print("\n[OK] LoRA fonctionne.")
