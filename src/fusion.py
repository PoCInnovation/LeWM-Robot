"""
Stratégies de fusion multi-camera (wrist + global) pour DINOv3 patches.

Chaque stratégie prend en entrée:
    z_wrist:  (B, N_patches, dim)
    z_global: (B, N_patches, dim)

Et retourne une représentation fusionnée que le predictor pourra consommer.

Les stratégies testées:
    1. ConcatSimple        : concat séquentiel sans modification
    2. ConcatWithViewEmb   : concat + embedding "view" pour distinguer les caméras
    3. CrossAttentionBD    : cross-attention bidirectionnelle (mutuelle)
    4. LateFusionCLS       : pooling moyen par caméra + MLP (baseline pauvre)
    5. ProprioGuided       : la proprioception sert de query pour fusionner (avancé)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class ConcatSimple(nn.Module):
    """
    Concat séquentielle des deux caméras.

    Output shape : (B, 2*N, dim)
    Params       : 0
    """
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim
        self.output_num_tokens_multiplier = 2

    def forward(self, z_wrist: torch.Tensor, z_global: torch.Tensor,
                proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.cat([z_wrist, z_global], dim=1)


class ConcatWithViewEmb(nn.Module):
    """
    Concat + embedding appris pour chaque caméra.

    Output shape : (B, 2*N, dim)
    Params       : 2 * dim
    """
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim
        # Petit embedding pour chaque vue, init petite
        self.view_emb_wrist  = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.view_emb_global = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.output_num_tokens_multiplier = 2

    def forward(self, z_wrist: torch.Tensor, z_global: torch.Tensor,
                proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_w = z_wrist  + self.view_emb_wrist
        z_g = z_global + self.view_emb_global
        return torch.cat([z_w, z_g], dim=1)


class CrossAttentionBidirectional(nn.Module):
    """
    Cross-attention dans les deux sens : wrist attends sur global ET global attends sur wrist.

    Output shape : (B, 2*N, dim)
    Params       : ~ 8 * dim²   (deux blocs d'attention + normes)
    """
    def __init__(self, dim: int = 768, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.output_num_tokens_multiplier = 2

        self.attn_w_to_g = nn.MultiheadAttention(dim, n_heads,
                                                  dropout=dropout, batch_first=True)
        self.attn_g_to_w = nn.MultiheadAttention(dim, n_heads,
                                                  dropout=dropout, batch_first=True)
        self.norm_w = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(dim)

    def forward(self, z_wrist: torch.Tensor, z_global: torch.Tensor,
                proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Wrist enrichi par global
        z_w_enriched, _ = self.attn_w_to_g(z_wrist, z_global, z_global)
        z_w = self.norm_w(z_wrist + z_w_enriched)

        # Global enrichi par wrist
        z_g_enriched, _ = self.attn_g_to_w(z_global, z_wrist, z_wrist)
        z_g = self.norm_g(z_global + z_g_enriched)

        return torch.cat([z_w, z_g], dim=1)


class LateFusionCLS(nn.Module):
    """
    Mean pooling par caméra puis MLP de fusion.

    Output shape : (B, 1, dim)
    Params       : ~ 4 * dim²
    """
    def __init__(self, dim: int = 768, hidden: int = 1024):
        super().__init__()
        self.dim = dim
        self.output_num_tokens_multiplier = 1.0 / 196  # un seul token au final

        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_wrist: torch.Tensor, z_global: torch.Tensor,
                proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled_w = z_wrist.mean(dim=1)
        pooled_g = z_global.mean(dim=1)
        fused = self.mlp(torch.cat([pooled_w, pooled_g], dim=-1))  # (B, dim)
        return fused.unsqueeze(1)  # (B, 1, dim)


class ProprioGuidedFusion(nn.Module):
    """
    La proprioception sert de query pour piocher dans les deux caméras.

    L'idée : connaître les angles des servos guide quelle partie de l'image regarder.
    Plus original, peut être un angle de papier.

    Output shape : (B, 1 + 2*N, dim)
    """
    def __init__(self, dim: int = 768, action_dim: int = 6, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.output_num_tokens_multiplier = 2  # approximation

        self.proprio_proj = nn.Linear(action_dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_wrist: torch.Tensor, z_global: torch.Tensor,
                proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert proprio is not None, "ProprioGuidedFusion needs proprioception"
        # (B, action_dim) → (B, 1, dim)
        proprio_token = self.proprio_proj(proprio).unsqueeze(1)

        # Toutes les patches dispo
        all_patches = torch.cat([z_wrist, z_global], dim=1)

        # Le proprio "interroge" toutes les patches
        relevant, _ = self.cross_attn(proprio_token, all_patches, all_patches)
        relevant = self.norm(relevant)

        # Combinaison finale : proprio_relevant + tous les patches
        return torch.cat([relevant, z_wrist, z_global], dim=1)


# Registre des stratégies pour facile expérimentation
FUSION_STRATEGIES = {
    "concat":          ConcatSimple,
    "concat_view":     ConcatWithViewEmb,
    "cross_attn_bd":   CrossAttentionBidirectional,
    "late_cls":        LateFusionCLS,
    "proprio_guided":  ProprioGuidedFusion,
}


def make_fusion(name: str, dim: int = 768, **kwargs) -> nn.Module:
    """Factory pour instancier une stratégie de fusion par son nom."""
    if name not in FUSION_STRATEGIES:
        raise ValueError(f"Unknown fusion '{name}'. Choose from {list(FUSION_STRATEGIES)}")
    return FUSION_STRATEGIES[name](dim=dim, **kwargs)


if __name__ == "__main__":
    # Sanity check : toutes les stratégies acceptent les bonnes entrées
    B, N, D = 4, 196, 768
    z_wrist  = torch.randn(B, N, D)
    z_global = torch.randn(B, N, D)
    proprio  = torch.randn(B, 6)

    for name in FUSION_STRATEGIES:
        try:
            kwargs = {"action_dim": 6} if name == "proprio_guided" else {}
            fusion = make_fusion(name, dim=D, **kwargs)
            out = fusion(z_wrist, z_global, proprio)
            n_params = sum(p.numel() for p in fusion.parameters())
            print(f"  {name:18s}  output: {tuple(out.shape)}  params: {n_params}")
        except Exception as e:
            print(f"  {name:18s}  ERROR: {e}")
