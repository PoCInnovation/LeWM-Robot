"""Future-feature predictor over V-JEPA 2 features.

Predicts the mean-pooled V-JEPA 2 representation of a frame H steps ahead, given
the current frame's patch features. Trained with MSE on cached features. Used
for the "predict the next frame" demo via nearest-neighbor retrieval against
the training-set feature pool.

Single-frame conditioning, no temporal context — the simplest setup that still
demonstrates V-JEPA features carry predictive information.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class FutureFeaturePredictor(nn.Module):
    """MLP that maps current-frame features → future-frame (mean-pooled) features.

    Forward:
        features: (B, P, P, D) — current-frame patch features.

    Returns:
        pred:     (B, D)       — predicted mean-pooled features H steps ahead.
    """

    def __init__(self, encoder_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, encoder_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.flatten(1, 2).mean(dim=1).float()
        return self.mlp(pooled)
