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
    """MLP that maps:
       (current-frame features, future actions) → future-frame (mean-pooled) features.

    Forward:
        features: (B, P, P, D) — current-frame patch features.
        actions:  (B, H, action_dim) — actions from t to t+H-1 (optional, required if action_dim is set).

    Returns:
        pred:     (B, D)       — predicted mean-pooled features H steps ahead.
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        action_dim: int | None = None,
        horizon: int | None = None,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.use_actions = action_dim is not None and horizon is not None

        if self.use_actions:
            # Project the flattened action sequence (horizon * action_dim) to a smaller feature representation
            self.action_flat_dim = horizon * action_dim
            self.action_proj = nn.Sequential(
                nn.Linear(self.action_flat_dim, hidden_dim // 2),
                nn.GELU(),
            )
            in_dim = encoder_dim + (hidden_dim // 2)
        else:
            in_dim = encoder_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, encoder_dim),
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        pooled = features.flatten(1, 2).mean(dim=1).float()  # (B, D)

        if self.use_actions:
            if actions is None:
                raise ValueError("actions are required for an action-conditioned predictor!")
            
            # Flatten action sequence: (B, H, action_dim) -> (B, H * action_dim)
            actions_flat = actions.flatten(1, 2).float()
            
            # Verify shape of flattened actions
            if actions_flat.shape[1] != self.action_flat_dim:
                raise ValueError(
                    f"Expected flattened action dimension {self.action_flat_dim} "
                    f"(horizon={self.horizon} * action_dim={self.action_dim}), "
                    f"got {actions_flat.shape[1]}"
                )
            
            # Project actions and concatenate with spatial visual features
            action_feats = self.action_proj(actions_flat)
            x = torch.cat([pooled, action_feats], dim=-1)
        else:
            x = pooled

        return self.mlp(x)
