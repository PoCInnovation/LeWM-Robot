"""Behavior-cloning policy head over V-JEPA 2 features.

Mean-pools spatial patch features, optionally concatenates proprioceptive state,
and predicts the action with a small MLP. Frame-wise: no temporal context.
Designed to validate the perception pipeline end-to-end on cached features.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MeanPoolMLPHead(nn.Module):
    """Frame-wise action regressor.

    Args:
        encoder_dim: D in (P, P, D) features.
        state_dim:   proprioceptive state size, or None to ignore state.
        action_dim:  output size.
        hidden_dim:  MLP hidden width.

    Forward:
        features: (B, P, P, D) — patch features from the frozen encoder.
        state:    (B, state_dim) or None.

    Returns:
        action:   (B, action_dim).
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        state_dim: int | None = 8,
        action_dim: int = 7,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.use_state = state_dim is not None
        in_dim = encoder_dim + (state_dim or 0)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        pooled = features.flatten(1, 2).mean(dim=1).float()  # (B, D)
        if self.use_state:
            if state is None:
                raise ValueError("state required: head was built with state_dim != None")
            pooled = torch.cat([pooled, state.float()], dim=-1)
        return self.mlp(pooled)
