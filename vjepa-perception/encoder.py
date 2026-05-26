"""Frozen V-JEPA 2 encoder wrapper.

Encodes each frame independently. Because V-JEPA 2 uses 2-frame tubelets, we
duplicate each frame temporally to satisfy the tubelet size, then take the
single resulting feature map.

Input:  (B, T, C, H, W) — frames already normalized with the V-JEPA 2 video processor
Output: (B, T, P, P, D) — patch features per frame
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel

from config import VJepa2EncoderConfig


class FrozenVJepa2Encoder(nn.Module):
    def __init__(self, config: VJepa2EncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.vjepa2_hf_repo)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        hf_hidden = getattr(self.encoder.config, "hidden_size", None)
        if hf_hidden is not None and hf_hidden != config.encoder_dim:
            raise ValueError(
                f"Encoder hidden_size mismatch: HF says {hf_hidden}, config says {config.encoder_dim}."
            )

    def train(self, mode: bool = True):
        """Force encoder to stay in eval mode regardless of parent state."""
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode each frame independently.

        Args:
            frames: (B, T, C, H, W). Already preprocessed with the V-JEPA 2 video processor.

        Returns:
            features: (B, T, P, P, D).
        """
        B, T, C, H, W = frames.shape
        P = self.config.n_patches_per_side
        D = self.config.encoder_dim
        tubelet_t = self.config.encoder_tubelet_t

        # Each frame is treated as a tubelet_t-frame clip (duplicated).
        x = frames.reshape(B * T, 1, C, H, W).expand(-1, tubelet_t, -1, -1, -1).contiguous()

        chunk = self.config.encoder_chunk_size
        outs = []
        for i in range(0, B * T, chunk):
            out = self.encoder.get_vision_features(x[i : i + chunk])
            outs.append(out)
        feats = torch.cat(outs, dim=0)            # (B*T, P*P, D)

        if feats.shape[1] != P * P:
            raise RuntimeError(
                f"Expected {P*P} patch tokens per frame, got {feats.shape[1]}. "
                "Check image_size / patch_size vs the HF model resolution."
            )

        return feats.reshape(B, T, P, P, D)