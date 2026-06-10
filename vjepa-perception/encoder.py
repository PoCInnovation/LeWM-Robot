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
    def forward(self, frames: torch.Tensor, temporal: bool | None = None) -> torch.Tensor:
        """Encode frames into patch-level features.

        Args:
            frames: (B, T, C, H, W). Already preprocessed with the V-JEPA 2 video processor.
            temporal: If True, uses true temporal encoding by pairing adjacent frames
                      or treating a T=2 input as a single physical clip.
                      If False, duplicates frames temporally to satisfy tubelet_t.
                      If None, defaults to self.config.temporal_encoding.

        Returns:
            features: (B, T_out, P, P, D).
                      If input has T = tubelet_t (e.g. 2), outputs (B, 1, P, P, D) as a single physical clip.
                      If input has T = 1, outputs (B, 1, P, P, D) using duplication.
                      If input has T > 2, outputs (B, T, P, P, D) with sequential/paired clips.
        """
        B, T, C, H, W = frames.shape
        P = self.config.n_patches_per_side
        D = self.config.encoder_dim
        tubelet_t = self.config.encoder_tubelet_t
        use_temporal = temporal if temporal is not None else getattr(self.config, "temporal_encoding", False)

        if T == tubelet_t:
            # Input already has the exact temporal size of the tubelet (e.g. 2 frames).
            # We process it directly as a single physical video clip.
            x = frames.contiguous()
            num_clips = B
        elif T == 1:
            # Single frame fallback: duplicate to satisfy tubelet_t.
            x = frames.expand(-1, tubelet_t, -1, -1, -1).contiguous()
            num_clips = B
        else:
            # We have a sequence of frames of length T.
            if use_temporal:
                if tubelet_t != 2:
                    raise ValueError(f"temporal mode is optimized for tubelet_t=2, got {tubelet_t}")
                # Pair frame t with frame t+1, duplicating the last frame for t = T-1 (optimized via torch.cat)
                next_frames = torch.cat([frames[:, 1:], frames[:, -1:]], dim=1)
                x = torch.stack([frames, next_frames], dim=2)  # (B, T, 2, C, H, W)
                x = x.reshape(B * T, tubelet_t, C, H, W).contiguous()
            else:
                # Classic behavior: duplicate each frame tubelet_t times
                x = frames.reshape(B * T, 1, C, H, W).expand(-1, tubelet_t, -1, -1, -1).contiguous()
            num_clips = B * T

        chunk = self.config.encoder_chunk_size
        outs = []
        for i in range(0, num_clips, chunk):
            out = self.encoder.get_vision_features(x[i : i + chunk])
            outs.append(out)
        feats = torch.cat(outs, dim=0)            # (num_clips, P*P, D)

        if feats.shape[1] != P * P:
            raise RuntimeError(
                f"Expected {P*P} patch tokens per frame, got {feats.shape[1]}. "
                "Check image_size / patch_size vs the HF model resolution."
            )

        if T == tubelet_t or T == 1:
            return feats.reshape(B, 1, P, P, D)
        else:
            return feats.reshape(B, T, P, P, D)