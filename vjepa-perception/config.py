"""Configuration for the frozen V-JEPA 2 encoder (perception-only path)."""

from __future__ import annotations
from dataclasses import dataclass


_ENCODER_DIMS = {
    "facebook/vjepa2-vitl-fpc64-256": 1024,   # ViT-L  300M
    "facebook/vjepa2-vith-fpc64-256": 1280,   # ViT-H  600M
    "facebook/vjepa2-vitg-fpc64-256": 1408,   # ViT-g  1B
    "facebook/vjepa2-vitg-fpc64-384": 1408,   # ViT-g  1B at 384px
}


@dataclass
class VJepa2EncoderConfig:
    """All you need to load and run the frozen V-JEPA 2 encoder."""

    vjepa2_hf_repo: str = "facebook/vjepa2-vitl-fpc64-256"
    encoder_dim: int = 1024
    encoder_tubelet_t: int = 2          # V-JEPA 2 uses 2-frame tubelets
    patch_size: int = 16
    image_size: int = 256
    encoder_chunk_size: int = 256       # max frames per encoder forward (memory cap)
    temporal_encoding: bool = True      # true temporal encoding by default

    @property
    def n_patches_per_side(self) -> int:
        return self.image_size // self.patch_size  # 256/16 = 16

    @property
    def n_patches(self) -> int:
        return self.n_patches_per_side ** 2        # 256

    def __post_init__(self):
        expected_dim = _ENCODER_DIMS.get(self.vjepa2_hf_repo)
        if expected_dim is not None and expected_dim != self.encoder_dim:
            raise ValueError(
                f"encoder_dim={self.encoder_dim} but {self.vjepa2_hf_repo} has dim "
                f"{expected_dim}. Update encoder_dim in the config."
            )