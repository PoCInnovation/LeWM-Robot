"""
Wrapper DINOv3 encoder.

Charge DINOv3 depuis HuggingFace, le passe en mode eval, gèle les poids,
et fournit une méthode `encode(images)` qui retourne les patches features.

Utilisé en :
    - Phase 1 (training)  : encoder les frames pour entraîner le predictor + LoRA
    - Phase 2 (inference) : encoder l'image courante et l'image-goal
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


# Identifiants HuggingFace des variantes
DINOV3_MODELS = {
    "small":  "facebook/dinov3-vits16-pretrain-lvd1689m",   # 21M  params (GATED)
    "base":   "facebook/dinov3-vitb16-pretrain-lvd1689m",   # 86M  params (GATED)
    "large":  "facebook/dinov3-vitl16-pretrain-lvd1689m",   # 300M params (GATED)
    "giant":  "facebook/dinov3-vitg16-pretrain-lvd1689m",   # 1.1B params (GATED)
}

# DINOv2 — alternative NON-GATED, utile pour dev en attendant l'accès DINOv3
# C'est ce que le paper DINO-WM original utilise.
DINOV2_MODELS = {
    "small":  "facebook/dinov2-small",       # 21M  params
    "base":   "facebook/dinov2-base",        # 86M  params
    "large":  "facebook/dinov2-large",       # 300M params
    "giant":  "facebook/dinov2-giant",       # 1.1B params
}

MODEL_FAMILIES = {
    "dinov3": DINOV3_MODELS,
    "dinov2": DINOV2_MODELS,
}


@dataclass
class DINOv3Config:
    """Configuration de l'encodeur (DINOv3 ou DINOv2)."""
    size: str = "base"               # "small" / "base" / "large" / "giant"
    family: str = "dinov3"           # "dinov3" (gated) / "dinov2" (public, fallback)
    device: str = "auto"             # "auto" / "cpu" / "cuda"
    dtype: torch.dtype = torch.float32
    image_size: int = 224
    patch_size: int = 16

    @property
    def model_id(self) -> str:
        if self.family not in MODEL_FAMILIES:
            raise ValueError(f"Family '{self.family}' must be one of {list(MODEL_FAMILIES)}")
        models = MODEL_FAMILIES[self.family]
        if self.size not in models:
            raise ValueError(f"Size '{self.size}' must be one of {list(models)}")
        return models[self.size]

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class DINOv3Encoder(nn.Module):
    """
    DINOv3 frozen encoder.

    Usage:
        config = DINOv3Config(size="base")
        encoder = DINOv3Encoder(config)

        # images: (B, 3, 224, 224) en float [0, 1]
        patches = encoder.encode(images)
        # patches: (B, num_patches, dim)
    """

    def __init__(self, config: DINOv3Config):
        super().__init__()
        self.config = config
        self.device_ = config.resolve_device()

        print(f"[DINOv3Encoder] Loading {config.model_id} on {self.device_}...")
        self.processor = AutoImageProcessor.from_pretrained(config.model_id)
        self.model = AutoModel.from_pretrained(config.model_id, torch_dtype=config.dtype)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device_)

        # Propriétés cachées
        self._embed_dim: Optional[int] = None
        print(f"[DINOv3Encoder] Loaded. embed_dim={self.embed_dim}, "
              f"num_patches={config.num_patches}")

    @property
    def embed_dim(self) -> int:
        """Dimension des features par patch."""
        if self._embed_dim is None:
            # Inférer depuis la config du model
            self._embed_dim = self.model.config.hidden_size
        return self._embed_dim

    @torch.no_grad()
    def encode(self, images: torch.Tensor, return_cls: bool = False) -> torch.Tensor:
        """
        Encode un batch d'images vers leurs features par patch.

        Args:
            images: (B, 3, H, W) en float [0, 1] ou uint8 [0, 255]
            return_cls: si True, renvoie aussi le CLS token (token global)

        Returns:
            patches: (B, num_patches, embed_dim)
            cls:     (B, embed_dim) si return_cls=True
        """
        # Normaliser et redimensionner si besoin
        images = self._preprocess(images)
        images = images.to(self.device_, dtype=self.config.dtype)

        outputs = self.model(pixel_values=images)
        hidden = outputs.last_hidden_state  # (B, 1 + num_patches, dim)

        cls_token = hidden[:, 0]
        patches = hidden[:, 1:]

        if return_cls:
            return patches, cls_token
        return patches

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Normalise les images au format attendu par DINOv3."""
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        # Resize si nécessaire
        H, W = images.shape[-2:]
        if H != self.config.image_size or W != self.config.image_size:
            images = nn.functional.interpolate(
                images,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Normalisation ImageNet (utilisée par DINOv3)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean.to(images.device)) / std.to(images.device)

        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


if __name__ == "__main__":
    # Sanity check : charger DINOv3 et encoder un batch dummy
    config = DINOv3Config(size="small")  # 21M params, rapide en CPU
    encoder = DINOv3Encoder(config)

    dummy = torch.rand(2, 3, 224, 224)
    patches = encoder.encode(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {patches.shape}  (expected: (2, 196, {encoder.embed_dim}))")
    print("OK")
