"""
Wrapper DINOv3 encoder.

Charge DINOv3 depuis HuggingFace, le passe en mode eval, gèle les poids,
et fournit une méthode `encode(images)` qui retourne les patches features.

Utilisé en :
    - Phase 1 (training)  : encoder les frames pour entraîner le predictor + LoRA
    - Phase 2 (inference) : encoder l'image courante et l'image-goal

Mode offline (nœuds de calcul sans internet) :
    - Poser HF_HUB_OFFLINE=1 (et TRANSFORMERS_OFFLINE=1) dans le job.
    - Pré-télécharger le modèle sur le nœud de login :
        huggingface-cli download <model_id>
    - Si le modèle n'est pas dans le cache, le chargement échoue immédiatement
      avec un message actionnable (pas de timeout réseau silencieux).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


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

# Normalisation ImageNet (utilisée par DINOv2 et DINOv3)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def hf_offline_mode() -> bool:
    """True si le mode offline HuggingFace est activé via l'environnement."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


@dataclass
class DINOv3Config:
    """Configuration de l'encodeur (DINOv3 ou DINOv2)."""
    size: str = "base"               # "small" / "base" / "large" / "giant"
    family: str = "dinov3"           # "dinov3" (gated) / "dinov2" (public, fallback)
    device: str = "auto"             # "auto" / "cpu" / "cuda"
    dtype: torch.dtype = torch.float32
    image_size: int = 224
    # patch_size nominal — la valeur RÉELLE est lue depuis la config du modèle
    # chargé (DINOv2 = 14, DINOv3 = 16). Utiliser encoder.num_patches.
    patch_size: int = 16

    @property
    def model_id(self) -> str:
        if self.family not in MODEL_FAMILIES:
            raise ValueError(f"Family '{self.family}' must be one of {list(MODEL_FAMILIES)}")
        models = MODEL_FAMILIES[self.family]
        if self.size not in models:
            raise ValueError(f"Size '{self.size}' must be one of {list(models)}")
        return models[self.size]

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def _load_model_failfast(model_id: str, dtype: torch.dtype):
    """
    Charge le modèle HF avec un diagnostic clair en cas d'échec offline.

    En mode offline (HF_HUB_OFFLINE=1), force local_files_only pour échouer
    immédiatement si le cache est incomplet, au lieu d'attendre un timeout.
    """
    offline = hf_offline_mode()
    try:
        return AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=offline,
        )
    except Exception as e:
        lines = [
            f"[DINOv3Encoder] Impossible de charger '{model_id}'.",
            f"  Mode offline : {'OUI' if offline else 'non'} "
            f"(HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '<non posé>')})",
            f"  HF_HOME      : {os.environ.get('HF_HOME', '<défaut ~/.cache/huggingface>')}",
            "",
            "  Causes probables :",
            "    - Le modèle n'est pas dans le cache HF local.",
            "    - (DINOv3) Modèle gated : accès Meta + token HF requis au téléchargement.",
            "",
            "  Solution — sur une machine AVEC internet (ex: nœud de login, via proxy) :",
            f"    huggingface-cli download {model_id}",
            "  puis relancer avec le même HF_HOME.",
        ]
        raise RuntimeError("\n".join(lines)) from e


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
        # NOTE : pas d'AutoImageProcessor — le preprocessing (resize + norm
        # ImageNet) est fait dans _preprocess(). Un chargement de processor
        # serait un appel Hub inutile (point de panne offline gratuit).
        self.model = _load_model_failfast(config.model_id, config.dtype)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device_)

        # DINOv3 a des register tokens : sortie = [CLS, registres..., patches].
        # DINOv2 n'en a pas : sortie = [CLS, patches]. On lit la config du
        # modèle pour slicer correctement dans les deux cas.
        self.num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0) or 0
        )

        # patch_size RÉEL du modèle (DINOv2 = 14, DINOv3 = 16) — le champ
        # de la dataclass n'est qu'un nominal, la vérité vient du modèle.
        self.patch_size = int(
            getattr(self.model.config, "patch_size", config.patch_size)
        )

        # Normalisation ImageNet en buffers (créés une fois, suivent le device)
        self.register_buffer(
            "_norm_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "_norm_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False)
        self._norm_mean = self._norm_mean.to(self.device_)
        self._norm_std = self._norm_std.to(self.device_)

        # Propriétés cachées
        self._embed_dim: Optional[int] = None
        print(f"[DINOv3Encoder] Loaded. embed_dim={self.embed_dim}, "
              f"num_patches={self.num_patches}, "
              f"patch_size={self.patch_size}, "
              f"register_tokens={self.num_register_tokens}")

    @property
    def num_patches(self) -> int:
        """Nombre réel de patches par image (basé sur le patch_size du modèle)."""
        return (self.config.image_size // self.patch_size) ** 2

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
        hidden = outputs.last_hidden_state  # (B, 1 + n_reg + num_patches, dim)

        cls_token = hidden[:, 0]
        # Retirer CLS + register tokens éventuels : ne garder que les patches
        patches = hidden[:, 1 + self.num_register_tokens:]

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

        images = images.to(self._norm_mean.device)
        return (images - self._norm_mean) / self._norm_std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


if __name__ == "__main__":
    # Sanity check : charger l'encodeur et encoder un batch dummy
    import sys
    family = sys.argv[1] if len(sys.argv) > 1 else "dinov2"
    size = sys.argv[2] if len(sys.argv) > 2 else "base"
    config = DINOv3Config(family=family, size=size)
    encoder = DINOv3Encoder(config)

    dummy = torch.rand(2, 3, 224, 224)
    patches = encoder.encode(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {patches.shape}  (expected: (2, {encoder.num_patches}, {encoder.embed_dim}))")
    assert patches.shape == (2, encoder.num_patches, encoder.embed_dim), \
        "Nombre de patches inattendu (register tokens mal retirés ?)"

    # uint8 doit donner le même résultat que float
    img = (torch.rand(1, 3, 224, 224) * 255).byte()
    p1 = encoder.encode(img)
    p2 = encoder.encode(img.float() / 255.0)
    assert torch.allclose(p1, p2, atol=1e-5), "uint8 vs float mismatch"
    print("OK")
