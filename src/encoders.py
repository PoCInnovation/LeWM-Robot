"""
Wrapper DINOv3 encoder.

Charge DINOv3 depuis HuggingFace, le passe en mode eval, gèle les poids,
et fournit une méthode `encode(images)` qui retourne les patches features.

Utilisé en :
    - Phase 1 (training)  : encoder les frames pour entraîner le predictor + LoRA
    - Phase 2 (inference) : encoder l'image courante et l'image-goal

GPU NVIDIA (RTX 5090) :
    - poids chargés en bf16 (dtype="auto") : VRAM /2, débit x2-3, aucune
      perte mesurable sur les latents (les sorties sont renvoyées en fp32) ;
    - attention SDPA (kernels SDPA selon le matériel) quand transformers le permet ;
    - le preprocessing (uint8 → float, resize, normalisation ImageNet) se fait
      SUR le GPU : on transfère les images brutes (4x moins d'octets en uint8)
      et on évite un aller-retour CPU.

Note : DINOv3 émet des register tokens ([CLS, reg×4, patches]) et DINOv2 a
des patches de 14 px (256 patches à 224). Les deux sont lus depuis la config
du modèle chargé ; ne rien hardcoder (cf. encoder.num_patches).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel

from src.device import resolve_model_dtype


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

_DTYPE_ALIASES = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
}


@dataclass
class DINOv3Config:
    """Configuration de l'encodeur (DINOv3 ou DINOv2)."""
    size: str = "base"               # "small" / "base" / "large" / "giant"
    family: str = "dinov3"           # "dinov3" (gated) / "dinov2" (public, fallback)
    device: str = "auto"             # "auto" / "cpu" / "cuda"
    # dtype des POIDS : "auto" = bf16 sur GPU compatible (5090), fp32 sinon.
    # Accepte aussi un torch.dtype ou "float32"/"bfloat16"/"float16".
    dtype: Union[str, torch.dtype] = "auto"
    image_size: int = 224
    # patch_size nominal — la valeur RÉELLE est lue depuis le modèle chargé
    # (DINOv2 = 14, DINOv3 = 16). Utiliser encoder.num_patches.
    patch_size: int = 16
    # "sdpa" = flash/mem-efficient attention via torch (recommandé sur 5090),
    # "eager" = implémentation de référence. None = défaut transformers.
    attn_implementation: Optional[str] = "sdpa"

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
        """Nominal (patch_size de la dataclass) — préférer encoder.num_patches."""
        return (self.image_size // self.patch_size) ** 2

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def resolve_dtype(self) -> torch.dtype:
        if isinstance(self.dtype, torch.dtype):
            return self.dtype
        key = str(self.dtype).lower().replace("torch.", "")
        if key in _DTYPE_ALIASES and key not in ("auto",):
            # dtype explicite : respecté tel quel, sauf bf16/fp16 sur CPU → fp32
            dt = _DTYPE_ALIASES[key]
            if dt != torch.float32 and self.resolve_device().type != "cuda":
                print(f"[DINOv3Encoder] dtype {key} demandé sur CPU → float32.")
                return torch.float32
            return dt
        return resolve_model_dtype(self.resolve_device(), key)


def _load_model(model_id: str, dtype: torch.dtype,
                attn_implementation: Optional[str] = None):
    """
    Charge le modèle HF. Si `attn_implementation` n'est pas supporté par
    cette architecture / version de transformers, retombe sur le défaut.
    Erreur actionnable si le modèle est introuvable (gated / hors-ligne).
    """
    kwargs = dict(torch_dtype=dtype)
    if attn_implementation:
        try:
            return AutoModel.from_pretrained(
                model_id, attn_implementation=attn_implementation, **kwargs)
        except (ValueError, TypeError) as e:
            print(f"[DINOv3Encoder] attn_implementation={attn_implementation!r} "
                  f"non supporté ({type(e).__name__}) → défaut transformers.")
    try:
        return AutoModel.from_pretrained(model_id, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"[DINOv3Encoder] Impossible de charger '{model_id}'.\n"
            f"  HF_HOME : {os.environ.get('HF_HOME', '<défaut ~/.cache/huggingface>')}\n"
            "  Causes probables : pas de réseau et modèle absent du cache, ou\n"
            "  (DINOv3) modèle gated : accès Meta + `huggingface-cli login` requis.\n"
            f"  Pré-télécharger : huggingface-cli download {model_id}"
        ) from e


class DINOv3Encoder(nn.Module):
    """
    DINOv3 frozen encoder.

    Usage:
        config = DINOv3Config(size="base")
        encoder = DINOv3Encoder(config)

        # images: (B, 3, 224, 224) en float [0, 1] ou uint8 [0, 255]
        patches = encoder.encode(images)
        # patches: (B, num_patches, dim) en float32
    """

    def __init__(self, config: DINOv3Config):
        super().__init__()
        self.config = config
        self.device_ = config.resolve_device()
        self.dtype_ = config.resolve_dtype()

        print(f"[DINOv3Encoder] Loading {config.model_id} on {self.device_} "
              f"({str(self.dtype_).replace('torch.', '')})...")
        # NOTE : pas d'AutoImageProcessor — le preprocessing (resize + norm
        # ImageNet) est fait dans _preprocess(), sur le device.
        self.model = _load_model(config.model_id, self.dtype_,
                                 config.attn_implementation)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device_)

        # DINOv3 a des register tokens : sortie = [CLS, registres..., patches].
        # DINOv2 n'en a pas : sortie = [CLS, patches].
        self.num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0) or 0)
        # patch_size RÉEL du modèle (DINOv2 = 14, DINOv3 = 16)
        self.patch_size = int(
            getattr(self.model.config, "patch_size", config.patch_size))

        # Normalisation ImageNet en buffers (créés une fois, sur le device)
        self.register_buffer(
            "_norm_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device_),
            persistent=False)
        self.register_buffer(
            "_norm_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1).to(self.device_),
            persistent=False)

        # Propriétés cachées
        self._embed_dim: Optional[int] = None
        attn = getattr(self.model.config, "_attn_implementation", None)
        print(f"[DINOv3Encoder] Loaded. embed_dim={self.embed_dim}, "
              f"num_patches={self.num_patches}, patch_size={self.patch_size}, "
              f"register_tokens={self.num_register_tokens}, attn={attn or '?'}")

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

    @torch.inference_mode()
    def encode(self, images: torch.Tensor, return_cls: bool = False,
               out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Encode un batch d'images vers leurs features par patch.

        Args:
            images: (B, 3, H, W) en float [0, 1] ou uint8 [0, 255], CPU ou GPU
            return_cls: si True, renvoie aussi le CLS token (token global)
            out_dtype: dtype de sortie (fp32 par défaut : les modules aval —
                       fusion, predictor — sont en fp32)

        Returns:
            patches: (B, num_patches, embed_dim)
            cls:     (B, embed_dim) si return_cls=True
        """
        images = self._preprocess(images)          # sur self.device_, fp32
        images = images.to(self.dtype_)

        outputs = self.model(pixel_values=images)
        hidden = outputs.last_hidden_state  # (B, 1 + n_reg + num_patches, dim)

        cls_token = hidden[:, 0].to(out_dtype)
        # Retirer CLS + register tokens éventuels : ne garder que les patches
        patches = hidden[:, 1 + self.num_register_tokens:].to(out_dtype)

        if return_cls:
            return patches, cls_token
        return patches

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalise les images au format attendu par DINOv3.

        Transfert vers le device AVANT toute conversion : en uint8, c'est
        4x moins d'octets sur le bus PCIe, et le resize/normalisation
        tournent sur le GPU.
        """
        images = images.to(self.device_, non_blocking=True)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.dtype != torch.float32:
            images = images.float()

        # Resize si nécessaire
        H, W = images.shape[-2:]
        if H != self.config.image_size or W != self.config.image_size:
            images = nn.functional.interpolate(
                images,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return (images - self._norm_mean) / self._norm_std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


if __name__ == "__main__":
    # Sanity check : charger l'encodeur et encoder un batch dummy
    import sys
    family = sys.argv[1] if len(sys.argv) > 1 else "dinov3"
    size = sys.argv[2] if len(sys.argv) > 2 else "small"
    config = DINOv3Config(family=family, size=size)
    encoder = DINOv3Encoder(config)

    dummy = torch.rand(2, 3, 224, 224)
    patches = encoder.encode(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {patches.shape} {patches.dtype} "
          f"(expected: (2, {encoder.num_patches}, {encoder.embed_dim}))")
    assert patches.shape == (2, encoder.num_patches, encoder.embed_dim), \
        "Nombre de patches inattendu (register tokens mal retirés ?)"

    # uint8 doit donner le même résultat que float
    img = (torch.rand(1, 3, 224, 224) * 255).byte()
    p1 = encoder.encode(img)
    p2 = encoder.encode(img.float() / 255.0)
    tol = 1e-5 if encoder.dtype_ == torch.float32 else 5e-2
    assert torch.allclose(p1, p2, atol=tol), "uint8 vs float mismatch"
    print("OK")
