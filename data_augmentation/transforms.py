"""
Primitives d'augmentation partagees entre les deux pipelines du projet.

    OFFLINE  data_augmentation/augment.py   -> ecrit un dataset augmente sur disque
    ONLINE   src/augmentation.py            -> augmente a la volee dans le dataloader

Les deux chemins appellent EXACTEMENT les memes fonctions cv2/numpy : une
augmentation online est donc strictement la meme transformation que son
equivalent offline, aux parametres pres (tires a chaque echantillon en online,
une fois par episode en offline).

Ce module ne depend QUE de numpy + opencv (pas de torch) : il reste importable
depuis l'environnement offline (data_augmentation/requirements.txt).

Convention images : numpy (H, W, 3) uint8 en BGR (convention OpenCV).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


TECHNIQUES: Tuple[str, ...] = ("crop", "blur", "color", "sensor")

IMAGE_TECHNIQUES: Tuple[str, ...] = ("crop", "blur", "color")

APPLY_ORDER: Tuple[str, ...] = ("crop", "blur", "color")

GEOMETRIC_TECHNIQUES: Tuple[str, ...] = ("crop",)

RECIPES: Tuple[Tuple[str, ...], ...] = (
    (),
    ("crop",),
    ("blur",),
    ("color",),
    ("sensor",),
    ("crop", "blur"),
    ("crop", "color"),
    ("crop", "sensor"),
    ("blur", "color"),
    ("blur", "sensor"),
    ("color", "sensor"),
)


@dataclass(frozen=True)
class Strength:
    """
    Amplitude de chaque technique. Les valeurs par defaut sont exactement
    celles du pipeline offline v1 (augment.py) : online et offline appliquent
    donc des transformations de meme force.
    """
    crop_scale_min: float = 0.75
    crop_scale_max: float = 0.95
    blur_ksizes: Tuple[int, ...] = (3, 5, 7)
    blur_sigma_min: float = 0.5
    blur_sigma_max: float = 1.5
    color_brightness: float = 0.2
    color_contrast: float = 0.2
    color_saturation: float = 0.2
    sensor_sigma: float = 0.01

    def scaled(self, intensity: float) -> "Strength":
        """
        Interpole vers l'identite : intensity=1.0 -> valeurs par defaut,
        0.5 -> moitie moins fort, 0.0 -> aucune deformation.
        """
        i = max(0.0, float(intensity))
        return replace(
            self,
            crop_scale_min=1.0 + (self.crop_scale_min - 1.0) * i,
            crop_scale_max=1.0 + (self.crop_scale_max - 1.0) * i,
            blur_sigma_min=self.blur_sigma_min * i,
            blur_sigma_max=self.blur_sigma_max * i,
            color_brightness=self.color_brightness * i,
            color_contrast=self.color_contrast * i,
            color_saturation=self.color_saturation * i,
            sensor_sigma=self.sensor_sigma * i,
        )


@dataclass(frozen=True)
class CropParams:
    top: int
    left: int
    height: int
    width: int
    src_h: int
    src_w: int


def sample_crop(rng: np.random.Generator, height: int, width: int,
                strength: Strength = Strength()) -> CropParams:
    """Tire une boite de recadrage (meme sequence de tirages que augment.py)."""
    s = rng.uniform(strength.crop_scale_min, strength.crop_scale_max)
    ch, cw = int(height * s), int(width * s)
    top = int(rng.integers(0, height - ch + 1))
    left = int(rng.integers(0, width - cw + 1))
    return CropParams(top, left, ch, cw, height, width)


def apply_crop(frame: np.ndarray, p: CropParams) -> np.ndarray:
    """
    Recadre puis re-zoome a la taille d'origine.

    Si l'image n'a pas la resolution pour laquelle la boite a ete tiree (deux
    cameras de definitions differentes), la boite est mise a l'echelle : le
    CADRAGE RELATIF reste alors identique sur les deux cameras.
    """
    h, w = frame.shape[:2]
    if (h, w) == (p.src_h, p.src_w):
        top, left, ch, cw = p.top, p.left, p.height, p.width
    else:
        sy, sx = h / p.src_h, w / p.src_w
        top = max(0, min(h - 1, int(round(p.top * sy))))
        left = max(0, min(w - 1, int(round(p.left * sx))))
        ch = max(1, min(h - top, int(round(p.height * sy))))
        cw = max(1, min(w - left, int(round(p.width * sx))))
    return cv2.resize(frame[top:top + ch, left:left + cw], (w, h))


@dataclass(frozen=True)
class BlurParams:
    ksize: int
    sigma: float


def sample_blur(rng: np.random.Generator,
                strength: Strength = Strength()) -> BlurParams:
    k = int(rng.choice(strength.blur_ksizes))
    sigma = float(rng.uniform(strength.blur_sigma_min, strength.blur_sigma_max))
    return BlurParams(k, sigma)


def apply_blur(frame: np.ndarray, p: BlurParams) -> np.ndarray:
    if p.sigma <= 1e-6:
        return frame
    return cv2.GaussianBlur(frame, (p.ksize, p.ksize), p.sigma)


@dataclass(frozen=True)
class ColorParams:
    brightness: float
    contrast: float
    saturation: float


def sample_color(rng: np.random.Generator,
                 strength: Strength = Strength()) -> ColorParams:
    b, c, s = (strength.color_brightness, strength.color_contrast,
               strength.color_saturation)
    return ColorParams(
        brightness=float(rng.uniform(1 - b, 1 + b)),
        contrast=float(rng.uniform(1 - c, 1 + c)),
        saturation=float(rng.uniform(1 - s, 1 + s)),
    )


def apply_color(frame: np.ndarray, p: ColorParams) -> np.ndarray:
    """Saturation et luminosite en HSV, puis contraste multiplicatif en BGR."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p.saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * p.brightness, 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(bgr * p.contrast, 0, 255).astype(np.uint8)


def sample_sensor_noise(rng: np.random.Generator, shape: Tuple[int, ...],
                        strength: Strength = Strength()) -> np.ndarray:
    """Bruit i.i.d. a ajouter a un vecteur action / observation.state."""
    if strength.sensor_sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    return rng.normal(0.0, strength.sensor_sigma, shape).astype(np.float32)


def apply_sensor_noise(arr: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32) + noise).astype(np.float32)


def sample_image_params(rng: np.random.Generator, techniques: Sequence[str],
                        height: int, width: int,
                        strength: Strength = Strength()) -> Dict[str, object]:
    """
    Tire les parametres des techniques IMAGE d'une recette.

    Le dict retourne est destine a etre REUTILISE tel quel sur toutes les
    images d'un meme echantillon (les deux cameras, les instants t et t+delta)
    afin de garder l'echantillon coherent.
    """
    params: Dict[str, object] = {}
    for name in APPLY_ORDER:
        if name not in techniques:
            continue
        if name == "crop":
            params["crop"] = sample_crop(rng, height, width, strength)
        elif name == "blur":
            params["blur"] = sample_blur(rng, strength)
        elif name == "color":
            params["color"] = sample_color(rng, strength)
    return params


def apply_image_params(frame: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    """Applique des parametres deja tires, dans l'ordre canonique."""
    out = frame
    for name in APPLY_ORDER:
        p = params.get(name)
        if p is None:
            continue
        if name == "crop":
            out = apply_crop(out, p)
        elif name == "blur":
            out = apply_blur(out, p)
        elif name == "color":
            out = apply_color(out, p)
    return out


def enabled_recipes(techniques: Sequence[str],
                    include_original: bool = True
                    ) -> List[Tuple[int, Tuple[str, ...]]]:
    """
    Filtre les 11 recettes selon les techniques activees.

    Returns:
        Liste de (index canonique dans RECIPES, recette). L'index canonique est
        conserve pour pouvoir tracer quelle recette a ete tiree, meme quand
        certaines techniques sont desactivees.
    """
    allowed = set(techniques)
    out: List[Tuple[int, Tuple[str, ...]]] = []
    for i, recipe in enumerate(RECIPES):
        if not recipe and not include_original:
            continue
        if all(t in allowed for t in recipe):
            out.append((i, recipe))
    return out


def sample_recipe(rng: np.random.Generator,
                  recipes: Sequence[Tuple[int, Tuple[str, ...]]]
                  ) -> Tuple[int, Tuple[str, ...]]:
    """Tire uniformement une recette parmi celles activees."""
    if not recipes:
        return 0, ()
    return recipes[int(rng.integers(0, len(recipes)))]


def recipe_label(recipe: Sequence[str]) -> str:
    return " + ".join(recipe) if recipe else "original"
