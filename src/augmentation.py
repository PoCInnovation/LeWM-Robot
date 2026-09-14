"""
Augmentation ON-THE-FLY (online) pour le dataloader d'entrainement.

Contrairement a data_augmentation/augment.py (offline, qui ecrit un dataset
augmente sur disque), ce module applique les augmentations A LA VOLEE, par
echantillon, avec des parametres TIRES AU HASARD a chaque appel : le modele ne
revoit jamais exactement la meme variation, et aucune video n'est dupliquee.

Les transformations elles-memes viennent de data_augmentation/transforms.py :
online et offline executent litteralement le meme code cv2.

Garanties de coherence (indispensables sur un dataset robotique) :
  1. Les MEMES parametres sont appliques aux instants t ET t+delta. Sinon le
     world model devrait predire un changement de luminosite/cadrage qui
     n'existe pas dans la dynamique reelle : il apprendrait du bruit.
  2. Les transformations GEOMETRIQUES (crop) sont appliquees avec le meme
     cadrage sur les DEUX cameras (front + wrist). Voir l'avertissement sur
     l'alignement image <-> action dans transforms.py.
  3. L'augmentation n'est appliquee que lorsque le dataset est en mode
     entrainement (LeRobotPairsDataset.set_augment(True)) — jamais en
     validation / eval / inference.

Activation : section `augmentation` de configs/default.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from data_augmentation.transforms import (
    GEOMETRIC_TECHNIQUES,
    IMAGE_TECHNIQUES,
    TECHNIQUES,
    Strength,
    apply_image_params,
    apply_sensor_noise,
    enabled_recipes,
    recipe_label,
    sample_image_params,
    sample_recipe,
    sample_sensor_noise,
)

CAMERA_KEYS: Dict[str, Tuple[str, ...]] = {
    "wrist":  ("wrist_t", "wrist_t1"),
    "global": ("global_t", "global_t1"),
}

SENSOR_KEYS: Tuple[str, ...] = ("action", "proprio")


@dataclass
class OnlineAugConfig:
    """Configuration de l'augmentation online (section `augmentation` du YAML)."""

    enabled: bool = False
    intensity: float = 1.0
    techniques: Tuple[str, ...] = TECHNIQUES
    include_original: bool = True
    share_across_cameras: bool = True
    passes: int = 1
    seed: int = 42
    strength: Strength = field(default_factory=Strength)

    @classmethod
    def from_dict(cls, section: Optional[Dict[str, Any]],
                  seed: int = 42) -> "OnlineAugConfig":
        """Construit la config depuis la section `augmentation` du YAML."""
        section = dict(section or {})
        techniques = section.get("techniques")
        if isinstance(techniques, dict):
            names = tuple(t for t in TECHNIQUES if bool(techniques.get(t, True)))
        elif isinstance(techniques, (list, tuple)):
            names = tuple(t for t in TECHNIQUES if t in techniques)
        else:
            names = TECHNIQUES

        strength_section = dict(section.get("strength") or {})
        base = Strength()
        strength = Strength(**{
            f: strength_section.get(f, getattr(base, f))
            for f in base.__dataclass_fields__
        })
        if isinstance(strength.blur_ksizes, list):
            strength = Strength(**{**strength.__dict__,
                                   "blur_ksizes": tuple(strength.blur_ksizes)})

        return cls(
            enabled=bool(section.get("enabled", False)),
            intensity=float(section.get("intensity", 1.0)),
            techniques=names,
            include_original=bool(section.get("include_original", True)),
            share_across_cameras=bool(section.get("share_across_cameras", True)),
            passes=int(section.get("passes", 1)),
            seed=int(section.get("seed", seed)),
            strength=strength,
        )

    def describe(self) -> str:
        recipes = enabled_recipes(self.techniques, self.include_original)
        geo = [t for t in self.techniques if t in GEOMETRIC_TECHNIQUES]
        lines = [
            f"[Augmentation] enabled={self.enabled}  intensity={self.intensity}",
            f"[Augmentation] techniques : {', '.join(self.techniques) or '(aucune)'}",
            f"[Augmentation] {len(recipes)} recettes : "
            f"{', '.join(recipe_label(r) for _, r in recipes)}",
            f"[Augmentation] params partages entre cameras : {self.share_across_cameras}",
        ]
        if geo:
            lines.append(
                f"[Augmentation] /!\\ technique GEOMETRIQUE active ({', '.join(geo)}) : "
                "le lien image <-> action n'est plus exact (actions non ajustees).")
        return "\n".join(lines)


class OnlineAugmentor:
    """
    Applique une recette tiree au hasard sur un echantillon du dataloader.

    Un echantillon = 4 images (front + wrist, aux instants t et t+delta) plus
    action / proprio. Les parametres sont tires UNE fois par echantillon puis
    partages, conformement aux garanties documentees en tete de module.
    """

    def __init__(self, config: OnlineAugConfig):
        self.config = config
        self.strength = config.strength.scaled(config.intensity)
        self._recipes = enabled_recipes(config.techniques, config.include_original)
        self._rng: Optional[np.random.Generator] = None
        self._rng_pid: Optional[int] = None

    def rng(self) -> np.random.Generator:
        """
        Generateur propre au process courant.

        Avec num_workers > 0 chaque worker doit tirer une sequence differente,
        sinon tous les workers appliquent les memes augmentations. On derive la
        graine de torch.initial_seed() (que PyTorch fait varier par worker ET
        par epoch) + l'id du worker + le pid.

        Note : avec num_workers > 0 l'ordre des tirages depend de
        l'ordonnancement des workers, donc le run n'est pas bit-reproductible.
        Mettre num_workers=0 pour un run strictement reproductible.
        """
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else 0
            entropy = [int(torch.initial_seed()) % (2 ** 63),
                       int(self.config.seed), int(worker_id), int(pid)]
            self._rng = np.random.default_rng(np.random.SeedSequence(entropy))
            self._rng_pid = pid
        return self._rng

    @staticmethod
    def _to_bgr_u8(t: torch.Tensor) -> np.ndarray:
        """(3, H, W) float [0,1] ou uint8, RGB  ->  (H, W, 3) uint8 BGR."""
        arr = t.detach()
        if arr.dtype != torch.uint8:
            arr = (arr.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        rgb = arr.permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray(rgb[:, :, ::-1])

    @staticmethod
    def _from_bgr_u8(a: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        """(H, W, 3) uint8 BGR  ->  (3, H, W) dans le dtype d'origine."""
        rgb = np.ascontiguousarray(a[:, :, ::-1])
        out = torch.from_numpy(rgb).permute(2, 0, 1)
        if like.dtype == torch.uint8:
            return out
        return out.to(like.dtype).div_(255.0)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        rng = self.rng()
        recipe_idx, recipe = sample_recipe(rng, self._recipes)

        sample["recipe_idx"] = int(recipe_idx)
        sample["is_augmented"] = int(bool(recipe))
        if not recipe:
            return sample

        image_techs = [t for t in recipe if t in IMAGE_TECHNIQUES]
        if image_techs:
            self._augment_images(sample, image_techs, rng)
        if "sensor" in recipe:
            self._augment_sensors(sample, rng)
        return sample

    def _augment_images(self, sample: Dict[str, Any],
                        techniques: Sequence[str],
                        rng: np.random.Generator) -> None:
        ref = sample[CAMERA_KEYS["wrist"][0]]
        height, width = int(ref.shape[-2]), int(ref.shape[-1])

        geo_techs = [t for t in techniques if t in GEOMETRIC_TECHNIQUES]
        photo_techs = [t for t in techniques if t not in GEOMETRIC_TECHNIQUES]
        geo_params = sample_image_params(rng, geo_techs, height, width,
                                         self.strength)

        if self.config.share_across_cameras:
            shared = sample_image_params(rng, photo_techs, height, width,
                                         self.strength)
            photo_params = {cam: shared for cam in CAMERA_KEYS}
        else:
            photo_params = {
                cam: sample_image_params(rng, photo_techs, height, width,
                                         self.strength)
                for cam in CAMERA_KEYS
            }

        for cam, keys in CAMERA_KEYS.items():
            params = {**geo_params, **photo_params[cam]}
            if not params:
                continue
            for key in keys:
                tensor = sample[key]
                frame = apply_image_params(self._to_bgr_u8(tensor), params)
                sample[key] = self._from_bgr_u8(frame, tensor)

    def _augment_sensors(self, sample: Dict[str, Any],
                         rng: np.random.Generator) -> None:
        for key in SENSOR_KEYS:
            tensor = sample.get(key)
            if tensor is None:
                continue
            arr = tensor.detach().cpu().numpy()
            noise = sample_sensor_noise(rng, arr.shape, self.strength)
            sample[key] = torch.from_numpy(
                apply_sensor_noise(arr, noise)).to(tensor.dtype)


def make_augmentor(config: Optional[OnlineAugConfig]) -> Optional[OnlineAugmentor]:
    """Renvoie None si l'augmentation est desactivee (chemin sans surcout)."""
    if config is None or not config.enabled or not config.techniques:
        return None
    return OnlineAugmentor(config)
