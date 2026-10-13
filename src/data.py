"""
Data loader pour LeRobot HuggingFace datasets.

Charge les 40 démos réelles depuis le Hub et fournit un Dataset PyTorch
qui retourne (wrist_image, global_image, action, proprio, next_wrist, next_global).

Augmentation ON-THE-FLY : si `LeRobotDataConfig.augment` est fourni et activé,
chaque échantillon est augmenté à la volée avec des paramètres tirés au hasard
(cf. src/augmentation.py). Elle ne s'applique QUE quand le dataset est en mode
entraînement — `set_augment(False)` la coupe pour la validation / l'éval.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from src.augmentation import OnlineAugConfig, make_augmentor

IMAGE_KEYS: Tuple[str, ...] = ("wrist_t", "global_t", "wrist_t1", "global_t1")


@dataclass
class LeRobotDataConfig:
    """Configuration du dataset LeRobot HF."""
    dataset_id: str                              # ex: "user/so101_pick_drop_duck"
    wrist_key: str = "observation.images.wrist"
    global_key: str = "observation.images.global"
    action_key: str = "action"
    proprio_key: str = "observation.state"
    delta_timesteps: int = 1                     # combien de steps en avant pour la pair (t, t+delta)
    image_size: int = 224
    cache_dir: Optional[str] = None
    augment: Optional[OnlineAugConfig] = None
    # True : images renvoyées en uint8 [0, 255] au lieu de float [0, 1].
    # 4x moins de RAM dans la file du DataLoader ; l'encodeur accepte l'uint8.
    uint8_images: bool = False
    video_backend: Optional[str] = None          # None = auto (torchcodec si OK, sinon pyav)


def resolve_video_backend(requested: Optional[str] = None) -> Optional[str]:
    """
    Choisit le backend de décodage vidéo.

    lerobot utilise torchcodec par défaut, mais torchcodec exige des libs
    FFmpeg système compatibles — souvent absentes sur une machine louée ou
    un conteneur. Si torchcodec ne se charge pas, on force pyav (pur pip).
    """
    if requested:
        return requested
    try:
        from torchcodec._core import ops as _ops  # noqa: F401
        _ops.load_torchcodec_shared_libraries()
        return None  # torchcodec OK → laisser le défaut lerobot
    except Exception:
        print("[LeRobotDataLoader] torchcodec indisponible (libs FFmpeg ?) "
              "→ backend vidéo 'pyav'.")
        return "pyav"


class LeRobotPairsDataset(Dataset):
    """
    Charge des paires (frame_t, frame_t+delta, action_t, proprio_t) depuis un
    LeRobot dataset HuggingFace.

    Returns:
        {
            "wrist_t":      (3, H, W),
            "global_t":     (3, H, W),
            "wrist_t1":     (3, H, W),
            "global_t1":    (3, H, W),
            "action":       (action_dim,),
            "proprio":      (proprio_dim,),
            "episode_idx":  int,
            "frame_idx":    int,
            "pair_id":      int,   # index de la paire ORIGINALE (stable)
            "is_augmented": int,   # 0 = échantillon propre, 1 = augmenté
            "recipe_idx":   int,   # index de la recette dans transforms.RECIPES
        }
    """

    def __init__(self, config: LeRobotDataConfig):
        self.config = config
        self._dataset = None
        self._pairs_index = []  # liste de (episode_idx, frame_idx) valides
        self._augmentor = make_augmentor(config.augment)
        self._augment_active = self._augmentor is not None
        self._load()

    def set_augment(self, active: bool) -> None:
        """
        Active / désactive l'augmentation online.

        À appeler AVANT de construire le DataLoader : avec num_workers > 0 les
        workers reçoivent une copie du dataset au démarrage de l'itération, un
        changement ultérieur ne leur parviendrait pas.
        """
        self._augment_active = bool(active) and self._augmentor is not None

    @property
    def augment_active(self) -> bool:
        return self._augment_active

    def _load(self) -> None:
        """Charge le dataset depuis HuggingFace."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            try:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            except ImportError:
                raise ImportError(
                    "lerobot n'est pas installé correctement. Installe avec :\n"
                    "  pip install git+https://github.com/huggingface/lerobot.git"
                )

        print(f"[LeRobotDataLoader] Loading {self.config.dataset_id}...")
        self._dataset = LeRobotDataset(
            self.config.dataset_id,
            root=self.config.cache_dir,
            video_backend=resolve_video_backend(self.config.video_backend),
        )

        # Construire l'index des paires valides en passant par episode_index/frame_index
        # API moderne : chaque index global est une frame, on lit son episode_index
        # depuis le tenseur de métadonnées pour ne pas tout charger en RAM.
        delta = self.config.delta_timesteps
        n_total = len(self._dataset)

        # Récupérer episode_index pour TOUTES les frames d'un coup (sans décoder les vidéos)
        # hf_dataset est le dataset HF brut sans les images
        # (une seule lecture de la colonne : l'accès élément par élément à une
        # colonne datasets>=4 coûte des minutes sur ~900k frames)
        ep_indices = torch.as_tensor(
            self._dataset.hf_dataset.with_format(None)["episode_index"]).flatten()

        # Paire valide ssi frame_i et frame_{i+delta} sont dans le même épisode
        starts = torch.nonzero(ep_indices[:-delta] == ep_indices[delta:]).flatten()
        self._pairs_index = list(zip(ep_indices[starts].tolist(), starts.tolist()))

        print(f"[LeRobotDataLoader] {self._dataset.num_episodes} episodes, "
              f"{n_total} frames, {len(self._pairs_index)} paires valides.")

    def __len__(self) -> int:
        return len(self._pairs_index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame_idx = self._pairs_index[idx]
        delta = self.config.delta_timesteps
        cfg = self.config

        sample_t  = self._dataset[frame_idx]
        sample_t1 = self._dataset[frame_idx + delta]

        out = {
            "wrist_t":     self._to_chw(sample_t[cfg.wrist_key]),
            "global_t":    self._to_chw(sample_t[cfg.global_key]),
            "wrist_t1":    self._to_chw(sample_t1[cfg.wrist_key]),
            "global_t1":   self._to_chw(sample_t1[cfg.global_key]),
            "action":      torch.as_tensor(sample_t[cfg.action_key], dtype=torch.float32),
            "proprio":     torch.as_tensor(sample_t[cfg.proprio_key], dtype=torch.float32),
            "episode_idx": int(ep_idx),
            "frame_idx":   int(frame_idx),
            "pair_id":     int(idx),
            "is_augmented": 0,
            "recipe_idx":  0,
        }

        if self._augmentor is not None and self._augment_active:
            out = self._augmentor(out)

        convert = self._to_uint8 if cfg.uint8_images else self._to_float
        for key in IMAGE_KEYS:
            out[key] = convert(out[key])
        return out

    def pair_episode(self, idx: int) -> int:
        """Épisode de la paire `idx`."""
        return self._pairs_index[idx][0]

    def pair_frame(self, idx: int) -> int:
        """Index global de la frame t de la paire `idx`."""
        return self._pairs_index[idx][1]

    def frame_sample(self, frame_idx: int) -> dict:
        """Images des 2 caméras + action/proprio d'UNE frame (sans augmentation)."""
        cfg = self.config
        sample = self._dataset[frame_idx]
        convert = self._to_uint8 if cfg.uint8_images else self._to_float
        return {
            "wrist":   convert(self._to_chw(sample[cfg.wrist_key])),
            "global":  convert(self._to_chw(sample[cfg.global_key])),
            "action":  torch.as_tensor(sample[cfg.action_key], dtype=torch.float32),
            "proprio": torch.as_tensor(sample[cfg.proprio_key], dtype=torch.float32),
        }

    @staticmethod
    def _to_chw(img) -> torch.Tensor:
        """Image LeRobot → tenseur (3, H, W), dtype d'origine conservé."""
        if isinstance(img, torch.Tensor):
            tensor = img
        else:
            import numpy as np
            tensor = torch.from_numpy(np.asarray(img))

        # Si (H, W, 3) → (3, H, W)
        if tensor.dim() == 3 and tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)

        return tensor

    @staticmethod
    def _to_float(tensor: torch.Tensor) -> torch.Tensor:
        """(3, H, W) uint8 [0, 255] ou float → float [0, 1]."""
        if tensor.dtype == torch.uint8:
            return tensor.float() / 255.0
        return tensor

    @staticmethod
    def _to_uint8(tensor: torch.Tensor) -> torch.Tensor:
        """(3, H, W) float [0, 1] ou uint8 → uint8 [0, 255].

        Sans perte pour les frames vidéo (décodées depuis de l'uint8).
        """
        if tensor.dtype == torch.uint8:
            return tensor
        return (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

    def _to_image(self, img) -> torch.Tensor:
        """Convertit une image LeRobot vers un tenseur (3, H, W) en float [0, 1]."""
        return self._to_float(self._to_chw(img))


class LeRobotFramesDataset(Dataset):
    """
    Frames uniques d'un LeRobotPairsDataset (passe propre de l'encodage).

    La frame t+1 d'une paire est la frame t de la suivante : en passant par
    les frames plutôt que par les paires, chaque frame n'est décodée et
    encodée qu'une fois (÷2 sur le temps d'encodage et le stockage).
    """

    def __init__(self, pairs: LeRobotPairsDataset, frames):
        self.pairs = pairs
        self.frames = list(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, k: int) -> dict:
        return self.pairs.frame_sample(self.frames[k])


def split_train_val(data: Dict[str, torch.Tensor], seed: int = 42,
                    val_fraction: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split train/val d'un fichier de latents pré-encodés, PAR ÉPISODE.

    Trois propriétés que le simple `randperm(n)` ne garantit pas :

      1. Split par ÉPISODE (`episode_idx`) : la frame t+1 d'une paire est la
         frame t de la suivante, et deux frames voisines sont quasi
         identiques. Un split par paire met ~96 % des paires de validation à
         côté d'une paire d'entraînement qui partage une frame — la val loss
         ne mesurerait plus la généralisation. On garde donc des épisodes
         entiers en validation.
      2. La VALIDATION ne contient QUE des échantillons non augmentés.
      3. Aucune variante augmentée d'un épisode de validation en train.

    Rétro-compatible : sans `episode_idx`, split par `pair_id` ; sans
    `pair_id` non plus, split aléatoire par ligne.

    Returns:
        (train_idx, val_idx) — deux LongTensor d'indices dans les latents.
    """
    n = len(data["action"])
    generator = torch.Generator().manual_seed(seed)

    group = data.get("episode_idx")
    if group is None:
        group = data.get("pair_id")
    if group is None:
        perm = torch.randperm(n, generator=generator)
        val_size = max(1, n // 5)
        return perm[val_size:], perm[:val_size]

    group = torch.as_tensor(group).flatten()
    is_augmented = torch.as_tensor(
        data.get("is_augmented", torch.zeros(n, dtype=torch.long))).flatten().bool()

    unique = torch.unique(group)
    perm = unique[torch.randperm(len(unique), generator=generator)]
    n_val = max(1, int(round(len(unique) * val_fraction)))
    if len(unique) > 1:
        n_val = min(n_val, len(unique) - 1)       # garder au moins un groupe en train
    in_val = torch.isin(group, perm[:n_val])

    val_idx = torch.nonzero(in_val & ~is_augmented, as_tuple=False).flatten()
    train_idx = torch.nonzero(~in_val, as_tuple=False).flatten()
    return train_idx, val_idx


def describe_split(data: Dict[str, torch.Tensor], train_idx: torch.Tensor,
                   val_idx: torch.Tensor) -> str:
    """Ligne de log résumant le split (et la part d'augmenté en train)."""
    is_aug = data.get("is_augmented")
    if is_aug is None:
        line = f"{len(train_idx)} train / {len(val_idx)} val"
    else:
        is_aug = torch.as_tensor(is_aug).flatten().bool()
        n_aug = int(is_aug[train_idx].sum())
        line = (f"{len(train_idx)} train ({n_aug} augmentés) / "
                f"{len(val_idx)} val (0 augmenté)")
    episodes = data.get("episode_idx")
    if episodes is not None:
        episodes = torch.as_tensor(episodes).flatten()
        line += (f" — épisodes : {len(torch.unique(episodes[train_idx]))} train / "
                 f"{len(torch.unique(episodes[val_idx]))} val")
    return line


def make_loader(config: LeRobotDataConfig, batch_size: int = 16,
                num_workers: int = 0, shuffle: bool = True) -> DataLoader:
    """Helper pour construire un DataLoader prêt à l'emploi."""
    dataset = LeRobotPairsDataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    # Test rapide : remplir avec ton dataset_id Hub
    import sys
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    else:
        print("Usage: python data.py <huggingface_dataset_id>")
        print("Example: python data.py lerobot/so100_pick_place_lego")
        sys.exit(0)

    cfg = LeRobotDataConfig(dataset_id=dataset_id)
    dataset = LeRobotPairsDataset(cfg)
    print(f"Dataset size: {len(dataset)} paires")

    sample = dataset[0]
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")
