"""
Data loader pour LeRobot HuggingFace datasets.

Charge les 40 démos réelles depuis le Hub et fournit un Dataset PyTorch
qui retourne (wrist_image, global_image, action, proprio, next_wrist, next_global).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


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
        }
    """

    def __init__(self, config: LeRobotDataConfig):
        self.config = config
        self._dataset = None
        self._pairs_index = []  # liste de (episode_idx, frame_idx) valides
        self._load()

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
        )

        # Construire l'index des paires valides en passant par episode_index/frame_index
        # API moderne : chaque index global est une frame, on lit son episode_index
        # depuis le tenseur de métadonnées pour ne pas tout charger en RAM.
        delta = self.config.delta_timesteps
        n_total = len(self._dataset)

        # Récupérer episode_index pour TOUTES les frames d'un coup (sans décoder les vidéos)
        # hf_dataset est le dataset HF brut sans les images
        ep_indices = self._dataset.hf_dataset["episode_index"]

        for i in range(n_total - delta):
            # Paire valide ssi frame_i et frame_{i+delta} sont dans le même épisode
            if ep_indices[i] == ep_indices[i + delta]:
                self._pairs_index.append((int(ep_indices[i]), i))

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
            "wrist_t":     self._to_image(sample_t[cfg.wrist_key]),
            "global_t":    self._to_image(sample_t[cfg.global_key]),
            "wrist_t1":    self._to_image(sample_t1[cfg.wrist_key]),
            "global_t1":   self._to_image(sample_t1[cfg.global_key]),
            "action":      torch.as_tensor(sample_t[cfg.action_key], dtype=torch.float32),
            "proprio":     torch.as_tensor(sample_t[cfg.proprio_key], dtype=torch.float32),
            "episode_idx": int(ep_idx),
            "frame_idx":   int(frame_idx),
        }
        return out

    def _to_image(self, img) -> torch.Tensor:
        """Convertit une image LeRobot vers un tenseur (3, H, W) en float [0, 1]."""
        if isinstance(img, torch.Tensor):
            tensor = img
        else:
            import numpy as np
            tensor = torch.from_numpy(np.asarray(img))

        # Si (H, W, 3) → (3, H, W)
        if tensor.dim() == 3 and tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)

        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0

        return tensor


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
