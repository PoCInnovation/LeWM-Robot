"""
Data loaders pour LeRobot HuggingFace datasets.

Deux datasets PyTorch :
    - LeRobotFramesDataset : une frame par item (pour l'encodage — chaque frame
      n'est encodée qu'UNE fois, les paires/séquences se reconstruisent après).
    - LeRobotPairsDataset  : paires (t, t+delta) par item (legacy, conservé
      pour compatibilité).

Mode offline (nœuds de calcul sans internet) :
    - lerobot charge 100% en local si le dataset est présent sur disque ;
      il ne touche au Hub qu'en cas de fichiers manquants.
    - `dataset_id` peut être un chemin local (dossier du dataset) : aucun
      accès Hub ne sera tenté.
    - Si le dataset est introuvable en mode offline, on échoue immédiatement
      avec un message actionnable (pas de timeout réseau).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.encoders import hf_offline_mode


@dataclass
class LeRobotDataConfig:
    """Configuration du dataset LeRobot HF."""
    dataset_id: str                              # "user/dataset" HF ou chemin local
    wrist_key: str = "observation.images.wrist"
    global_key: str = "observation.images.global"
    action_key: str = "action"
    proprio_key: str = "observation.state"
    delta_timesteps: int = 1                     # steps en avant pour la paire (t, t+delta)
    image_size: int = 224
    cache_dir: Optional[str] = None
    video_backend: Optional[str] = None          # None = auto (torchcodec si dispo, sinon pyav)


def _import_lerobot_dataset_cls():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            return LeRobotDataset
        except ImportError:
            raise ImportError(
                "lerobot n'est pas installé correctement. Installe avec :\n"
                "  pip install 'lerobot>=0.5'"
            )


def resolve_video_backend(requested: Optional[str] = None) -> Optional[str]:
    """
    Choisit le backend de décodage vidéo.

    lerobot utilise torchcodec par défaut, mais torchcodec exige des libs
    ffmpeg système compatibles — souvent absentes/incompatibles sur cluster.
    Si torchcodec ne s'importe pas proprement, on force pyav (pur pip).
    """
    if requested:
        return requested
    try:
        from torchcodec._core import ops as _ops  # noqa: F401
        _ops.load_torchcodec_shared_libraries()
        return None  # torchcodec OK → laisser le défaut lerobot
    except Exception:
        print("[LeRobotData] torchcodec indisponible (libs ffmpeg ?) "
              "→ backend vidéo 'pyav'.")
        return "pyav"


def open_lerobot_dataset(dataset_id: str, cache_dir: Optional[str] = None,
                         video_backend: Optional[str] = None):
    """
    Ouvre un dataset LeRobot depuis le Hub (avec cache) ou un chemin local.

    - Si `dataset_id` est un dossier existant → chargement local direct
      (repo_id = nom du dossier, root = ce dossier). Aucun accès Hub.
    - Sinon → comportement HF standard (cache local, download si absent).

    En mode offline (HF_HUB_OFFLINE=1), tout échec devient une erreur
    immédiate et actionnable.
    """
    LeRobotDataset = _import_lerobot_dataset_cls()

    local_path = Path(dataset_id).expanduser()
    if local_path.is_dir():
        repo_id = local_path.name
        root = local_path
        print(f"[LeRobotData] Dataset local : {local_path}")
    else:
        repo_id = dataset_id
        root = cache_dir
        print(f"[LeRobotData] Dataset Hub : {repo_id} (cache={root or 'défaut'})")

    backend = resolve_video_backend(video_backend)
    try:
        return LeRobotDataset(repo_id, root=root, video_backend=backend)
    except Exception as e:
        offline = hf_offline_mode()
        lines = [
            f"[LeRobotData] Impossible de charger le dataset '{dataset_id}'.",
            f"  Mode offline : {'OUI' if offline else 'non'}",
            f"  Erreur       : {type(e).__name__}: {e}",
            "",
            "  Causes probables :",
            "    - Le dataset n'est pas présent en local (et offline => pas de download).",
            "    - Chemin local incorrect, ou version/format incompatible avec lerobot.",
            "",
            "  Solutions :",
            "    - Sur une machine AVEC internet :",
            f"        huggingface-cli download --repo-type dataset {dataset_id}",
            "      puis pointer dataset.cache_dir (ou HF_LEROBOT_HOME) vers la copie.",
            "    - Ou passer directement le CHEMIN LOCAL du dataset comme dataset_id.",
        ]
        raise RuntimeError("\n".join(lines)) from e


def _episode_index_array(dataset) -> np.ndarray:
    """Renvoie episode_index pour toutes les frames, en numpy (sans décoder les vidéos)."""
    col = dataset.hf_dataset["episode_index"]
    if isinstance(col, torch.Tensor):
        return col.cpu().numpy()
    return np.asarray([int(x) for x in col], dtype=np.int64)


def _to_image(img) -> torch.Tensor:
    """Convertit une image LeRobot vers un tenseur (3, H, W) en float [0, 1]."""
    if isinstance(img, torch.Tensor):
        tensor = img
    else:
        tensor = torch.from_numpy(np.asarray(img))

    # Si (H, W, 3) → (3, H, W)
    if tensor.dim() == 3 and tensor.shape[0] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)

    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0

    return tensor


class LeRobotFramesDataset(Dataset):
    """
    Une FRAME par item — pour l'encodage one-pass (chaque frame encodée 1 fois).

    Returns:
        {
            "wrist":       (3, H, W),
            "global":      (3, H, W),
            "action":      (action_dim,),
            "proprio":     (proprio_dim,),
            "episode_idx": int,
            "frame_idx":   int,   # index GLOBAL dans le dataset
        }

    Expose `episodes` : liste de (episode_idx, global_start, length), dans
    l'ordre du dataset — permet de regrouper les frames encodées par épisode.
    """

    def __init__(self, config: LeRobotDataConfig):
        self.config = config
        self._dataset = open_lerobot_dataset(
            config.dataset_id, config.cache_dir, config.video_backend)

        ep = _episode_index_array(self._dataset)
        self._ep_indices = ep

        # Bornes d'épisodes (frames contiguës par épisode dans LeRobot)
        boundaries = np.flatnonzero(np.diff(ep) != 0) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(ep)]])
        self.episodes = [(int(ep[s]), int(s), int(e - s))
                         for s, e in zip(starts, ends)]

        print(f"[LeRobotFramesDataset] {len(self.episodes)} episodes, "
              f"{len(ep)} frames.")

    def __len__(self) -> int:
        return len(self._ep_indices)

    def __getitem__(self, idx: int) -> dict:
        cfg = self.config
        sample = self._dataset[idx]
        return {
            "wrist":       _to_image(sample[cfg.wrist_key]),
            "global":      _to_image(sample[cfg.global_key]),
            "action":      torch.as_tensor(sample[cfg.action_key], dtype=torch.float32),
            "proprio":     torch.as_tensor(sample[cfg.proprio_key], dtype=torch.float32),
            "episode_idx": int(self._ep_indices[idx]),
            "frame_idx":   int(idx),
        }


class LeRobotPairsDataset(Dataset):
    """
    Paires (frame_t, frame_t+delta, action_t, proprio_t) — legacy.

    NOTE : encode chaque frame ~2 fois quand utilisé pour le pré-encodage ;
    préférer LeRobotFramesDataset + reconstruction des paires en aval.

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
        self._dataset = open_lerobot_dataset(
            config.dataset_id, config.cache_dir, config.video_backend)

        # Index des paires valides, vectorisé : (i, i+delta) dans le même épisode
        delta = config.delta_timesteps
        ep = _episode_index_array(self._dataset)
        if delta >= len(ep):
            raise ValueError(f"delta_timesteps={delta} >= nombre de frames ({len(ep)})")
        valid = np.flatnonzero(ep[:-delta] == ep[delta:])
        self._pairs_index = [(int(ep[i]), int(i)) for i in valid]

        print(f"[LeRobotPairsDataset] {self._dataset.num_episodes} episodes, "
              f"{len(ep)} frames, {len(self._pairs_index)} paires valides "
              f"(delta={delta}).")

    def __len__(self) -> int:
        return len(self._pairs_index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame_idx = self._pairs_index[idx]
        delta = self.config.delta_timesteps
        cfg = self.config

        sample_t  = self._dataset[frame_idx]
        sample_t1 = self._dataset[frame_idx + delta]

        return {
            "wrist_t":     _to_image(sample_t[cfg.wrist_key]),
            "global_t":    _to_image(sample_t[cfg.global_key]),
            "wrist_t1":    _to_image(sample_t1[cfg.wrist_key]),
            "global_t1":   _to_image(sample_t1[cfg.global_key]),
            "action":      torch.as_tensor(sample_t[cfg.action_key], dtype=torch.float32),
            "proprio":     torch.as_tensor(sample_t[cfg.proprio_key], dtype=torch.float32),
            "episode_idx": int(ep_idx),
            "frame_idx":   int(frame_idx),
        }


def make_loader(config: LeRobotDataConfig, batch_size: int = 16,
                num_workers: int = 0, shuffle: bool = True) -> DataLoader:
    """Helper pour construire un DataLoader prêt à l'emploi (paires legacy)."""
    dataset = LeRobotPairsDataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    # Test rapide : remplir avec ton dataset_id Hub ou un chemin local
    import sys
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    else:
        print("Usage: python data.py <huggingface_dataset_id | chemin/local>")
        print("Example: python data.py divisio74/duck_dataset_v3")
        sys.exit(0)

    # NOTE : clés alignées sur configs/default.yaml (dataset duck = "front")
    cfg = LeRobotDataConfig(
        dataset_id=dataset_id,
        global_key="observation.images.front",
    )

    print("\n=== LeRobotFramesDataset ===")
    frames = LeRobotFramesDataset(cfg)
    print(f"Frames: {len(frames)}")
    print(f"Episodes (3 premiers): {frames.episodes[:3]}")
    s = frames[0]
    for key, val in s.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")

    print("\n=== LeRobotPairsDataset (legacy) ===")
    pairs = LeRobotPairsDataset(cfg)
    print(f"Paires: {len(pairs)}")
    sample = pairs[0]
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")
