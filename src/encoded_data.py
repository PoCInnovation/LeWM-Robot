"""
Format des données pré-encodées (v2) + utilitaires de chargement.

Layout sur disque (un DOSSIER par dataset encodé) :
    results/encoded/<name>/
        ep_00000.pt          # un fichier par épisode (~50-350 MB : Lustre-friendly)
        ep_00001.pt
        ...
        shard_meta_0.json    # méta + stats partielles écrites par chaque shard
        shard_meta_1.json
        meta.json            # écrit par le merge (ou auto si num_shards=1)

Chaque ep_XXXXX.pt contient :
    {
        "episode_idx": int,
        "z_wrist":  (T, N, D) float16 (ou float32 avec --fp32),
        "z_global": (T, N, D),
        "action":   (T, A) float32,   # action_t (alignée sur la frame t)
        "proprio":  (T, P) float32,
        "frame_start": int,           # index global de la 1re frame
    }

meta.json contient embed_dim, num_patches, encoder, dataset, la liste des
épisodes, et les STATS DE NORMALISATION action/proprio (mean/std/min/max) —
indispensables pour entraîner en actions normalisées et pour que le CEM
échantillonne dans le bon espace.

Pourquoi des séquences par épisode (vs paires en vrac) :
    - chaque frame encodée UNE fois (l'ancien format encodait tout 2x) ;
    - delta_timesteps choisi AU TRAINING (pas figé à l'encodage) ;
    - fenêtres multi-step possibles (rollout autorégressif).
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# ────────────────────────────────────────────────────────────────
# Stats de normalisation
# ────────────────────────────────────────────────────────────────

@dataclass
class NormStats:
    """Stats de normalisation z-score + bornes, par dimension."""
    mean: torch.Tensor      # (D,)
    std: torch.Tensor       # (D,)
    min: torch.Tensor       # (D,)
    max: torch.Tensor       # (D,)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def normalized_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bornes min/max du dataset, exprimées dans l'espace normalisé."""
        return self.normalize(self.min), self.normalize(self.max)

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("mean", "std", "min", "max")}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(**{k: torch.tensor(d[k], dtype=torch.float32)
                      for k in ("mean", "std", "min", "max")})

    @classmethod
    def from_partials(cls, partials: List[dict]) -> "NormStats":
        """Agrège des stats partielles {sum, sumsq, min, max, count} de shards."""
        total = sum(p["count"] for p in partials)
        s = torch.stack([torch.tensor(p["sum"]) for p in partials]).sum(0)
        sq = torch.stack([torch.tensor(p["sumsq"]) for p in partials]).sum(0)
        mn = torch.stack([torch.tensor(p["min"]) for p in partials]).min(0).values
        mx = torch.stack([torch.tensor(p["max"]) for p in partials]).max(0).values
        mean = s / total
        var = (sq / total - mean ** 2).clamp(min=0)
        std = var.sqrt().clamp(min=1e-6)
        return cls(mean=mean.float(), std=std.float(),
                   min=mn.float(), max=mx.float())


def partial_stats(x: torch.Tensor) -> dict:
    """Stats partielles d'un tenseur (T, D), à agréger entre shards."""
    return {
        "sum":   x.sum(0).tolist(),
        "sumsq": (x ** 2).sum(0).tolist(),
        "min":   x.min(0).values.tolist(),
        "max":   x.max(0).values.tolist(),
        "count": int(x.shape[0]),
    }


# ────────────────────────────────────────────────────────────────
# Chargement
# ────────────────────────────────────────────────────────────────

class EncodedEpisodes:
    """
    Charge un dataset encodé v2 (dossier avec meta.json + ep_*.pt).

    Les épisodes sont chargés en RAM (fp16) au premier accès et gardés.
    """

    def __init__(self, path: str | Path):
        self.root = Path(path)
        meta_path = self.root / "meta.json"
        if not meta_path.exists():
            shards = sorted(self.root.glob("shard_meta_*.json"))
            hint = (f"\n  Des shard_meta existent ({len(shards)}) : lancer le merge :"
                    f"\n    python scripts/02_encode_dataset.py --output {self.root} --finalize"
                    if shards else "")
            raise FileNotFoundError(
                f"[EncodedEpisodes] meta.json introuvable dans {self.root}."
                f"{hint}\n  (ancien format .pt unique ? ré-encoder avec le "
                f"nouveau scripts/02_encode_dataset.py)")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.embed_dim: int = self.meta["embed_dim"]
        self.num_patches: int = self.meta["num_patches"]
        self.action_stats = NormStats.from_dict(self.meta["action_stats"])
        self.proprio_stats = NormStats.from_dict(self.meta["proprio_stats"])

        self._episodes_info = self.meta["episodes"]  # [{idx, length, file}]
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self._episodes_info)

    @property
    def episode_indices(self) -> List[int]:
        return [e["idx"] for e in self._episodes_info]

    @property
    def total_frames(self) -> int:
        return sum(e["length"] for e in self._episodes_info)

    def episode(self, i: int) -> dict:
        """Épisode par position (0..len-1). Tenseurs en dtype stocké (fp16)."""
        if i not in self._cache:
            info = self._episodes_info[i]
            self._cache[i] = torch.load(self.root / info["file"],
                                        weights_only=True)
        return self._cache[i]

    def summary(self) -> str:
        return (f"{len(self)} episodes, {self.total_frames} frames, "
                f"embed_dim={self.embed_dim}, num_patches={self.num_patches}, "
                f"encoder={self.meta.get('encoder_family')}-"
                f"{self.meta.get('encoder_size')}")


def split_episodes(n_episodes: int, val_fraction: float,
                   seed: int) -> Tuple[List[int], List[int]]:
    """
    Split train/val PAR ÉPISODE (positions 0..n-1).

    Splitter par paires fuit de l'information : deux frames voisines du même
    épisode sont quasi identiques → val loss optimiste. On isole des épisodes
    entiers en validation.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_episodes, generator=g).tolist()
    n_val = max(1, int(round(n_episodes * val_fraction)))
    if n_episodes > 1:
        n_val = min(n_val, n_episodes - 1)
    return sorted(perm[n_val:]), sorted(perm[:n_val])


class PairsView(Dataset):
    """
    Vue "paires (t, t+delta)" sur des épisodes encodés — pour le training
    one-step. delta est choisi ICI, au training, pas à l'encodage.

    Returns (tenseurs float32) :
        z_wrist_t, z_global_t, z_wrist_t1, z_global_t1 : (N, D)
        action  : (A,)   action_t (normalisée si norm_actions)
        proprio : (P,)   proprio_t (normalisée si norm_proprio)
    """

    def __init__(self, data: EncodedEpisodes, episode_positions: Sequence[int],
                 delta: int = 1, norm_actions: bool = True,
                 norm_proprio: bool = True):
        self.data = data
        self.delta = int(delta)
        self.norm_actions = norm_actions
        self.norm_proprio = norm_proprio
        assert self.delta >= 1

        self.index: List[Tuple[int, int]] = []   # (episode_position, t)
        for pos in episode_positions:
            T = data._episodes_info[pos]["length"]
            for t in range(T - self.delta):
                self.index.append((pos, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        pos, t = self.index[i]
        ep = self.data.episode(pos)
        d = self.delta

        action = ep["action"][t].float()
        proprio = ep["proprio"][t].float()
        if self.norm_actions:
            action = self.data.action_stats.normalize(action)
        if self.norm_proprio:
            proprio = self.data.proprio_stats.normalize(proprio)

        return {
            "z_wrist_t":   ep["z_wrist"][t].float(),
            "z_global_t":  ep["z_global"][t].float(),
            "z_wrist_t1":  ep["z_wrist"][t + d].float(),
            "z_global_t1": ep["z_global"][t + d].float(),
            "action":      action,
            "proprio":     proprio,
        }


class SequenceView(Dataset):
    """
    Vue "fenêtres de longueur H+1" — pour le training multi-step autorégressif.

    Returns (float32) :
        z_wrist, z_global : (H+1, N, D)  frames t..t+H*delta (stride delta)
        actions : (H, A)  actions t..t+(H-1)*delta (normalisées si demandé)
        proprio : (P,)    proprio_t
    """

    def __init__(self, data: EncodedEpisodes, episode_positions: Sequence[int],
                 horizon: int, delta: int = 1, norm_actions: bool = True,
                 norm_proprio: bool = True):
        self.data = data
        self.horizon = int(horizon)
        self.delta = int(delta)
        self.norm_actions = norm_actions
        self.norm_proprio = norm_proprio
        assert self.horizon >= 1 and self.delta >= 1

        span = self.horizon * self.delta
        self.index: List[Tuple[int, int]] = []
        for pos in episode_positions:
            T = data._episodes_info[pos]["length"]
            for t in range(T - span):
                self.index.append((pos, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        pos, t = self.index[i]
        ep = self.data.episode(pos)
        d, H = self.delta, self.horizon
        frame_ts = list(range(t, t + (H + 1) * d, d))          # H+1 frames
        action_ts = list(range(t, t + H * d, d))               # H actions

        actions = ep["action"][action_ts].float()
        proprio = ep["proprio"][t].float()
        if self.norm_actions:
            actions = self.data.action_stats.normalize(actions)
        if self.norm_proprio:
            proprio = self.data.proprio_stats.normalize(proprio)

        return {
            "z_wrist":  ep["z_wrist"][frame_ts].float(),
            "z_global": ep["z_global"][frame_ts].float(),
            "actions":  actions,
            "proprio":  proprio,
        }


def merge_shard_metas(output_dir: str | Path) -> dict:
    """
    Fusionne les shard_meta_*.json d'un dossier encodé en meta.json final.

    Vérifie la cohérence (embed_dim, encodeur, num_shards attendus) et agrège
    les stats de normalisation. Renvoie le meta dict écrit.
    """
    output_dir = Path(output_dir)
    shard_files = sorted(output_dir.glob("shard_meta_*.json"))
    if not shard_files:
        raise FileNotFoundError(f"Aucun shard_meta_*.json dans {output_dir}")

    shards = []
    for f in shard_files:
        with open(f) as fh:
            shards.append(json.load(fh))

    num_shards = shards[0]["num_shards"]
    got = sorted(s["shard_index"] for s in shards)
    if got != list(range(num_shards)):
        raise RuntimeError(
            f"Shards incomplets : attendu 0..{num_shards - 1}, trouvé {got}. "
            f"(des jobs de l'array ont échoué ?)")

    for key in ("embed_dim", "num_patches", "encoder_family", "encoder_size",
                "dataset_id", "image_size"):
        vals = {json.dumps(s[key]) for s in shards}
        if len(vals) > 1:
            raise RuntimeError(f"Incohérence entre shards sur '{key}': {vals}")

    episodes = sorted((e for s in shards for e in s["episodes"]),
                      key=lambda e: e["idx"])
    action_stats = NormStats.from_partials([s["action_partial"] for s in shards])
    proprio_stats = NormStats.from_partials([s["proprio_partial"] for s in shards])

    meta = {k: shards[0][k] for k in
            ("embed_dim", "num_patches", "encoder_family", "encoder_size",
             "dataset_id", "image_size", "fps", "seed")}
    meta.update({
        "num_episodes": len(episodes),
        "total_frames": sum(e["length"] for e in episodes),
        "episodes": episodes,
        "action_stats": action_stats.to_dict(),
        "proprio_stats": proprio_stats.to_dict(),
        "storage_dtype": shards[0].get("storage_dtype", "float16"),
        "format_version": 2,
    })

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[merge] meta.json écrit : {len(episodes)} episodes, "
          f"{meta['total_frames']} frames.")
    return meta
