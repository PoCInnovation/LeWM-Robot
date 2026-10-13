"""
Latents pré-encodés (sortie de 02_encode_dataset.py) : chargement économe en
RAM et accès par batch, quel que soit le format du .pt.

Deux formats :
  - v2 (02 actuel) — chaque image n'est stockée qu'une fois :
        z_wrist, z_global : (F, P, D) tables de frames (bf16 sur GPU)
        idx_t, idx_t1     : (N,) ligne de la frame t / t+1 de chaque paire
    Dans la passe propre, la frame t+1 d'une paire est la frame t de la
    suivante : elle n'est encodée et stockée qu'une fois (÷2 sur le disque,
    la RAM et la VRAM). Les passes augmentées stockent leurs 2 frames par
    paire (les paramètres d'augmentation sont tirés par paire).
  - v1 (ancien) — 4 tenseurs par paire : z_wrist_t, z_global_t, z_wrist_t1,
    z_global_t1 : (N, P, D).

Sur RTX 4090 (24 GB VRAM, 32 GB RAM), le dataset complet en v2/bf16 fait
~16 GB : il tient en VRAM à côté du training, et le chargement en mmap
évite de le faire transiter en entier par la RAM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import torch

from src.device import DeviceLike, place_tensors

FORMAT_VERSION = 2
LEGACY_KEYS: Tuple[str, ...] = ("z_wrist_t", "z_global_t", "z_wrist_t1", "z_global_t1")


def load_encoded(path: Union[str, Path]) -> dict:
    """
    torch.load en mmap : les tenseurs restent adossés au fichier et ne
    passent en RAM que page par page, au moment où on les lit (copie vers
    le GPU, ou indexation d'un batch). Pages de cache, donc récupérables par
    le système : pas de pic à la taille du fichier.
    """
    return torch.load(path, weights_only=False, map_location="cpu", mmap=True)


def build_frame_index(pair_frames: Sequence[int], delta: int
                      ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Frames uniques d'une liste de paires (t, t+delta).

    Args:
        pair_frames : index global de la frame t de chaque paire
        delta       : écart t → t+1 (dataset.delta_timesteps)

    Returns:
        frames : index globaux des frames uniques, triés
        idx_t  : (N,) ligne de `frames` contenant la frame t de chaque paire
        idx_t1 : (N,) ligne de `frames` contenant la frame t+delta
    """
    frames = sorted(set(pair_frames) | {f + delta for f in pair_frames})
    row = {f: r for r, f in enumerate(frames)}
    idx_t = torch.tensor([row[f] for f in pair_frames], dtype=torch.long)
    idx_t1 = torch.tensor([row[f + delta] for f in pair_frames], dtype=torch.long)
    return frames, idx_t, idx_t1


class LatentPairs:
    """
    Paires de latents (t, t+1) + actions, hébergées en VRAM si elles tiennent.

    `fetch(idx)` renvoie le batch en fp32 sur le device d'entraînement : le
    stockage peut être en bf16 (lossless : l'encodeur calcule en bf16), les
    modules aval reçoivent toujours du fp32 comme avant.
    """

    def __init__(self, data: dict, requested: str = "auto",
                 train_device: DeviceLike = "cpu", reserve_gb: float = 5.0,
                 verbose: bool = True):
        self.train_device = torch.device(train_device)
        self.indexed = "z_wrist" in data
        if self.indexed:
            tensors = {"wrist": data["z_wrist"], "global": data["z_global"],
                       "idx_t": torch.as_tensor(data["idx_t"]).long(),
                       "idx_t1": torch.as_tensor(data["idx_t1"]).long()}
        else:
            tensors = {k: data[k] for k in LEGACY_KEYS}
        tensors["action"] = torch.as_tensor(data["action"]).float()
        if "proprio" in data:          # état courant : actions en mode "delta"
            tensors["proprio"] = torch.as_tensor(data["proprio"]).float()

        self.tensors, self.device = place_tensors(
            tensors, requested, self.train_device, verbose=verbose,
            reserve_gb=reserve_gb)

        ref = self.tensors["wrist" if self.indexed else "z_wrist_t"]
        self.num_patches = ref.shape[1]
        self.embed_dim = ref.shape[2]
        self.action_dim = self.tensors["action"].shape[-1]

    def __len__(self) -> int:
        return self.tensors["action"].shape[0]

    def _out(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.train_device, non_blocking=True).float()

    def fetch_t(self, idx: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z_wrist_t, z_global_t, action) — pour les probes de 03."""
        idx = idx.to(self.device)
        t = self.tensors
        if self.indexed:
            rows = t["idx_t"][idx]
            zw, zg = t["wrist"][rows], t["global"][rows]
        else:
            zw, zg = t["z_wrist_t"][idx], t["z_global_t"][idx]
        return self._out(zw), self._out(zg), self._out(t["action"][idx])

    def fetch(self, idx: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """(z_wrist_t, z_global_t, z_wrist_t1, z_global_t1, action)."""
        idx = idx.to(self.device)
        t = self.tensors
        if self.indexed:
            rows_t, rows_t1 = t["idx_t"][idx], t["idx_t1"][idx]
            out = (t["wrist"][rows_t], t["global"][rows_t],
                   t["wrist"][rows_t1], t["global"][rows_t1])
        else:
            out = tuple(t[k][idx] for k in LEGACY_KEYS)
        return tuple(self._out(x) for x in out) + (self._out(t["action"][idx]),)

    def normalize_actions(self, stats: "ActionStats") -> None:
        """Passe les actions hébergées en espace normalisé (une fois, sur place).

        Mode "delta" : l'action devient le déplacement commandé (cible − état
        courant) avant normalisation, comme dans `action_targets`.
        """
        if stats.mode == "delta":
            if "proprio" not in self.tensors:
                raise ValueError("Actions en mode delta : latents sans 'proprio'.")
            self.tensors["action"] = self.tensors["action"] - self.tensors["proprio"]
        self.tensors["action"] = stats.normalize(self.tensors["action"])

    def materialize(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Dict au format v1 (4 tenseurs par paire) pour un sous-ensemble."""
        zw_t, zg_t, zw_t1, zg_t1, act = self.fetch(idx)
        return {"z_wrist_t": zw_t, "z_global_t": zg_t,
                "z_wrist_t1": zw_t1, "z_global_t1": zg_t1, "action": act}


def latent_nbytes(n_rows: int, num_patches: int, embed_dim: int,
                  dtype: torch.dtype, n_cameras: int = 2) -> int:
    """Taille des tables de frames (toutes caméras) en octets."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    return n_rows * n_cameras * num_patches * embed_dim * itemsize


@dataclass
class ActionStats:
    """
    Normalisation z-score des actions, calculée sur le train.

    Les actions SO-101 sont en degrés (≈ [-105, +100], écart-type jusqu'à ~27
    selon l'articulation) : le predictor est entraîné en espace normalisé et
    le CEM y échantillonne (moyenne 0, écart-type 1 = distribution du dataset,
    bornes = min/max du dataset). Les stats voyagent dans le checkpoint ; les
    actions planifiées sont dénormalisées avant affichage / envoi au robot.

    mode :
        "absolute" — l'action est la position cible des articulations ;
        "delta"    — l'action est le déplacement commandé (cible − état
                     courant). Sur les démos réelles, la cible est presque la
                     position courante (écart-type 2–6° contre 25–55°) : en
                     absolu, le signal utile pèse ~10 % et le predictor finit
                     par ignorer l'action ; en delta, il pèse 100 %.
                     Cible à envoyer au robot = état courant + delta.
    """
    mean: List[float]
    std: List[float]
    low: List[float]
    high: List[float]
    mode: str = "absolute"

    @classmethod
    def from_actions(cls, actions: torch.Tensor, min_std: float = 1e-6,
                     mode: str = "absolute") -> "ActionStats":
        a = actions.double()
        std = a.std(0, unbiased=False)
        std = torch.where(std < min_std, torch.ones_like(std), std)  # articulation figée
        return cls(mean=a.mean(0).tolist(), std=std.tolist(),
                   low=a.min(0).values.tolist(), high=a.max(0).values.tolist(),
                   mode=mode)

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "ActionStats":
        return cls(**{k: list(d[k]) for k in ("mean", "std", "low", "high")},
                   mode=d.get("mode", "absolute"))

    def to_dict(self) -> Dict[str, List[float]]:
        return asdict(self)

    def _t(self, values, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(values, dtype=like.dtype, device=like.device)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self._t(self.mean, actions)) / self._t(self.std, actions)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self._t(self.std, actions) + self._t(self.mean, actions)

    def normalized_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bornes du dataset (min/max par articulation) en espace normalisé."""
        return (self.normalize(torch.tensor(self.low)),
                self.normalize(torch.tensor(self.high)))


def frameskip_pairs(data: dict, k: int) -> dict:
    """
    Paires (t, t+k) construites depuis les frames déjà encodées (format v2).

    À 30 fps, le bras bouge de ~0,2° entre deux frames : dans les latents,
    ce mouvement se noie dans le bruit (compression vidéo) et le predictor
    apprend à ignorer l'action. Sur k frames (k=5 ≈ 0,17 s, comme DINO-WM),
    le mouvement devient visible.

    Pour chaque frame propre t dont les paires t, t+1, ..., t+k-1 existent
    dans le même épisode :
        z_t      = frame t
        z_{t+k}  = frame t+k (cible de la paire t+k-1)
        action   = action de la paire t+k-1 (dernière cible commandée)
        proprio  = état en t
    → en mode "delta", l'action vaut le déplacement commandé sur la fenêtre.
    Les lignes augmentées (images propres à une paire t → t+1) sont écartées.

    Returns:
        nouveau dict au format v2 (mêmes tables de frames, partagées)
    """
    if k <= 1:
        return data
    if "z_wrist" not in data:
        raise ValueError("frameskip : format v2 (tables de frames) requis")
    episode = torch.as_tensor(data["episode_idx"]).flatten()
    frame = torch.as_tensor(data["frame_idx"]).flatten()
    rows = torch.arange(len(episode))
    is_aug = data.get("is_augmented")
    if is_aug is not None:
        rows = rows[~torch.as_tensor(is_aug).flatten().bool()]
    row_of = {(int(episode[r]), int(frame[r])): int(r) for r in rows}
    first, last = [], []
    for (ep, f), r in row_of.items():
        window = [row_of.get((ep, f + j)) for j in range(k)]
        if all(w is not None for w in window):
            first.append(r)
            last.append(window[-1])
    order = torch.argsort(torch.tensor(first))        # ordre d'origine (épisodes, frames)
    first = torch.tensor(first)[order]
    last = torch.tensor(last)[order]

    out = {key: v for key, v in data.items()
           if not (torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == len(episode))}
    out.update(z_wrist=data["z_wrist"], z_global=data["z_global"])
    out["idx_t"] = torch.as_tensor(data["idx_t"])[first]
    out["idx_t1"] = torch.as_tensor(data["idx_t1"])[last]
    out["action"] = torch.as_tensor(data["action"])[last]
    for key in ("proprio", "episode_idx", "frame_idx", "pair_id"):
        if key in data:
            out[key] = torch.as_tensor(data[key])[first]
    out["is_augmented"] = torch.zeros(len(first), dtype=torch.long)
    out["recipe_idx"] = torch.zeros(len(first), dtype=torch.long)
    out["frameskip"] = k
    return out


ACTION_MODES = ("absolute", "delta")


def action_targets(data: dict, mode: str = "absolute") -> torch.Tensor:
    """Actions telles que le predictor les voit (avant normalisation)."""
    actions = torch.as_tensor(data["action"]).float()
    if mode == "delta":
        actions = actions - torch.as_tensor(data["proprio"]).float()
    elif mode != "absolute":
        raise ValueError(f"mode d'action inconnu : {mode}")
    return actions


def train_action_stats(data: dict, train_idx: torch.Tensor,
                       mode: str = "absolute") -> ActionStats:
    """Stats des actions du train, hors échantillons augmentés (bruit capteur)."""
    actions = action_targets(data, mode)[train_idx]
    is_aug = data.get("is_augmented")
    if is_aug is not None:
        clean = ~torch.as_tensor(is_aug).flatten().bool()[train_idx]
        if bool(clean.any()):
            actions = actions[clean]
    return ActionStats.from_actions(actions.float(), mode=mode)
