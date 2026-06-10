"""Loader for V-JEPA 2 cached features.

Reads per-episode .safetensors files written by `precompute_features.py` and
emits sliding-window clips for downstream training.

Usage:
    from cached_loader import CachedVJepa2Dataset
    from torch.utils.data import DataLoader

    ds = CachedVJepa2Dataset("./cached_features/libero_object",
                              clip_length=16, stride=2)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    for batch in dl:
        features = batch["features"]   # (B, 16, P, P, D)  cache dtype
        states = batch["states"]       # (B, 16, state_dim)
        actions = batch["actions"]     # (B, 16, action_dim)
        # downstream model here

Reads are zero-copy via `safetensors.safe_open` and only the requested slice
is materialized, so memory cost per __getitem__ is O(clip_length), not O(episode).
"""

from __future__ import annotations
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from safetensors import safe_open


class CachedVJepa2Dataset(Dataset):
    """Sliding-window loader over precomputed V-JEPA 2 features.

    Each __getitem__ returns:
        features:       (clip_length, P, P, D)        cache dtype
        states:         (clip_length, state_dim)      float32
        actions:        (clip_length, action_dim)     float32
        frame_indices:  (clip_length,)                int64
        episode_index:  scalar int

    Args:
        cache_dir:   directory containing metadata.json + episode_*.safetensors
        clip_length: number of frames per emitted clip
        stride:      temporal stride between consecutive frames within a clip
                     (stride=1 → native fps; stride=2 → half the source fps; etc.)
    """

    def __init__(self, cache_dir: str | Path, clip_length: int = 16, stride: int = 1):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.clip_length = clip_length
        self.stride = stride
        self._handles = {}  # Lazily cached safe_open file handles (worker-safe)

        # Frames spanned by one clip in the cache (at native fps).
        self._clip_span = (clip_length - 1) * stride + 1

        # Build the (path, ep_idx, start) index.
        self._index: list[tuple[Path, int, int]] = []
        for ep_str, n_frames in sorted(
            self.metadata["episode_lengths"].items(), key=lambda kv: int(kv[0])
        ):
            ep_idx = int(ep_str)
            ep_path = self.cache_dir / f"episode_{ep_idx:06d}.safetensors"
            n_starts = max(0, n_frames - self._clip_span + 1)
            for start in range(n_starts):
                self._index.append((ep_path, ep_idx, start))

        if not self._index:
            raise ValueError(
                f"No valid clips: clip_length={clip_length} with stride={stride} "
                f"requires episodes of at least {self._clip_span} frames."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        path, ep_idx, start = self._index[idx]
        end = start + self._clip_span
        sl = slice(start, end, self.stride)
        
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt")
            
        f = self._handles[path]
        features = f.get_slice("features")[sl]
        states = f.get_slice("states")[sl]
        actions = f.get_slice("actions")[sl]
        frame_indices = f.get_slice("frame_indices")[sl]
        return {
            "features": features,
            "states": states,
            "actions": actions,
            "frame_indices": frame_indices,
            "episode_index": ep_idx,
        }


def episode_iter(cache_dir: str | Path):
    """Iterate over whole episodes (not clips). Useful for offline analysis.

    Yields dicts identical in shape to a __getitem__ result, but with the full
    episode length instead of clip_length.
    """
    cache_dir = Path(cache_dir)
    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)
    for ep_str in sorted(metadata["episode_lengths"].keys(), key=int):
        ep_idx = int(ep_str)
        path = cache_dir / f"episode_{ep_idx:06d}.safetensors"
        with safe_open(path, framework="pt") as f:
            yield {
                "features": f.get_tensor("features"),
                "states": f.get_tensor("states"),
                "actions": f.get_tensor("actions"),
                "frame_indices": f.get_tensor("frame_indices"),
                "episode_index": ep_idx,
            }