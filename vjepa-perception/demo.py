"""Generate a behavior-cloning demo plot from a trained head.

For a held-out episode, plots predicted vs ground-truth actions per dimension.

Usage:
    python demo.py \\
        --cache_dir /content/cached_features/libero_object_smoke \\
        --checkpoint ./checkpoints/bc_smoke/best.pt \\
        --output demo.png
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from policy_head import MeanPoolMLPHead


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode", type=int, default=None,
                   help="Episode index. Defaults to the highest episode in the cache (likely held out at train time).")
    p.add_argument("--output", default="demo.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_episode(cache_dir: Path, ep: int):
    path = cache_dir / f"episode_{ep:06d}.safetensors"
    with safe_open(path, framework="pt") as f:
        return {
            "features": f.get_tensor("features"),
            "states": f.get_tensor("states"),
            "actions": f.get_tensor("actions"),
            "frame_indices": f.get_tensor("frame_indices"),
        }


def main():
    args = parse_args()
    # matplotlib is imported lazily so importing this module elsewhere stays light.
    import matplotlib.pyplot as plt

    cache_dir = Path(args.cache_dir)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    with open(cache_dir / "metadata.json") as f:
        meta = json.load(f)
    if args.episode is None:
        args.episode = max(int(k) for k in meta["episode_lengths"].keys())
    print(f"Episode {args.episode}  ({meta['episode_lengths'][str(args.episode)]} frames)")

    ep_data = load_episode(cache_dir, args.episode)

    state_dim = ckpt["state_dim"]
    head = MeanPoolMLPHead(
        encoder_dim=ckpt["encoder_dim"],
        state_dim=state_dim,
        action_dim=ckpt["action_dim"],
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(args.device)
    head.load_state_dict(ckpt["model_state"])
    head.eval()

    feats = ep_data["features"].to(args.device)
    states = ep_data["states"].to(args.device) if state_dim is not None else None
    actions_gt = ep_data["actions"].cpu().numpy()

    with torch.no_grad():
        preds = head(feats, states).cpu().numpy()

    action_dim = preds.shape[-1]
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 1.6 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(actions_gt[:, i], label="ground truth", lw=2)
        ax.plot(preds[:, i], label="predicted", lw=1.5, linestyle="--")
        ax.set_ylabel(f"a[{i}]")
        ax.grid(alpha=0.3)
    axes[0].set_title(
        f"Episode {args.episode} — V-JEPA 2 frozen features → BC head (val MSE: {ckpt['val_mse']:.4f})"
    )
    axes[-1].set_xlabel("frame (post-stride)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"saved plot: {args.output}")

    mse_per_dim = ((preds - actions_gt) ** 2).mean(axis=0)
    print("MSE per action dim:")
    for i, m in enumerate(mse_per_dim):
        print(f"  a[{i}]: {m:.4f}")
    print(f"overall MSE: {mse_per_dim.mean():.4f}")


if __name__ == "__main__":
    main()
