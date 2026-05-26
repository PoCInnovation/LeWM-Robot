"""Train a behavior-cloning head on cached V-JEPA 2 features.

Usage:
    python train.py \\
        --cache_dir /content/cached_features/libero_object_smoke \\
        --output_dir ./checkpoints/bc_smoke \\
        --val_episodes 1 --epochs 30

Splits by episode: the last `--val_episodes` episodes are held out for validation
so the model can't see val frames during training. Logs train/val MSE per epoch
and saves the best checkpoint by val MSE. Also prints a mean-action baseline so
you can tell whether the visual features are actually being used.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from cached_loader import CachedVJepa2Dataset
from policy_head import MeanPoolMLPHead


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_episodes", type=int, default=1,
                   help="Number of trailing episodes held out for validation.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--no_state", action="store_true",
                   help="Drop the proprioceptive state input (vision-only head).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def split_indices_by_episode(ds: CachedVJepa2Dataset, n_val: int):
    """Return (train_clip_indices, val_clip_indices) splitting by episode."""
    ep_per_clip = [ep for _, ep, _ in ds._index]
    unique_eps = sorted(set(ep_per_clip))
    if n_val >= len(unique_eps):
        raise ValueError(
            f"--val_episodes={n_val} but cache only has {len(unique_eps)} episodes."
        )
    val_eps = set(unique_eps[-n_val:])
    train_idx = [i for i, ep in enumerate(ep_per_clip) if ep not in val_eps]
    val_idx = [i for i, ep in enumerate(ep_per_clip) if ep in val_eps]
    return train_idx, val_idx, sorted(val_eps)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    with open(cache_dir / "metadata.json") as f:
        meta = json.load(f)
    encoder_dim = meta["encoder_dim"]

    full_ds = CachedVJepa2Dataset(cache_dir, clip_length=1, stride=1)
    sample = full_ds[0]
    state_dim_full = sample["states"].shape[-1]
    state_dim = None if args.no_state else state_dim_full
    action_dim = sample["actions"].shape[-1]

    train_idx, val_idx, val_eps = split_indices_by_episode(full_ds, args.val_episodes)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    print(f"encoder_dim={encoder_dim}  state_dim={state_dim}  action_dim={action_dim}")
    print(f"total clips={len(full_ds)}  episodes={meta['num_episodes']}")
    print(f"train clips={len(train_ds)}  val clips={len(val_ds)}  (val episodes={val_eps})")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    head = MeanPoolMLPHead(
        encoder_dim=encoder_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"params: {n_params / 1e3:.1f}k")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # Mean-action baseline: predict the train-set mean action for every val frame.
    train_actions = torch.stack([full_ds[i]["actions"].squeeze(0) for i in train_idx])
    val_actions = torch.stack([full_ds[i]["actions"].squeeze(0) for i in val_idx])
    mean_action = train_actions.mean(dim=0)
    baseline_mse = ((val_actions - mean_action) ** 2).mean().item()
    print(f"baseline (predict mean train action) val MSE = {baseline_mse:.4f}\n")

    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        head.train()
        train_losses = []
        for batch in train_dl:
            feats = batch["features"].squeeze(1).to(device)        # (B, P, P, D)
            actions = batch["actions"].squeeze(1).to(device)       # (B, A)
            state = (batch["states"].squeeze(1).to(device)
                     if state_dim is not None else None)
            pred = head(feats, state)
            loss = loss_fn(pred, actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_mse = sum(train_losses) / len(train_losses)

        head.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                feats = batch["features"].squeeze(1).to(device)
                actions = batch["actions"].squeeze(1).to(device)
                state = (batch["states"].squeeze(1).to(device)
                         if state_dim is not None else None)
                val_losses.append(loss_fn(head(feats, state), actions).item())
        val_mse = sum(val_losses) / len(val_losses)

        improved = val_mse < best_val
        if improved:
            best_val = val_mse
            torch.save({
                "model_state": head.state_dict(),
                "args": vars(args),
                "encoder_dim": encoder_dim,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden_dim": args.hidden_dim,
                "epoch": epoch,
                "val_mse": val_mse,
                "val_episodes": val_eps,
            }, out / "best.pt")

        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        marker = " *" if improved else ""
        print(f"epoch {epoch:3d}  train {train_mse:.4f}  val {val_mse:.4f}{marker}")

    with open(out / "history.json", "w") as f:
        json.dump({
            "history": history,
            "baseline_val_mse": baseline_mse,
            "best_val_mse": best_val,
        }, f, indent=2)

    print(f"\nbest val MSE: {best_val:.4f}  (baseline: {baseline_mse:.4f})")
    if best_val < baseline_mse * 0.8:
        print("→ model is meaningfully better than baseline. Features are being used.")
    else:
        print("→ model is close to baseline. Add more episodes or train longer.")
    print(f"checkpoint: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
