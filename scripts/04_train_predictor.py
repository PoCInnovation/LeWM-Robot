"""
Entraîne le predictor (world model) sur des latents pré-encodés (format v2).

ADAPTATION MACHINE UNIQUEMENT (Adastra MI250X) — la logique d'entraînement
d'origine est préservée à l'identique :
    - one-step MSE (z_t, action) → z_t+1
    - fusion entraînée conjointement avec le predictor
    - actions brutes (pas de normalisation)
    - split train/val aléatoire par paires (seedé)

Ce qui change (machine) :
    - lit le format v2 par épisode (produit par l'encodage shardé de 02)
    - autocast bf16 (excellent sur MI250X, off sur CPU)
    - checkpoint last.pt + --resume : survit au walltime 24 h / --requeue

Usage:
    python scripts/04_train_predictor.py --encoded-data results/encoded/<name>
        [--fusion concat_view] [--n-epochs 30] [--batch-size 32] [--lr 1e-4]
        [--output-dir results/checkpoints/predictor_sim] [--resume]
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.predictor import WorldModelPredictor, PredictorConfig
from src.fusion import make_fusion
from src.encoded_data import EncodedEpisodes, PairsView
from src.config import load_config, set_seed, log_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data", required=True,
                        help="Dossier des latents pré-encodés (format v2)")
    parser.add_argument("--output-dir",
                        default="results/checkpoints/predictor_sim")
    parser.add_argument("--fusion", default="concat_view",
                        help="Stratégie de fusion à utiliser (choisie après M1)")
    parser.add_argument("--delta", type=int, default=None,
                        help="delta_timesteps de la paire (défaut : config)")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-bf16", action="store_true",
                        help="Désactiver l'autocast bf16 (défaut: activé sur GPU)")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre depuis <output-dir>/last.pt")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    log_environment()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and not args.no_bf16
    out_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() \
        else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"

    delta = args.delta if args.delta is not None \
        else int(cfg["dataset"].get("delta_timesteps", 1))

    # === Données (format v2, actions BRUTES comme dans la version d'origine) ===
    data_path = Path(args.encoded_data)
    data = EncodedEpisodes(data_path if data_path.is_absolute()
                           else ROOT / data_path)
    print(f"\n[Data] {data.summary()}")
    embed_dim = data.embed_dim
    action_dim = data.action_stats.mean.shape[0]

    pairs = PairsView(data, list(range(len(data))), delta=delta,
                      norm_actions=False, norm_proprio=False)

    # Split train/val par paires, seedé (logique d'origine)
    n = len(pairs)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    val_size = max(1, n // 5)
    val_idx, train_idx = perm[:val_size].tolist(), perm[val_size:].tolist()
    print(f"\nSplit : {len(train_idx)} train / {len(val_idx)} val "
          f"(paires, delta={delta})")

    train_loader = DataLoader(Subset(pairs, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(Subset(pairs, val_idx),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # === Build fusion + predictor (logique d'origine : fusion entraînée) ===
    fusion = make_fusion(args.fusion, dim=embed_dim)
    pred_cfg = PredictorConfig(
        embed_dim=embed_dim,
        action_dim=action_dim,
        n_layers=6,
        n_heads=12 if embed_dim == 768 else 8,
        ffn_dim=embed_dim * 4,
    )
    predictor = WorldModelPredictor(pred_cfg)
    print(f"\n[Fusion] strategy : {args.fusion}, params : "
          f"{sum(p.numel() for p in fusion.parameters()):,}")
    print(f"[Predictor] params : {predictor.count_parameters():,}")
    print(f"[Precision] bf16 autocast : {'ON' if use_bf16 else 'off'}")

    fusion.to(device)
    predictor.to(device)

    params = list(fusion.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    # === Resume (machine : walltime 24 h) ===
    start_epoch, best_val, history = 0, float("inf"), []
    if args.resume and last_path.exists():
        ck = torch.load(last_path, weights_only=False, map_location=device)
        predictor.load_state_dict(ck["predictor_state_dict"])
        fusion.load_state_dict(ck["fusion_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        history = ck["history"]
        torch.set_rng_state(ck["rng_state"].cpu())
        if device == "cuda" and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng_state"].cpu())
        print(f"[Resume] Reprise à l'epoch {start_epoch} "
              f"(best val: {best_val:.5f})")
    elif args.resume:
        print(f"[Resume] Pas de {last_path}, démarrage from scratch.")

    def checkpoint_payload(epoch, val_loss):
        return {
            "predictor_state_dict": predictor.state_dict(),
            "fusion_state_dict":    fusion.state_dict(),
            "predictor_config":     pred_cfg.__dict__,
            "fusion_name":          args.fusion,
            "embed_dim":            embed_dim,
            "action_dim":           action_dim,
            "delta":                delta,
            "encoded_data_meta": {k: data.meta[k] for k in
                                  ("encoder_family", "encoder_size",
                                   "image_size", "dataset_id")},
            "epoch":                epoch,
            "val_loss":             val_loss,
            "best_val":             best_val,
            "history":              history,
        }

    def run_batch(batch):
        zw_t = batch["z_wrist_t"].to(device)
        zg_t = batch["z_global_t"].to(device)
        zw_t1 = batch["z_wrist_t1"].to(device)
        zg_t1 = batch["z_global_t1"].to(device)
        act = batch["action"].to(device)

        # Fusion cross-cam (entraînée conjointement — logique d'origine)
        z_t = fusion(zw_t, zg_t)
        z_t1 = fusion(zw_t1, zg_t1)
        pred = predictor(z_t, act)
        return nn.functional.mse_loss(pred, z_t1)

    # === Training loop ===
    print(f"\n=== Training epochs {start_epoch}..{args.n_epochs - 1} ===\n")
    for epoch in range(start_epoch, args.n_epochs):
        fusion.train(); predictor.train()
        train_losses = []
        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss = run_batch(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / max(1, len(train_losses))

        # Val
        fusion.eval(); predictor.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    val_losses.append(run_batch(batch).item())
        val_loss = sum(val_losses) / max(1, len(val_losses))
        scheduler.step()

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "lr": scheduler.get_last_lr()[0]})
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint_payload(epoch, val_loss), best_path)
            print(f"  [best → {best_path.name}]")
        torch.save({**checkpoint_payload(epoch, val_loss),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (torch.cuda.get_rng_state()
                                       if device == "cuda" else None)},
                   last_path)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n=== Training terminé ===")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Checkpoints   : {best_path} (+ last.pt, history.json)")


if __name__ == "__main__":
    main()
