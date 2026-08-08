"""
Entraîne le LoRA adapter sur les démos réelles pré-encodées (format v2).

ADAPTATION MACHINE UNIQUEMENT — logique d'origine préservée :
    - predictor figé + LoRA dans toutes les couches linéaires
    - fusion gelée (chargée du checkpoint de 04)
    - one-step MSE, actions brutes, split par paires (seedé)

Ce qui change (machine) : format v2, bf16, checkpoint last.pt + --resume.

Usage:
    python scripts/05_train_lora.py --encoded-data results/encoded/<real>
        [--predictor-ckpt results/checkpoints/predictor_sim/best.pt]
        [--output-dir results/checkpoints/predictor_real] [--resume]
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
from src.lora import LoRAConfig, inject_lora, get_lora_parameters
from src.encoded_data import EncodedEpisodes, PairsView
from src.config import load_config, set_seed, log_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictor-ckpt",
                        default="results/checkpoints/predictor_sim/best.pt",
                        help="Checkpoint du predictor entraîné (de 04)")
    parser.add_argument("--encoded-data", required=True,
                        help="Démos réelles pré-encodées (dossier format v2)")
    parser.add_argument("--output-dir",
                        default="results/checkpoints/predictor_real")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--delta", type=int, default=None,
                        help="Défaut : le delta du checkpoint 04")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
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

    # === Charger le predictor entraîné sur sim ===
    ckpt_path = Path(args.predictor_ckpt)
    ckpt_path = ckpt_path if ckpt_path.is_absolute() else ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"[ERREUR] Pas de predictor checkpoint trouvé : {ckpt_path}")
        print("Lance d'abord : python scripts/04_train_predictor.py")
        sys.exit(1)

    print(f"\nChargement du predictor : {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    pred_cfg = PredictorConfig(**ckpt["predictor_config"])
    predictor = WorldModelPredictor(pred_cfg)
    predictor.load_state_dict(ckpt["predictor_state_dict"])

    fusion = make_fusion(ckpt["fusion_name"], dim=ckpt["embed_dim"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    delta = args.delta if args.delta is not None else int(ckpt.get("delta", 1))

    print(f"  Predictor params : {predictor.count_parameters():,}")
    print(f"  Fusion strategy  : {ckpt['fusion_name']}")
    print(f"  Embed dim        : {ckpt['embed_dim']}  |  delta : {delta}")

    # === Injection LoRA (toutes les Linear — logique d'origine) ===
    print(f"\n=== Injection LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}) ===")
    lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)
    predictor = inject_lora(predictor, lora_cfg, verbose=False)

    # === Démos réelles (format v2, actions brutes) ===
    data_path = Path(args.encoded_data)
    data = EncodedEpisodes(data_path if data_path.is_absolute()
                           else ROOT / data_path)
    print(f"\n[Data] {data.summary()}")
    if data.embed_dim != ckpt["embed_dim"]:
        print(f"[ERREUR] embed_dim données ({data.embed_dim}) != "
              f"checkpoint ({ckpt['embed_dim']}) — encodeurs différents ?")
        sys.exit(1)

    pairs = PairsView(data, list(range(len(data))), delta=delta,
                      norm_actions=False, norm_proprio=False)
    n = len(pairs)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    val_size = max(1, n // 5)
    val_idx, train_idx = perm[:val_size].tolist(), perm[val_size:].tolist()
    print(f"Split : {len(train_idx)} train / {len(val_idx)} val (paires)")

    train_loader = DataLoader(Subset(pairs, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(Subset(pairs, val_idx),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    fusion.to(device).eval()       # fusion gelée (entraînée en Phase A)
    for p in fusion.parameters():
        p.requires_grad = False
    predictor.to(device)

    lora_params = get_lora_parameters(predictor)
    print(f"\n  Params LoRA à entraîner : "
          f"{sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    # === Resume ===
    start_epoch, best_val, history = 0, float("inf"), []
    if args.resume and last_path.exists():
        ck = torch.load(last_path, weights_only=False, map_location=device)
        predictor.load_state_dict(ck["predictor_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        history = ck["history"]
        torch.set_rng_state(ck["rng_state"].cpu())
        print(f"[Resume] Reprise à l'epoch {start_epoch}")

    def run_batch(batch):
        act = batch["action"].to(device)
        with torch.no_grad():
            z_t = fusion(batch["z_wrist_t"].to(device),
                         batch["z_global_t"].to(device))
            z_t1 = fusion(batch["z_wrist_t1"].to(device),
                          batch["z_global_t1"].to(device))
        pred = predictor(z_t, act)
        return nn.functional.mse_loss(pred, z_t1)

    def checkpoint_payload(epoch, val_loss):
        return {
            "predictor_state_dict": predictor.state_dict(),
            "fusion_state_dict":    fusion.state_dict(),
            "predictor_config":     pred_cfg.__dict__,
            "fusion_name":          ckpt["fusion_name"],
            "lora_config":          lora_cfg.__dict__,
            "embed_dim":            ckpt["embed_dim"],
            "action_dim":           ckpt["action_dim"],
            "delta":                delta,
            "encoded_data_meta":    ckpt.get("encoded_data_meta"),
            "epoch":                epoch,
            "val_loss":             val_loss,
            "best_val":             best_val,
            "history":              history,
        }

    # === Training loop ===
    print(f"\n=== Training LoRA epochs {start_epoch}..{args.n_epochs - 1} ===\n")
    for epoch in range(start_epoch, args.n_epochs):
        predictor.train()
        train_losses = []
        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss = run_batch(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / max(1, len(train_losses))

        predictor.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    val_losses.append(run_batch(batch).item())
        val_loss = sum(val_losses) / max(1, len(val_losses))
        scheduler.step()

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint_payload(epoch, val_loss), best_path)
            print(f"  [best → {best_path.name}]")
        torch.save({**checkpoint_payload(epoch, val_loss),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state()},
                   last_path)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n=== LoRA training terminé ===")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Checkpoint    : {best_path}")


if __name__ == "__main__":
    main()
