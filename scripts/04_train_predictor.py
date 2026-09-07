"""
Entraîne le predictor (world model) sur les données pré-encodées.

Phase A du training :
    - Fusion cross-cam + predictor entraînés from scratch
    - Loss : MSE one-step z_t → z_t+1
    - Output : checkpoint predictor_simu.pt

GPU (RTX 5090, 32 GB) :
    - latents hébergés en VRAM (hardware.data_device: auto) → zéro transfert ;
    - autocast bf16 + TF32, AdamW fused, --compile (torch.compile) opt-in ;
    - --batch-size 64 par défaut ; mesurer 32/64/128/256 avec le benchmark
      avant d’augmenter (la mémoire dépend de la fusion et de l’encodeur) ;
    - pic VRAM + durée loggés par epoch (dans le checkpoint, clé history).

Usage:
    python scripts/04_train_predictor.py --config configs/default.yaml
        [--encoded-data results/encoded/encoded_data.pt]
        [--fusion concat_view] [--n-epochs 30] [--batch-size 64] [--lr 1e-4]
        [--precision auto] [--data-device auto] [--compile]
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.predictor import WorldModelPredictor, PredictorConfig
from src.fusion import make_fusion
from src.config import load_config, set_seed, log_environment, setup_hardware
from src.device import (resolve_amp_dtype, place_tensors, make_adamw,
                        maybe_compile, unwrap, autocast_ctx,
                        peak_vram_gb, reset_peak_vram)


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data",
                        default="results/encoded/sim_encoded_data.pt",
                        help="Données pré-encodées (depuis 02_encode_dataset)")
    parser.add_argument("--output",
                        default="results/checkpoints/predictor_simu.pt")
    parser.add_argument("--fusion", default="concat_view",
                        help="Stratégie de fusion à utiliser (choisie après M1)")
    parser.add_argument("--rollout-horizon", type=int, default=4,
                        help="Longueur du rollout multi-step pendant le training "
                             "(non utilisé pour l'instant : one-step)")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--precision", default=None,
                        help="auto (bf16 sur GPU) / bf16 / fp16 / fp32. "
                             "Défaut : hardware.precision")
    parser.add_argument("--data-device", default=None,
                        help="auto / cuda / cpu — où héberger les latents. "
                             "Défaut : hardware.data_device")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile du predictor (défaut : hardware.compile)")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    hw = setup_hardware(cfg)
    log_environment()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = resolve_amp_dtype(device, args.precision or hw["precision"])
    use_compile = (args.compile or bool(hw.get("compile", False))) and device == "cuda"
    out_path = resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # === Charger les données pré-encodées ===
    encoded_path = resolve(args.encoded_data)
    if not encoded_path.exists():
        print(f"[ERREUR] Pas de données pré-encodées trouvées : {encoded_path}")
        print("Lance d'abord : python scripts/02_encode_dataset.py")
        sys.exit(1)

    print(f"\nChargement des données : {encoded_path}")
    data = torch.load(encoded_path, weights_only=False, map_location="cpu")
    embed_dim  = data["embed_dim"]
    action_dim = data["action"].shape[-1]
    n = len(data["action"])

    print(f"  N paires   : {n}")
    print(f"  Embed dim  : {embed_dim}")
    print(f"  Action dim : {action_dim}")

    tensors, data_device = place_tensors(
        {k: data[k] for k in ("z_wrist_t", "z_global_t", "z_wrist_t1",
                              "z_global_t1", "action")},
        args.data_device or hw["data_device"], device)

    def fetch(idx):
        return tuple(tensors[k][idx].to(device, non_blocking=True)
                     for k in ("z_wrist_t", "z_global_t", "z_wrist_t1",
                               "z_global_t1", "action"))

    # === Build fusion + predictor ===
    fusion = make_fusion(args.fusion, dim=embed_dim)
    pred_cfg = PredictorConfig(
        embed_dim=embed_dim,
        action_dim=action_dim,
        n_layers=args.n_layers,
        n_heads=12 if embed_dim % 12 == 0 else 8,
        ffn_dim=embed_dim * 4,
    )
    predictor = WorldModelPredictor(pred_cfg)
    print(f"\n[Fusion] strategy : {args.fusion}, params : "
          f"{sum(p.numel() for p in fusion.parameters()):,}")
    print(f"[Predictor] params : {predictor.count_parameters():,}")
    print(f"[Precision] autocast: {amp_dtype or 'off (fp32)'}")

    fusion.to(device).train()
    predictor.to(device).train()
    raw_predictor = predictor                       # pour le checkpoint
    predictor = maybe_compile(predictor, use_compile)

    # === Split train/val ===
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    val_size = max(1, n // 5)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    print(f"\nSplit : {len(train_idx)} train / {len(val_idx)} val")

    # === Optimizer ===
    params = list(fusion.parameters()) + list(raw_predictor.parameters())
    optimizer = make_adamw(params, args.lr, 1e-4, device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    # === Training loop ===
    print(f"\n=== Training {args.n_epochs} epochs ===\n")
    history = []
    best_val = float("inf")

    for epoch in range(args.n_epochs):
        # Train
        fusion.train(); predictor.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        train_losses = []
        reset_peak_vram()
        t_epoch = time.time()

        for i in tqdm(range(0, len(perm), args.batch_size),
                      desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            zw_t, zg_t, zw_t1, zg_t1, act = fetch(perm[i:i + args.batch_size])

            with autocast_ctx(device, amp_dtype):
                # Fusion cross-cam
                z_t  = fusion(zw_t, zg_t)
                z_t1 = fusion(zw_t1, zg_t1)
                # Prediction one-step (pour le moment ; multi-step demande des séquences)
                pred = predictor(z_t, act)
            loss = nn.functional.mse_loss(pred.float(), z_t1.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        # Val
        fusion.eval(); predictor.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(val_idx), args.batch_size):
                zw_t, zg_t, zw_t1, zg_t1, act = fetch(val_idx[i:i + args.batch_size])
                with autocast_ctx(device, amp_dtype):
                    z_t  = fusion(zw_t, zg_t)
                    z_t1 = fusion(zw_t1, zg_t1)
                    pred = predictor(z_t, act)
                val_losses.append(
                    nn.functional.mse_loss(pred.float(), z_t1.float()).item())

        val_loss = sum(val_losses) / len(val_losses)
        scheduler.step()
        epoch_s = time.time() - t_epoch
        vram = peak_vram_gb() if device == "cuda" else 0.0

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "epoch_seconds": epoch_s, "peak_vram_gb": vram})
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"[{epoch_s:.0f}s, VRAM pic {vram:.1f} GB]")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "predictor_state_dict": unwrap(raw_predictor).state_dict(),
                "fusion_state_dict":    fusion.state_dict(),
                "predictor_config":     pred_cfg.__dict__,
                "fusion_name":          args.fusion,
                "embed_dim":            embed_dim,
                "action_dim":           action_dim,
                "epoch":                epoch,
                "val_loss":             val_loss,
                "history":              history,
            }, out_path)
            print(f"  [Best val, saved to {out_path.name}]")

    print(f"\n=== Training terminé ===")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Checkpoint    : {out_path}")


if __name__ == "__main__":
    main()
