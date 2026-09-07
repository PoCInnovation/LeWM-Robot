"""
Entraîne le LoRA adapter sur les démos réelles.

Phase B du training :
    - Predictor figé (chargé depuis le checkpoint de 04_train_predictor.py)
    - Injection de LoRA dans les couches linéaires du predictor
    - Seuls les paramètres LoRA sont entraînés
    - Fusion gelée (chargée du checkpoint)
    - Données : démos réelles pré-encodées (depuis 02_encode_dataset)
    - Output : checkpoint predictor_real.pt avec LoRA dedans

GPU (RTX 5090) : latents en VRAM, autocast bf16, AdamW fused, --compile
opt-in. Mesurer le temps nécessaire avec 07_benchmark.py.

Usage:
    python scripts/05_train_lora.py --config configs/default.yaml
        [--real-data results/encoded/real_encoded_data.pt]
        [--predictor-ckpt results/checkpoints/predictor_simu.pt]
        [--lora-rank 8] [--n-epochs 20] [--batch-size 32]
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
from src.lora import LoRAConfig, inject_lora, get_lora_parameters
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
    parser.add_argument("--predictor-ckpt",
                        default="results/checkpoints/predictor_simu.pt",
                        help="Checkpoint du predictor entraîné (de 04)")
    parser.add_argument("--real-data",
                        default="results/encoded/real_encoded_data.pt",
                        help="Démos réelles pré-encodées (depuis 02)")
    parser.add_argument("--output",
                        default="results/checkpoints/predictor_real.pt")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--precision", default=None,
                        help="auto (bf16 sur GPU) / bf16 / fp16 / fp32")
    parser.add_argument("--data-device", default=None,
                        help="auto / cuda / cpu — où héberger les latents")
    parser.add_argument("--compile", action="store_true")
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

    # === Charger le predictor entraîné sur sim ===
    ckpt_path = resolve(args.predictor_ckpt)
    if not ckpt_path.exists():
        print(f"[ERREUR] Pas de predictor checkpoint trouvé : {ckpt_path}")
        print("Lance d'abord : python scripts/04_train_predictor.py")
        sys.exit(1)

    print(f"\nChargement du predictor : {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    pred_cfg = PredictorConfig(**ckpt["predictor_config"])
    predictor = WorldModelPredictor(pred_cfg)
    predictor.load_state_dict(ckpt["predictor_state_dict"])

    fusion = make_fusion(ckpt["fusion_name"], dim=ckpt["embed_dim"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])

    print(f"  Predictor params : {predictor.count_parameters():,}")
    print(f"  Fusion strategy  : {ckpt['fusion_name']}")
    print(f"  Embed dim        : {ckpt['embed_dim']}")
    print(f"[Precision] autocast: {amp_dtype or 'off (fp32)'}")

    # === Injection LoRA ===
    print(f"\n=== Injection LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}) ===")
    lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)
    predictor = inject_lora(predictor, lora_cfg, verbose=False)

    # === Charger les démos réelles pré-encodées ===
    real_path = resolve(args.real_data)
    if not real_path.exists():
        print(f"\n[ERREUR] Données réelles non trouvées : {real_path}")
        print("Pré-encode tes démos avec : python scripts/02_encode_dataset.py "
              f"--output {real_path}")
        sys.exit(1)

    print(f"\nChargement des démos réelles : {real_path}")
    real = torch.load(real_path, weights_only=False, map_location="cpu")
    if real["embed_dim"] != ckpt["embed_dim"]:
        print(f"[ERREUR] embed_dim des données réelles ({real['embed_dim']}) != "
              f"checkpoint ({ckpt['embed_dim']}) — encodeurs différents ?")
        sys.exit(1)
    n = len(real["action"])
    print(f"  N paires : {n}")

    tensors, data_device = place_tensors(
        {k: real[k] for k in ("z_wrist_t", "z_global_t", "z_wrist_t1",
                              "z_global_t1", "action")},
        args.data_device or hw["data_device"], device)

    def fetch(idx):
        return tuple(tensors[k][idx].to(device, non_blocking=True)
                     for k in ("z_wrist_t", "z_global_t", "z_wrist_t1",
                               "z_global_t1", "action"))

    # Split train/val (gardons quelques démos en val)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    val_size = max(1, n // 5)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    fusion.to(device).eval()        # fusion gelée (entraînée en Phase A)
    for p in fusion.parameters():
        p.requires_grad = False
    predictor.to(device).train()    # predictor en train mode pour LoRA
    raw_predictor = predictor
    predictor = maybe_compile(predictor, use_compile)

    # === Optimiser uniquement les params LoRA ===
    lora_params = get_lora_parameters(raw_predictor)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"\n  Params LoRA à entraîner : {n_lora:,}")

    optimizer = make_adamw(lora_params, args.lr, 1e-4, device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    def run_batch(idx):
        zw_t, zg_t, zw_t1, zg_t1, act = fetch(idx)
        with autocast_ctx(device, amp_dtype):
            with torch.no_grad():
                z_t  = fusion(zw_t, zg_t)
                z_t1 = fusion(zw_t1, zg_t1)
            pred = predictor(z_t, act)
        return nn.functional.mse_loss(pred.float(), z_t1.float())

    # === Training loop ===
    print(f"\n=== Training LoRA sur démos réelles ===\n")
    history = []
    best_val = float("inf")

    for epoch in range(args.n_epochs):
        predictor.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        train_losses = []
        reset_peak_vram()
        t_epoch = time.time()

        for i in tqdm(range(0, len(perm), args.batch_size),
                      desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            loss = run_batch(perm[i:i + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        # Val
        predictor.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(val_idx), args.batch_size):
                val_losses.append(run_batch(val_idx[i:i + args.batch_size]).item())

        val_loss = sum(val_losses) / len(val_losses)
        scheduler.step()
        epoch_s = time.time() - t_epoch
        vram = peak_vram_gb() if device == "cuda" else 0.0
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "epoch_seconds": epoch_s, "peak_vram_gb": vram})
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"[{epoch_s:.0f}s, VRAM pic {vram:.1f} GB]")

        if val_loss < best_val:
            best_val = val_loss
            # Sauvegarder le predictor complet (predictor + LoRA inside)
            torch.save({
                "predictor_state_dict": unwrap(raw_predictor).state_dict(),
                "fusion_state_dict":    fusion.state_dict(),
                "predictor_config":     pred_cfg.__dict__,
                "fusion_name":          ckpt["fusion_name"],
                "lora_config":          lora_cfg.__dict__,
                "embed_dim":            ckpt["embed_dim"],
                "action_dim":           ckpt["action_dim"],
                "epoch":                epoch,
                "val_loss":             val_loss,
                "history":              history,
            }, out_path)
            print(f"  [Best val, saved]")

    print(f"\n=== LoRA training terminé ===")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Checkpoint    : {out_path}")


if __name__ == "__main__":
    main()
