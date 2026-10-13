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
import json
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
from src.data import split_train_val, describe_split
from src.latents import (ActionStats, LatentPairs, frameskip_pairs, load_encoded,
                         train_action_stats)
from src.device import (resolve_amp_dtype, make_adamw,
                        maybe_compile, unwrap, autocast_ctx,
                        peak_vram_gb, reset_peak_vram)
from src.tracking import Stopwatch, add_wandb_args, tracker_from_args


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
    parser.add_argument("--action-stats", default="ckpt", choices=["ckpt", "real"],
                        help="ckpt : stats d'actions du predictor (même domaine). "
                             "real : stats du train réel, quand le predictor vient "
                             "d'un domaine aux unités différentes (simu en [-100, 100], "
                             "robot en degrés) ; à combiner avec --train-action-embed.")
    parser.add_argument("--train-action-embed", action="store_true",
                        help="Entraîne aussi l'embedding d'action (Linear 6 → dim) : "
                             "il réapprend ce que signifie une action réelle.")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Fréquence (en batches) du log de la loss instantanée.")
    add_wandb_args(parser)
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
    # même écart entre frames que le predictor de 04
    frameskip = int(ckpt.get("frameskip", 1))
    real = frameskip_pairs(load_encoded(real_path), frameskip)
    if frameskip > 1:
        print(f"  Frameskip {frameskip} (du checkpoint) : {len(real['action'])} paires")
    if real["embed_dim"] != ckpt["embed_dim"]:
        print(f"[ERREUR] embed_dim des données réelles ({real['embed_dim']}) != "
              f"checkpoint ({ckpt['embed_dim']}) — encodeurs différents ?")
        sys.exit(1)
    latents = LatentPairs(real, args.data_device or hw["data_device"], device,
                          reserve_gb=float(hw["vram_reserve_gb"]))
    data_device = latents.device
    fetch = latents.fetch
    n = len(latents)
    print(f"  N paires : {n}")

    # Split train/val (gardons quelques démos en val)
    train_idx, val_idx = split_train_val(real, seed=cfg["seed"],
                                         val_fraction=float(cfg.get("dataset", {}).get("val_fraction", 0.2)))
    print(f"Split : {describe_split(real, train_idx, val_idx)}")

    # Par défaut, mêmes stats d'actions que le predictor de 04 : le LoRA adapte
    # un modèle qui a appris dans cet espace normalisé. Si 04 vient d'un autre
    # domaine (simu en [-100, 100], robot en degrés), ces stats ne veulent rien
    # dire ici : --action-stats real les recalcule sur le train réel.
    if "action_stats" not in ckpt:
        sys.exit("[ERREUR] Checkpoint sans action_stats : réentraîner 04 avec le "
                 "code actuel (actions normalisées).")
    if args.action_stats == "real":
        mode = ActionStats.from_dict(ckpt["action_stats"]).mode
        action_stats = train_action_stats(real, train_idx, mode=mode)
        print(f"  Stats d'actions : recalculées sur le train réel (mode {mode})")
    else:
        action_stats = ActionStats.from_dict(ckpt["action_stats"])
    latents.normalize_actions(action_stats)

    fusion.to(device).eval()        # fusion gelée (entraînée en Phase A)
    for p in fusion.parameters():
        p.requires_grad = False
    predictor.to(device).train()    # predictor en train mode pour LoRA
    raw_predictor = predictor
    predictor = maybe_compile(predictor, use_compile)

    # === Optimiser uniquement les params LoRA ===
    lora_params = get_lora_parameters(raw_predictor)
    n_lora = sum(p.numel() for p in lora_params)
    if args.train_action_embed:
        embed_params = list(raw_predictor.action_emb.parameters())
        for p in embed_params:
            p.requires_grad = True
        lora_params = lora_params + embed_params
        print(f"  + embedding d'action entraînable : "
              f"{sum(p.numel() for p in embed_params):,} params")
    print(f"\n  Params LoRA à entraîner : {n_lora:,}")

    optimizer = make_adamw(lora_params, args.lr, 1e-4, device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    def run_batch(idx, with_identity=False):
        zw_t, zg_t, zw_t1, zg_t1, act = fetch(idx)
        with autocast_ctx(device, amp_dtype):
            with torch.no_grad():
                z_t  = fusion(zw_t, zg_t)
                z_t1 = fusion(zw_t1, zg_t1)
            pred = predictor(z_t, act)
        loss = nn.functional.mse_loss(pred.float(), z_t1.float())
        if with_identity:
            return loss, nn.functional.mse_loss(z_t.float(), z_t1.float())
        return loss

    # === Suivi W&B ===
    run = tracker_from_args(
        args, default_name=f"lora-r{args.lora_rank}", cfg=cfg, root=ROOT,
        extra_config={
            "device": device,
            "amp_dtype": str(amp_dtype),
            "data_device": str(data_device),
            "compile": use_compile,
            "embed_dim": ckpt["embed_dim"],
            "fusion_name": ckpt["fusion_name"],
            "n_pairs": n,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "predictor_params": raw_predictor.count_parameters(),
            "lora_params": n_lora,
            "lora_trainable_fraction": round(n_lora / raw_predictor.count_parameters(), 5),
            "sim_checkpoint_val_loss": ckpt.get("val_loss"),
            "phase": "B_real",
        },
    )

    # === Training loop ===
    print(f"\n=== Training LoRA sur démos réelles ===\n")
    history = []
    best_val = float("inf")
    global_step = 0
    run_clock = Stopwatch()

    for epoch in range(args.n_epochs):
        predictor.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        train_losses = []
        grad_norms = []
        reset_peak_vram()
        t_epoch = time.time()

        for i in tqdm(range(0, len(perm), args.batch_size),
                      desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            loss = run_batch(perm[i:i + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0))
            optimizer.step()
            train_losses.append(loss.item())
            grad_norms.append(grad_norm)
            global_step += 1

            if args.log_every and global_step % args.log_every == 0:
                run.log({
                    "train/batch_loss": loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }, step=global_step)

        train_loss = sum(train_losses) / len(train_losses)

        # Val
        predictor.eval()
        val_losses, id_losses = [], []
        with torch.no_grad():
            for i in range(0, len(val_idx), args.batch_size):
                loss, id_loss = run_batch(val_idx[i:i + args.batch_size],
                                          with_identity=True)
                val_losses.append(loss.item())
                id_losses.append(id_loss.item())

        val_loss = sum(val_losses) / len(val_losses)
        id_loss = sum(id_losses) / len(id_losses)
        id_ratio = val_loss / id_loss if id_loss > 0 else float("nan")
        # Relevé avant scheduler.step(), sinon c'est le lr de l'epoch suivant.
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_s = time.time() - t_epoch
        vram = peak_vram_gb() if device == "cuda" else 0.0
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "val_identity": id_loss,
                        "epoch_seconds": epoch_s, "peak_vram_gb": vram})
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"(identité {id_loss:.5f}, ratio {id_ratio:.3f})  "
              f"[{epoch_s:.0f}s, VRAM pic {vram:.1f} GB]")

        run.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/identity_mse": id_loss,
            "val/ratio_vs_identity": id_ratio,
            "train/lr": current_lr,
            "train/grad_norm_mean": sum(grad_norms) / len(grad_norms),
            "train/grad_norm_max": max(grad_norms),
            "perf/epoch_seconds": epoch_s,
            "perf/samples_per_s": (len(train_idx) / epoch_s
                                   if epoch_s > 0 else 0.0),
            "perf/gpu_peak_gb": vram if device == "cuda" else None,
        }, step=global_step)

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
                "action_stats":         action_stats.to_dict(),
                "frameskip":            frameskip,
                "epoch":                epoch,
                "val_loss":             val_loss,
                "val_identity":         id_loss,
                "history":              history,
            }, out_path)
            print(f"  [Best val, saved]")
            run.log_artifact(
                out_path, name="predictor_real", artifact_type="model",
                metadata={"epoch": epoch, "val_loss": val_loss,
                          "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha},
            )

    total_seconds = run_clock.elapsed
    best = min(history, key=lambda h: h["val"])
    with open(out_path.with_name(out_path.stem + ".done.json"), "w") as f:
        json.dump({
            "best_val_loss": best_val,
            "best_ratio_vs_identity": best["val"] / best["val_identity"],
            "epochs": args.n_epochs,
            "best_epoch": best["epoch"],
            "epoch_seconds_mean": total_seconds / max(1, args.n_epochs),
            "train_seconds": total_seconds,
            "checkpoint": str(out_path),
        }, f, indent=2)
    run.log_summary({
        "best_val_loss": best_val,
        "best_ratio_vs_identity": best["val"] / best["val_identity"],
        "total_seconds": total_seconds,
        "epochs": args.n_epochs,
        "sim_to_real_gap": (best_val - ckpt["val_loss"]
                            if ckpt.get("val_loss") is not None else None),
    })
    run.finish()

    print(f"\n=== LoRA training terminé ===")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Durée         : {total_seconds / 60:.1f} min")
    print(f"Checkpoint    : {out_path}")
    if run.url:
        print(f"W&B           : {run.url}")


if __name__ == "__main__":
    main()
