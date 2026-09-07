"""Vérifie le runtime et un pas d'entraînement SDPA/AdamW, sans téléchargement."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.config import load_config, setup_hardware, log_environment
from src.device import (validate_cuda_runtime, get_device, resolve_amp_dtype,
                        autocast_ctx, make_adamw)
from src.fusion import make_fusion
from src.predictor import PredictorConfig, WorldModelPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    validate_cuda_runtime(require_cuda=args.require_cuda)
    hw = setup_hardware(load_config(ROOT / args.config))
    log_environment()
    device = get_device()
    amp = resolve_amp_dtype(device, hw["precision"])
    fusion = make_fusion("cross_attn_bd", dim=64).to(device)
    model = WorldModelPredictor(PredictorConfig(
        embed_dim=64, action_dim=6, n_layers=1, n_heads=4, ffn_dim=128,
    )).to(device)
    params = list(fusion.parameters()) + list(model.parameters())
    opt = make_adamw(params, 1e-4, 1e-4, device)
    zw, zg = (torch.randn(2, 16, 64, device=device) for _ in range(2))
    action = torch.randn(2, 6, device=device)
    with autocast_ctx(device, amp):
        pred = model(fusion(zw, zg), action)
    loss = pred.float().square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not all(
        torch.isfinite(p.grad).all() for p in params if p.grad is not None
    ):
        raise RuntimeError("Loss ou gradients non finis lors du contrôle matériel.")
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[OK] Fusion + SDPA + backward + AdamW sur {device}, "
          f"autocast={amp or 'fp32'}.")


if __name__ == "__main__":
    main()
