"""
Sanity check : vérifier que DINOv3 charge et encode correctement.

Ne dépend d'aucune donnée — utilise des images aléatoires.
Lance ce script en premier pour valider l'environnement.

Sur GPU (RTX 5090) : mesure le débit en bf16 (défaut) et compare les latents
bf16 vs fp32 (similarité cosine) pour vérifier que la précision réduite ne
dégrade pas les features.

Usage:
    python scripts/01_test_dinov3.py [--family dinov2] [--size small]
                                      [--dtype auto|bf16|fp32] [--batch-size 64]
"""

import sys
from pathlib import Path

# Permet d'importer src/ depuis ce script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time
import argparse
import torch

from src.encoders import DINOv3Config, DINOv3Encoder
from src.config import load_config, log_environment, setup_hardware
from src.device import peak_vram_gb, reset_peak_vram


def _timed_encode(encoder, batch, n_repeat: int = 3) -> float:
    """Temps moyen (s) d'un encode(), avec warmup et synchronisation CUDA."""
    encoder.encode(batch)                                   # warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat):
        encoder.encode(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--family", default=None,
                        choices=["dinov3", "dinov2"],
                        help="Surcharge encoder.family (dinov2 = public)")
    parser.add_argument("--size", default=None,
                        choices=["small", "base", "large", "giant"])
    parser.add_argument("--dtype", default=None,
                        help="Surcharge encoder.dtype (auto/bf16/fp16/fp32)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Défaut : 64 sur GPU, 4 sur CPU")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    setup_hardware(cfg)
    log_environment()

    print("=" * 60)
    print("TEST 1 : Chargement et encodage de l'encodeur")
    print("=" * 60)

    on_gpu = torch.cuda.is_available()
    batch_size = args.batch_size or (64 if on_gpu else 4)

    # Encodeur de la config (DINOv3 gated ; fallback public : --family dinov2)
    config = DINOv3Config(
        family=args.family or cfg["encoder"].get("family", "dinov3"),
        size=args.size or cfg["encoder"]["size"],
        image_size=cfg["encoder"]["image_size"],
        dtype=args.dtype or cfg["encoder"].get("dtype", "auto"),
        attn_implementation=cfg["encoder"].get("attn_implementation", "sdpa"),
    )
    reset_peak_vram()
    encoder = DINOv3Encoder(config)

    print(f"\nDevice utilisé : {encoder.device_}  |  dtype poids : {encoder.dtype_}")
    print(f"Embed dim      : {encoder.embed_dim}")
    print(f"Num patches    : {encoder.num_patches} "
          f"(patch_size réel : {encoder.patch_size})")
    print(f"Register tokens: {encoder.num_register_tokens}")

    # Test avec un batch dummy
    print(f"\n--- Test encodage batch dummy (B={batch_size}) ---")
    batch = torch.rand(batch_size, 3, 224, 224)
    dt = _timed_encode(encoder, batch)
    patches = encoder.encode(batch)
    print(f"Input  : {tuple(batch.shape)}")
    print(f"Output : {tuple(patches.shape)} {patches.dtype}")
    print(f"Time   : {dt * 1000:.1f} ms/batch  "
          f"({dt / batch_size * 1000:.2f} ms/image, "
          f"{batch_size / dt:.1f} img/s)")
    assert patches.shape == (batch_size, encoder.num_patches, encoder.embed_dim), \
        "Nombre de patches inattendu (register tokens mal retirés ?)"

    # Test avec CLS
    print("\n--- Test extraction CLS ---")
    patches, cls = encoder.encode(batch, return_cls=True)
    print(f"Patches : {tuple(patches.shape)}")
    print(f"CLS     : {tuple(cls.shape)}")

    # Test avec image uint8 (comme depuis LeRobot)
    print("\n--- Test avec image uint8 ---")
    batch_uint8 = (torch.rand(2, 3, 224, 224) * 255).byte()
    p_u8 = encoder.encode(batch_uint8)
    p_f = encoder.encode(batch_uint8.float() / 255.0)
    print(f"Input  : {tuple(batch_uint8.shape)} ({batch_uint8.dtype})")
    print(f"Output : {tuple(p_u8.shape)}  |  max |uint8 - float| = "
          f"{(p_u8 - p_f).abs().max().item():.2e}")

    # bf16 vs fp32 : la précision réduite ne doit pas changer les latents
    if on_gpu and encoder.dtype_ != torch.float32:
        print("\n--- Précision réduite vs fp32 ---")
        ref = DINOv3Encoder(DINOv3Config(
            family=config.family, size=config.size,
            image_size=config.image_size, dtype=torch.float32,
            attn_implementation=config.attn_implementation))
        small = batch[:8]
        p_lo, p_hi = encoder.encode(small), ref.encode(small)
        cos = torch.nn.functional.cosine_similarity(
            p_lo.flatten(1), p_hi.flatten(1), dim=-1)
        dt_hi = _timed_encode(ref, batch)
        print(f"cos(bf16, fp32) : min={cos.min().item():.5f} "
              f"mean={cos.mean().item():.5f}  (attendu > 0.99)")
        print(f"Débit           : {encoder.dtype_} {batch_size / dt:.1f} img/s  "
              f"vs fp32 {batch_size / dt_hi:.1f} img/s  "
              f"(x{dt_hi / dt:.1f})")
        del ref
        if cos.min().item() < 0.99:
            print("[ATTENTION] latents bf16 trop éloignés de fp32 — "
                  "utiliser encoder.dtype: fp32")
        print(f"VRAM pic        : {peak_vram_gb():.2f} GB")

    print("\n[OK] L'encodeur fonctionne correctement.")
    print("\nProchaine étape : encoder le dataset")
    print("  python scripts/02_encode_dataset.py")


if __name__ == "__main__":
    main()
