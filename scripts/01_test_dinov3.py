"""
Sanity check : vérifier que DINOv3 charge et encode correctement.

Ne dépend d'aucune donnée — utilise des images aléatoires.
Lance ce script en premier pour valider l'environnement.
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
from src.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--family", default=None,
                        choices=["dinov3", "dinov2"],
                        help="Surcharge encoder.family (dinov2 = public)")
    parser.add_argument("--size", default=None,
                        choices=["small", "base", "large", "giant"])
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)

    print("=" * 60)
    print("TEST 1 : Chargement et encodage de l'encodeur")
    print("=" * 60)

    # Encodeur de la config (DINOv3 gated ; fallback public : --family dinov2)
    config = DINOv3Config(
        family=args.family or cfg["encoder"].get("family", "dinov3"),
        size=args.size or cfg["encoder"]["size"],
        image_size=cfg["encoder"]["image_size"],
    )
    encoder = DINOv3Encoder(config)

    print(f"\nDevice utilisé : {encoder.device_}")
    print(f"Embed dim      : {encoder.embed_dim}")
    print(f"Num patches    : {encoder.num_patches} "
          f"(patch_size réel : {encoder.patch_size})")
    print(f"Register tokens: {encoder.num_register_tokens}")

    # Test avec un batch dummy
    print("\n--- Test encodage batch dummy ---")
    batch = torch.rand(4, 3, 224, 224)
    t0 = time.time()
    patches = encoder.encode(batch)
    dt = time.time() - t0
    print(f"Input  : {batch.shape}")
    print(f"Output : {patches.shape}")
    print(f"Time   : {dt:.3f}s  ({dt / batch.shape[0] * 1000:.1f} ms/image)")

    # Test avec CLS
    print("\n--- Test extraction CLS ---")
    patches, cls = encoder.encode(batch, return_cls=True)
    print(f"Patches : {patches.shape}")
    print(f"CLS     : {cls.shape}")

    # Test avec image uint8 (comme depuis LeRobot)
    print("\n--- Test avec image uint8 ---")
    batch_uint8 = (torch.rand(2, 3, 224, 224) * 255).byte()
    patches = encoder.encode(batch_uint8)
    print(f"Input  : {batch_uint8.shape} ({batch_uint8.dtype})")
    print(f"Output : {patches.shape}")

    print("\n[OK] DINOv3 fonctionne correctement.")
    print("\nProchaine étape : tester la fusion multi-camera")
    print("  python scripts/03_compare_fusion.py")


if __name__ == "__main__":
    main()
