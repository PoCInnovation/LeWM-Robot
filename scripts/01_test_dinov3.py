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
import torch

from src.encoders import DINOv3Config, DINOv3Encoder


def main():
    print("=" * 60)
    print("TEST 1 : Chargement et encodage DINOv3")
    print("=" * 60)

    # DINOv3-small (21M params) — accès approuvé par Meta.
    # Si accès non approuvé, fallback : family="dinov2" (public).
    config = DINOv3Config(family="dinov3", size="small", image_size=224)
    encoder = DINOv3Encoder(config)

    print(f"\nDevice utilisé : {encoder.device_}")
    print(f"Embed dim      : {encoder.embed_dim}")
    print(f"Num patches    : {config.num_patches}")

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
