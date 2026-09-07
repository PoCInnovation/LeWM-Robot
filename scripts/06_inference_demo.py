"""
Démo d'inférence end-to-end : image courante + image-goal → planning → action.

Charge :
    - DINOv3 encoder (figé, bf16 + SDPA sur GPU)
    - Fusion cross-cam (poids depuis le predictor checkpoint)
    - Predictor entraîné (Phase A sim + Phase B LoRA, depuis 05)
    - CEM planner (rollouts en autocast bf16, par chunks)

Et démontre :
    1. Tu donnes deux paires (wrist, front) — current et goal
    2. Le pipeline encode et fusionne
    3. Le planner CEM cherche une séquence d'actions
    4. Affichage de la séquence d'actions optimale

Pour tester sans robot réel : prend deux frames du dataset (current = t0, goal = t30
par exemple) et vérifie que le planner trouve des actions cohérentes.

GPU (RTX 5090) : utiliser --rollout-chunk pour borner la mémoire des
rollouts. Mesurer la latence réelle avec 07_benchmark.py.

Usage:
    python scripts/06_inference_demo.py [--lora-ckpt PATH] [--predictor-ckpt PATH]
                                          [--n-samples 200] [--rollout-chunk 0]
                                          [--precision auto]
"""

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.encoders import DINOv3Config, DINOv3Encoder
from src.fusion import make_fusion
from src.predictor import WorldModelPredictor, PredictorConfig
from src.lora import LoRAConfig, inject_lora
from src.planner import CEMPlanner, CEMConfig, MPCController
from src.config import load_config, set_seed, log_environment, setup_hardware
from src.device import peak_vram_gb, reset_peak_vram


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_predictor_with_lora(ckpt_path: Path,
                              device: str) -> tuple:
    """
    Charge un predictor avec ou sans LoRA depuis un checkpoint.

    Returns:
        predictor (avec LoRA si applicable), fusion, metadata dict
    """
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    # Reconstituer le predictor
    pred_cfg = PredictorConfig(**ckpt["predictor_config"])
    predictor = WorldModelPredictor(pred_cfg)

    # Si LoRA est dedans, on injecte d'abord (la state_dict aura les clés A et B)
    has_lora = "lora_config" in ckpt
    if has_lora:
        lora_cfg = LoRAConfig(**ckpt["lora_config"])
        predictor = inject_lora(predictor, lora_cfg, verbose=False)

    predictor.load_state_dict(ckpt["predictor_state_dict"])

    # Fusion
    fusion = make_fusion(ckpt["fusion_name"], dim=ckpt["embed_dim"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])

    predictor.to(device).eval()
    fusion.to(device).eval()

    metadata = {
        "fusion_name":     ckpt["fusion_name"],
        "embed_dim":       ckpt["embed_dim"],
        "action_dim":      ckpt["action_dim"],
        "has_lora":        has_lora,
        "epoch":           ckpt.get("epoch", "?"),
        "val_loss":        ckpt.get("val_loss", "?"),
    }
    return predictor, fusion, metadata


def encode_image_pair(encoder, fusion, wrist_img, front_img, device):
    """
    Encode (wrist, front) → patches fusionnés.

    Args:
        wrist_img : (3, H, W) float ou uint8
        front_img : (3, H, W) float ou uint8

    Returns:
        z_fused : (N, dim) latent fusionné
    """
    with torch.no_grad():
        if wrist_img.dim() == 3:
            wrist_img = wrist_img.unsqueeze(0)
            front_img = front_img.unsqueeze(0)

        z_wrist  = encoder.encode(wrist_img)      # transfert + preprocessing sur GPU
        z_front  = encoder.encode(front_img)
        z_fused  = fusion(z_wrist, z_front)
    return z_fused.squeeze(0)


def make_planner(predictor, action_dim, args, hw):
    cem_cfg = CEMConfig(
        horizon=args.horizon,
        n_samples=args.n_samples,
        n_elites=args.n_elites,
        n_iterations=args.n_iter,
        action_dim=action_dim,
        cost_type="cosine",
        precision=args.precision or hw["precision"],
        rollout_chunk=args.rollout_chunk,
    )
    planner = CEMPlanner(predictor, cem_cfg)
    print(f"[CEM] n_samples={args.n_samples} horizon={args.horizon} "
          f"iter={args.n_iter} chunk={args.rollout_chunk or 'all'} "
          f"autocast={planner.amp_dtype or 'off'}")
    return planner


def timed_plan(planner, z_current, z_goal):
    """Plan + latence (warmup pour ne pas mesurer l'init CUDA)."""
    if planner.device.type == "cuda":
        planner.plan(z_current, z_goal)
        torch.cuda.synchronize()
    reset_peak_vram()
    t0 = time.time()
    actions, diag = planner.plan(z_current, z_goal, return_diagnostics=True)
    if planner.device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) * 1000
    print(f"Planning latency : {latency_ms:.0f} ms  |  VRAM pic : {peak_vram_gb():.2f} GB")
    return actions, diag


def demo_from_dataset(args, encoder, fusion, predictor, device):
    """
    Démo en utilisant deux frames du dataset comme "current" et "goal".

    Cette démo simule le cas où l'on a un robot réel et une image cible.
    On prend frame_0 comme état courant et frame_K comme état goal.
    """
    from src.data import LeRobotDataConfig, LeRobotPairsDataset

    print("\n--- Démo : frames du dataset ---")
    data_cfg = LeRobotDataConfig(
        dataset_id=args.dataset_id,
        wrist_key=args.wrist_key,
        global_key=args.global_key,
    )
    dataset = LeRobotPairsDataset(data_cfg)
    print(f"Dataset chargé : {len(dataset)} paires disponibles")

    # Prendre deux frames d'un même épisode, séparées de --goal-offset
    base_idx = args.demo_idx
    goal_idx = min(base_idx + args.goal_offset, len(dataset) - 1)

    current_sample = dataset[base_idx]
    goal_sample = dataset[goal_idx]

    print(f"  État courant : idx={base_idx}, episode={current_sample['episode_idx']}")
    print(f"  État goal    : idx={goal_idx}, episode={goal_sample['episode_idx']}")
    if current_sample["episode_idx"] != goal_sample["episode_idx"]:
        print("  [ATTENTION] current et goal ne sont pas dans le même épisode — "
              "réduire --goal-offset ou changer --demo-idx.")

    z_current = encode_image_pair(
        encoder, fusion,
        current_sample["wrist_t"], current_sample["global_t"], device,
    )
    z_goal = encode_image_pair(
        encoder, fusion,
        goal_sample["wrist_t"], goal_sample["global_t"], device,
    )
    print(f"  z_current shape : {tuple(z_current.shape)}")
    print(f"  z_goal shape    : {tuple(z_goal.shape)}")

    return z_current, z_goal, current_sample["action"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictor-ckpt", default=None,
                        help="Checkpoint predictor + LoRA (priorité)")
    parser.add_argument("--lora-ckpt",
                        default="results/checkpoints/predictor_real.pt",
                        help="Checkpoint avec LoRA (depuis 05)")
    parser.add_argument("--fallback-ckpt",
                        default="results/checkpoints/predictor_simu.pt",
                        help="Si pas de LoRA, charger le predictor sim seul")
    parser.add_argument("--dataset-id", default=None,
                        help="Défaut : dataset.hf_id de la config")
    parser.add_argument("--wrist-key",  default=None)
    parser.add_argument("--global-key", default=None)
    parser.add_argument("--demo-idx",    type=int, default=0,
                        help="Frame de départ dans le dataset")
    parser.add_argument("--goal-offset", type=int, default=30,
                        help="Offset (en frames) entre current et goal")
    parser.add_argument("--horizon",     type=int, default=10)
    parser.add_argument("--n-samples",   type=int, default=200)
    parser.add_argument("--n-elites",    type=int, default=20)
    parser.add_argument("--n-iter",      type=int, default=3)
    parser.add_argument("--rollout-chunk", type=int, default=0,
                        help="Candidats déroulés par passe (0 = tous). "
                             "Ex: --n-samples 4000 --rollout-chunk 1000")
    parser.add_argument("--precision", default=None,
                        help="auto (bf16 sur GPU) / fp32 pour les rollouts CEM")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    hw = setup_hardware(cfg)
    log_environment()
    args.dataset_id = args.dataset_id or cfg["dataset"]["hf_id"]
    args.wrist_key = args.wrist_key or cfg["dataset"]["wrist_key"]
    args.global_key = args.global_key or cfg["dataset"]["global_key"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")

    # === Charger DINOv3 ===
    print("\n--- Chargement DINOv3 ---")
    enc_cfg = DINOv3Config(
        family=cfg["encoder"].get("family", "dinov3"),
        size=cfg["encoder"]["size"],
        image_size=cfg["encoder"]["image_size"],
        dtype=cfg["encoder"].get("dtype", "auto"),
        attn_implementation=cfg["encoder"].get("attn_implementation", "sdpa"),
    )
    encoder = DINOv3Encoder(enc_cfg)

    # === Charger le predictor (avec LoRA si dispo, sinon sans) ===
    print("\n--- Chargement du predictor ---")
    ckpt_path = resolve(args.predictor_ckpt or args.lora_ckpt)
    if not ckpt_path.exists():
        # Fallback sur le predictor sim seul
        ckpt_path = resolve(args.fallback_ckpt)

    if not ckpt_path.exists():
        print(f"\n[ATTENTION] Aucun checkpoint trouvé.")
        print(f"  Cherché : {args.lora_ckpt}, {args.fallback_ckpt}")
        print("\nCette démo va utiliser un predictor random init pour montrer")
        print("le pipeline. Les actions n'auront aucun sens fonctionnel,")
        print("c'est juste pour valider l'inférence end-to-end.\n")
        run_random_demo(encoder, args, device, cfg, hw)
        return

    predictor, fusion, meta = load_predictor_with_lora(ckpt_path, device)
    if encoder.embed_dim != meta["embed_dim"]:
        sys.exit(f"[ERREUR] embed_dim encodeur ({encoder.embed_dim}) != "
                 f"checkpoint ({meta['embed_dim']}) — mauvaise size/famille ?")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"  Fusion     : {meta['fusion_name']}")
    print(f"  LoRA       : {'OUI' if meta['has_lora'] else 'non'}")
    print(f"  Val loss   : {meta['val_loss']}")
    print(f"  Predictor params : {sum(p.numel() for p in predictor.parameters()):,}")

    # === Récupérer les latents current et goal ===
    z_current, z_goal, action_taken = demo_from_dataset(
        args, encoder, fusion, predictor, device)

    # === CEM Planner ===
    print("\n--- Planning avec CEM ---")
    planner = make_planner(predictor, meta["action_dim"], args, hw)
    actions, diag = timed_plan(planner, z_current, z_goal)
    print(f"Actions shape    : {tuple(actions.shape)}")
    print(f"Best cost final  : {diag['best_cost'][-1]:.5f}")
    print(f"Cost evolution   : {[round(c, 5) for c in diag['best_cost']]}")

    # === Comparaison avec l'action vraie au temps t0 ===
    print(f"\nAction prédite par CEM (step 0) : {actions[0].cpu().numpy().round(3)}")
    print(f"Action vraie dans le dataset    : {action_taken.numpy().round(3)}")
    print(f"Différence absolue              : "
          f"{(actions[0].cpu() - action_taken).abs().numpy().round(3)}")
    print("\nNote : Si predictor pas encore entraîné, la différence sera élevée.")


def run_random_demo(encoder, args, device, cfg, hw):
    """Démo avec un predictor random (pour valider le pipeline sans entraînement)."""
    from src.data import LeRobotDataConfig, LeRobotPairsDataset

    # Charger un dataset pour avoir des dims réelles
    data_cfg = LeRobotDataConfig(
        dataset_id=args.dataset_id,
        wrist_key=args.wrist_key,
        global_key=args.global_key,
    )
    dataset = LeRobotPairsDataset(data_cfg)

    # Prendre une frame pour estimer les dims
    sample = dataset[0]
    z_wrist = encoder.encode(sample["wrist_t"].unsqueeze(0))
    embed_dim = z_wrist.shape[-1]
    action_dim = sample["action"].shape[-1]

    # Predictor random
    pred_cfg = PredictorConfig(
        embed_dim=embed_dim,
        action_dim=action_dim,
        n_layers=4,
        n_heads=6 if embed_dim % 6 == 0 else 8,
    )
    predictor = WorldModelPredictor(pred_cfg).to(device).eval()

    # Fusion concat_view (baseline)
    fusion = make_fusion("concat_view", dim=embed_dim).to(device).eval()

    # Encoder current et goal
    current_sample = dataset[args.demo_idx]
    goal_sample = dataset[min(args.demo_idx + args.goal_offset, len(dataset) - 1)]
    z_current = encode_image_pair(
        encoder, fusion,
        current_sample["wrist_t"], current_sample["global_t"], device)
    z_goal = encode_image_pair(
        encoder, fusion,
        goal_sample["wrist_t"], goal_sample["global_t"], device)

    print(f"z_current : {tuple(z_current.shape)}")
    print(f"z_goal    : {tuple(z_goal.shape)}")

    # CEM (avec predictor random — juste pour valider le flow)
    planner = make_planner(predictor, action_dim, args, hw)
    actions, _ = timed_plan(planner, z_current, z_goal)
    print(f"\nActions générées : {tuple(actions.shape)}")
    print(f"Action[0]        : {actions[0].cpu().numpy().round(3)}")
    print("\n[OK] Pipeline d'inférence end-to-end fonctionnel.")
    print("(Predictor random → actions non significatives,")
    print(" lance 04 + 05 pour avoir un vrai predictor entraîné.)")


if __name__ == "__main__":
    main()
