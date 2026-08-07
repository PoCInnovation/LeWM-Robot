"""
Démo d'inférence end-to-end : état courant + état goal → CEM → actions.

Corrections vs l'ancienne version :
    - Le CEM planifie en espace d'actions NORMALISÉ (stats du checkpoint) et
      les actions sont dénormalisées à la sortie — l'ancien clamp [-1,1] sur
      des actions en degrés (±100) ne pouvait rien produire d'utilisable.
    - merge_lora() appelé après chargement : inférence ~30 % plus rapide.
    - Mode --from-encoded : tourne 100 % sur latents pré-encodés, AUCUN accès
      Hub/encodeur — c'est le mode pour l'éval offline massive sur cluster.
    - Le goal est garanti dans le MÊME épisode que l'état courant.

Usage:
    # Éval offline sur latents pré-encodés (recommandé sur cluster) :
    python scripts/06_inference_demo.py --from-encoded results/encoded/<name>
        [--ckpt results/checkpoints/predictor_real/best.pt]
        [--episode 0] [--start 0] [--goal-offset 30]

    # Depuis les images du dataset (nécessite encodeur + dataset) :
    python scripts/06_inference_demo.py --dataset-id <id|chemin local>
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.fusion import make_fusion
from src.predictor import WorldModelPredictor, PredictorConfig
from src.lora import LoRAConfig, inject_lora, merge_lora
from src.planner import CEMPlanner, CEMConfig
from src.encoded_data import EncodedEpisodes, NormStats
from src.config import load_config, set_seed, log_environment


def load_predictor(ckpt_path: Path, device: str):
    """
    Charge predictor + fusion + stats depuis un checkpoint (04 ou 05).

    Si le checkpoint contient un LoRA, il est injecté, chargé puis MERGÉ
    dans les poids (équivalent mathématique, ~30 % plus rapide).
    """
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    pred_cfg = PredictorConfig(**ckpt["predictor_config"])
    predictor = WorldModelPredictor(pred_cfg)

    has_lora = "lora_config" in ckpt
    if has_lora:
        predictor = inject_lora(predictor, LoRAConfig(**ckpt["lora_config"]),
                                verbose=False)
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    if has_lora:
        predictor = merge_lora(predictor)
        print("[Predictor] LoRA mergé dans les poids (inférence rapide).")

    fusion = make_fusion(ckpt["fusion_name"], dim=ckpt["embed_dim"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])

    predictor.to(device).eval()
    fusion.to(device).eval()

    if not ckpt.get("normalized_actions"):
        print("[ATTENTION] Checkpoint SANS normalisation d'actions (ancien "
              "format) — le CEM va probablement produire du bruit. "
              "Ré-entraîner avec le nouveau 04.")
        stats = None
    else:
        stats = NormStats.from_dict(ckpt["action_stats"])

    meta = {
        "fusion_name": ckpt["fusion_name"],
        "embed_dim":   ckpt["embed_dim"],
        "action_dim":  ckpt["action_dim"],
        "delta":       ckpt.get("delta", 1),
        "has_lora":    has_lora,
        "val_loss":    ckpt.get("val_loss", "?"),
    }
    return predictor, fusion, stats, meta


def make_planner(predictor, meta, stats, args):
    if stats is not None:
        low, high = stats.normalized_bounds()
        print(f"[CEM] Bornes normalisées par dim :")
        print(f"      low  = {low.numpy().round(2)}")
        print(f"      high = {high.numpy().round(2)}")
    else:
        low, high = -1.0, 1.0

    cem_cfg = CEMConfig(
        horizon=args.horizon,
        n_samples=args.n_samples,
        n_elites=args.n_elites,
        n_iterations=args.n_iter,
        action_dim=meta["action_dim"],
        action_low=low,
        action_high=high,
        cost_type="cosine",
    )
    return CEMPlanner(predictor, cem_cfg)


def demo_from_encoded(args, device):
    """Éval 100 % offline sur latents pré-encodés (aucun Hub, aucun encodeur)."""
    data = EncodedEpisodes(ROOT / args.from_encoded
                           if not Path(args.from_encoded).is_absolute()
                           else args.from_encoded)
    print(f"[Data] {data.summary()}")

    ckpt_path = resolve_ckpt(args)
    predictor, fusion, stats, meta = load_predictor(ckpt_path, device)
    print(f"[Checkpoint] {ckpt_path}  (LoRA: {meta['has_lora']}, "
          f"val_loss: {meta['val_loss']})")
    if data.embed_dim != meta["embed_dim"]:
        sys.exit(f"[ERREUR] embed_dim données ({data.embed_dim}) != "
                 f"checkpoint ({meta['embed_dim']})")

    # État courant et goal DANS LE MÊME ÉPISODE
    ep = data.episode(args.episode)
    T = ep["action"].shape[0]
    t0 = min(args.start, T - 2)
    t_goal = min(t0 + args.goal_offset, T - 1)
    print(f"[Démo] épisode pos={args.episode} (idx={ep['episode_idx']}), "
          f"frames {t0} → {t_goal} (sur {T})")

    with torch.no_grad():
        z_current = fusion(ep["z_wrist"][t0:t0 + 1].float().to(device),
                           ep["z_global"][t0:t0 + 1].float().to(device))[0]
        z_goal = fusion(ep["z_wrist"][t_goal:t_goal + 1].float().to(device),
                        ep["z_global"][t_goal:t_goal + 1].float().to(device))[0]

    planner = make_planner(predictor, meta, stats, args)

    t_start = time.time()
    actions_norm, diag = planner.plan(z_current, z_goal, return_diagnostics=True)
    latency = (time.time() - t_start) * 1000
    print(f"\n[CEM] latence : {latency:.0f} ms  |  "
          f"coût final : {diag['best_cost'][-1]:.5f}")
    print(f"[CEM] évolution du coût : {[round(c, 5) for c in diag['best_cost']]}")

    if stats is not None:
        actions = stats.denormalize(actions_norm.cpu())
    else:
        actions = actions_norm.cpu()

    true_action = ep["action"][t0]
    print(f"\nAction CEM (step 0, dénormalisée) : {actions[0].numpy().round(2)}")
    print(f"Action vraie du dataset (t0)       : {true_action.numpy().round(2)}")
    print(f"Différence absolue                 : "
          f"{(actions[0] - true_action).abs().numpy().round(2)}")
    print("\nNote : sans entraînement suffisant du predictor, l'écart sera "
          "élevé — regarder surtout la DÉCROISSANCE du coût CEM.")


def demo_from_images(args, device, cfg):
    """Démo depuis les images du dataset (nécessite encodeur + dataset)."""
    from src.encoders import DINOv3Config, DINOv3Encoder
    from src.data import LeRobotDataConfig, LeRobotFramesDataset

    ckpt_path = resolve_ckpt(args)
    predictor, fusion, stats, meta = load_predictor(ckpt_path, device)
    print(f"[Checkpoint] {ckpt_path}  (LoRA: {meta['has_lora']})")

    # Même défaut de famille que 02 (config) — pas de divergence silencieuse
    enc_cfg = DINOv3Config(
        family=cfg["encoder"].get("family", "dinov3"),
        size=cfg["encoder"]["size"],
        image_size=cfg["encoder"]["image_size"],
    )
    encoder = DINOv3Encoder(enc_cfg)
    if encoder.embed_dim != meta["embed_dim"]:
        sys.exit(f"[ERREUR] embed_dim encodeur ({encoder.embed_dim}) != "
                 f"checkpoint ({meta['embed_dim']}) — mauvaise size/famille ?")

    data_cfg = LeRobotDataConfig(
        dataset_id=args.dataset_id or cfg["dataset"]["hf_id"],
        wrist_key=cfg["dataset"]["wrist_key"],
        global_key=cfg["dataset"]["global_key"],
        cache_dir=cfg["dataset"]["cache_dir"],
    )
    dataset = LeRobotFramesDataset(data_cfg)

    # current et goal dans le même épisode
    ep_idx, start, length = dataset.episodes[args.episode]
    t0 = start + min(args.start, length - 2)
    t_goal = start + min(args.start + args.goal_offset, length - 1)
    cur, goal = dataset[t0], dataset[t_goal]
    print(f"[Démo] épisode {ep_idx}, frames {t0 - start} → {t_goal - start}")

    def encode_pair(sample):
        with torch.no_grad():
            zw = encoder.encode(sample["wrist"].unsqueeze(0).to(device))
            zg = encoder.encode(sample["global"].unsqueeze(0).to(device))
            return fusion(zw, zg)[0]

    z_current, z_goal = encode_pair(cur), encode_pair(goal)

    planner = make_planner(predictor, meta, stats, args)
    t_start = time.time()
    actions_norm, diag = planner.plan(z_current, z_goal, return_diagnostics=True)
    print(f"\n[CEM] latence : {(time.time() - t_start) * 1000:.0f} ms  |  "
          f"coût : {[round(c, 5) for c in diag['best_cost']]}")

    actions = stats.denormalize(actions_norm.cpu()) if stats else actions_norm.cpu()
    print(f"\nAction CEM (step 0)  : {actions[0].numpy().round(2)}")
    print(f"Action vraie (t0)    : {cur['action'].numpy().round(2)}")


def resolve_ckpt(args) -> Path:
    for candidate in ([args.ckpt] if args.ckpt else []) + [
            "results/checkpoints/predictor_real/best.pt",
            "results/checkpoints/predictor_sim/best.pt"]:
        p = ROOT / candidate if not Path(candidate).is_absolute() else Path(candidate)
        if p.exists():
            return p
    sys.exit("[ERREUR] Aucun checkpoint trouvé — lancer 04 (et 05) d'abord, "
             "ou passer --ckpt.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint (défaut: predictor_real/best.pt puis "
                             "predictor_sim/best.pt)")
    parser.add_argument("--from-encoded", default=None,
                        help="Dossier de latents pré-encodés → mode 100%% "
                             "offline (ni Hub ni encodeur)")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--episode", type=int, default=0,
                        help="Position de l'épisode de démo")
    parser.add_argument("--start", type=int, default=0,
                        help="Frame de départ dans l'épisode")
    parser.add_argument("--goal-offset", type=int, default=30,
                        help="Offset (frames) entre current et goal "
                             "(borné au même épisode)")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-elites", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    log_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")

    if args.from_encoded:
        demo_from_encoded(args, device)
    else:
        demo_from_images(args, device, cfg)


if __name__ == "__main__":
    main()
