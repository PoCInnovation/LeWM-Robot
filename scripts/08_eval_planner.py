"""
Évaluation hors ligne du planner CEM sur les épisodes de VALIDATION.

Pour régler le CEM (horizon, nb de candidats, itérations, coût), 06 ne suffit
pas : il planifie une seule paire d'images. Ici, sur des points de départ
tirés dans les épisodes de validation (jamais vus par 04) :

    état courant = frame t, goal = frame t+H (même épisode), H = --horizon
    → CEM planifie H actions, comparées aux H actions réellement jouées
      dans la démo entre t et t+H.

Métriques (espace normalisé, 1.0 = un écart-type de l'articulation) :
    mae_first : |action planifiée[0] - action démo[t]|     (celle que le MPC exécute)
    mae_seq   : même chose moyennée sur les H pas
    *_ratio   : divisé par la baseline « action moyenne du dataset » (= 0 en
                espace normalisé) — < 1 : le planner fait mieux que rien
    ms_per_plan : latence d'un plan complet

Tourne depuis les latents pré-encodés (sortie de 02) : ni encodeur, ni vidéo.

Usage:
    python scripts/08_eval_planner.py --encoded-data results/encoded/encoded_data.pt
        --predictor-ckpt results/checkpoints/predictor_simu.pt
        [--horizon 10] [--n-samples 1000] [--n-iter 3] [--cost cosine]
        [--n-starts 64] [--output results/cem_eval.json]
"""

import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.checkpoint import load_world_model
from src.config import load_config, set_seed, setup_hardware
from src.data import split_train_val
from src.latents import LatentPairs, frameskip_pairs, load_encoded
from src.planner import CEMConfig, CEMPlanner
from src.tracking import add_wandb_args, tracker_from_args


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def plan_starts(data: dict, val_idx: torch.Tensor, horizon: int, stride: int = 1):
    """
    Points de départ valides : les paires t, t+s, ..., t+(H-1)·s existent toutes
    dans le même épisode de validation (actions de la démo + goal = frame
    t+H·s), s = frameskip du predictor.

    Returns:
        liste de listes de H lignes de paires (dans les latents)
    """
    episode = torch.as_tensor(data["episode_idx"])
    frame = torch.as_tensor(data["frame_idx"])
    row_of = {}
    for r in val_idx.tolist():
        row_of.setdefault((int(episode[r]), int(frame[r])), r)
    starts = []
    for (ep, f), r in sorted(row_of.items()):
        rows = [row_of.get((ep, f + k * stride)) for k in range(horizon)]
        if all(x is not None for x in rows):
            starts.append(rows)
    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data", default=None,
                        help="Défaut : paths.encoded_data")
    parser.add_argument("--predictor-ckpt",
                        default="results/checkpoints/predictor_simu.pt")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Pas planifiés ; le goal est la frame t+horizon×frameskip")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-elites", type=int, default=None,
                        help="Défaut : 10 %% de --n-samples")
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--cost", default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--initial-std", type=float, default=1.0,
                        help="Écart-type initial (1.0 = celui du dataset)")
    parser.add_argument("--n-starts", type=int, default=64,
                        help="Points de départ évalués (répartis sur les épisodes de val)")
    parser.add_argument("--rollout-chunk", type=int, default=500)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--output", default="results/cem_eval.json")
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    hw = setup_hardware(cfg, verbose=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor, fusion, meta = load_world_model(resolve(args.predictor_ckpt), device)
    stats = meta["action_stats"]
    if stats is None:
        sys.exit("[ERREUR] Checkpoint sans action_stats : réentraîner 04.")

    frameskip = meta["frameskip"]
    data = frameskip_pairs(
        load_encoded(resolve(args.encoded_data or cfg["paths"]["encoded_data"])), frameskip)
    _, val_idx = split_train_val(data, seed=cfg["seed"],
                                 val_fraction=float(cfg.get("dataset", {}).get("val_fraction", 0.2)))
    latents = LatentPairs(data, "auto", device, reserve_gb=float(hw["vram_reserve_gb"]),
                          verbose=False)
    latents.normalize_actions(stats)

    starts = plan_starts(data, val_idx, args.horizon, stride=frameskip)
    if not starts:
        sys.exit(f"[ERREUR] Aucun point de départ avec {args.horizon} pas dans les épisodes de val.")
    g = torch.Generator().manual_seed(cfg["seed"])
    pick = torch.randperm(len(starts), generator=g)[:args.n_starts].sort().values
    starts = [starts[i] for i in pick.tolist()]

    low, high = stats.normalized_bounds()
    planner = CEMPlanner(predictor, CEMConfig(
        horizon=args.horizon, n_samples=args.n_samples,
        n_elites=args.n_elites or max(2, args.n_samples // 10),
        n_iterations=args.n_iter, action_dim=meta["action_dim"],
        action_low=low, action_high=high, initial_std=args.initial_std,
        cost_type=args.cost, precision=args.precision or hw["precision"],
        rollout_chunk=args.rollout_chunk))

    name = f"cem-H{args.horizon}-N{args.n_samples}-it{args.n_iter}-{args.cost}"
    print(f"[Eval CEM] {name} — {len(starts)} départs (épisodes de val), "
          f"checkpoint {Path(args.predictor_ckpt).name}")

    mae_first, mae_seq, base_first, base_seq, costs, latencies = [], [], [], [], [], []
    deg_first = []
    std = torch.tensor(stats.std)
    for rows in starts:
        idx = torch.tensor(rows)
        zw_t, zg_t, zw_t1, zg_t1, act = latents.fetch(idx)
        with torch.no_grad():
            z_current = fusion(zw_t[:1], zg_t[:1])[0]
            z_goal = fusion(zw_t1[-1:], zg_t1[-1:])[0]      # frame t+H
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        plan, diag = planner.plan(z_current, z_goal, return_diagnostics=True)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)
        err = (plan.float().cpu() - act.cpu()).abs()          # (H, action_dim)
        mae_first.append(float(err[0].mean()))
        deg_first.append(float((err[0] * std).mean()))     # en degrés, par articulation
        mae_seq.append(float(err.mean()))
        base_first.append(float(act[0].abs().mean()))
        base_seq.append(float(act.abs().mean()))
        costs.append(diag["best_cost"][-1])

    def mean(x):
        return sum(x) / len(x)

    result = {
        "name": name,
        "horizon": args.horizon, "frameskip": frameskip, "n_samples": args.n_samples,
        "n_elites": planner.config.n_elites, "n_iter": args.n_iter,
        "cost": args.cost, "initial_std": args.initial_std,
        "n_starts": len(starts),
        "mae_first": mean(mae_first), "mae_seq": mean(mae_seq),
        "mae_first_ratio": mean(mae_first) / mean(base_first),
        "mae_seq_ratio": mean(mae_seq) / mean(base_seq),
        "mae_first_deg": mean(deg_first),
        "final_cost": mean(costs),
        "ms_per_plan": mean(latencies),
        "checkpoint": str(args.predictor_ckpt),
    }
    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Eval CEM] MAE 1er pas {result['mae_first']:.3f} "
          f"(ratio vs action moyenne {result['mae_first_ratio']:.3f}, "
          f"~{result['mae_first_deg']:.1f}°)  séquence {result['mae_seq']:.3f} "
          f"(ratio {result['mae_seq_ratio']:.3f})  "
          f"{result['ms_per_plan']:.0f} ms/plan  → {out}")

    run = tracker_from_args(args, default_name=name, cfg=cfg, root=ROOT,
                            extra_config={"phase": "CEM_eval"})
    run.log_summary({k: v for k, v in result.items() if isinstance(v, (int, float))})
    run.finish()


if __name__ == "__main__":
    main()
