"""
Pilote le SO-101 réel avec le world model : MPC (CEM) vers une image-goal.

    caméras wrist + front → DINOv3 → fusion → CEM sur le predictor
    → 1re action du plan → cible articulaire (état + delta) → bras
    → attente d'un pas du modèle (frameskip frames) → on recommence

Sécurité :
    - sans --execute, RIEN n'est envoyé au bras : la boucle tourne et affiche
      les cibles proposées (à faire d'abord, en bougeant le bras à la main) ;
    - chaque cible est bornée à ±--max-step-deg de l'état courant, ici ET
      dans le driver lerobot (max_relative_target) ;
    - Ctrl-C : sortie propre, couple coupé à la déconnexion.

Prérequis :
    - même bras (même calibration lerobot, --robot-id) et mêmes caméras
      (mêmes positions, 640 × 480) que pour l'enregistrement de
      duck_dataset_v3 : le modèle ne connaît que ce point de vue ;
    - un goal : --goal-file (capturé avec --save-goal, la scène mise à la
      main dans l'état voulu) ou --goal-episode (dernière frame d'une démo,
      demande les vidéos du dataset).

Usage:
    # 1. capturer une image-goal (scène dans l'état final voulu)
    python scripts/09_robot_mpc.py --port /dev/ttyACM0 --robot-id <id> \\
        --wrist-cam 0 --front-cam 2 --save-goal results/goals/canard.pt
    # 2. boucle à blanc (rien n'est envoyé)
    python scripts/09_robot_mpc.py --predictor-ckpt results/campaign/c3_real/best_predictor.pt \\
        --port /dev/ttyACM0 --robot-id <id> --wrist-cam 0 --front-cam 2 \\
        --goal-file results/goals/canard.pt
    # 3. exécution
    ... --execute
    # sans matériel (test de la chaîne) : --fake
"""

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.checkpoint import load_world_model
from src.config import load_config, set_seed, setup_hardware
from src.control import CAMERAS, FakeRobot, WorldModelAgent, image_to_chw, run_mpc
from src.planner import CEMConfig, CEMPlanner


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def camera_source(value: str):
    return int(value) if value.isdigit() else Path(value)


def make_robot(args):
    if args.fake:
        return FakeRobot(height=args.cam_height, width=args.cam_width)
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    cams = {name: OpenCVCameraConfig(index_or_path=camera_source(src), fps=args.fps,
                                     width=args.cam_width, height=args.cam_height)
            for name, src in (("wrist", args.wrist_cam), ("front", args.front_cam))}
    return SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, cameras=cams,
        max_relative_target=float(args.max_step_deg)))


def goal_from_episode(cfg, episode: int):
    """Images de la dernière frame d'un épisode du dataset d'entraînement."""
    from src.data import LeRobotDataConfig, LeRobotPairsDataset
    ds = LeRobotPairsDataset(LeRobotDataConfig(
        dataset_id=cfg["dataset"]["hf_id"], wrist_key=cfg["dataset"]["wrist_key"],
        global_key=cfg["dataset"]["global_key"], cache_dir=cfg["dataset"]["cache_dir"],
        delta_timesteps=cfg["dataset"]["delta_timesteps"], uint8_images=True))
    frames = [ds.pair_frame(i) for i in range(len(ds)) if ds.pair_episode(i) == episode]
    if not frames:
        sys.exit(f"[ERREUR] épisode {episode} absent du dataset")
    sample = ds.frame_sample(max(frames) + cfg["dataset"]["delta_timesteps"])
    return sample["wrist"], sample["global"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictor-ckpt", default="results/campaign/c3_real/best_predictor.pt")
    # matériel
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default=None,
                        help="Nom de la calibration lerobot du bras (celle de l'enregistrement)")
    parser.add_argument("--wrist-cam", default="0", help="Index ou chemin OpenCV")
    parser.add_argument("--front-cam", default="1", help="Index ou chemin OpenCV")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="fps du dataset (30)")
    parser.add_argument("--fake", action="store_true", help="Robot simulé (sans matériel)")
    # goal
    parser.add_argument("--goal-file", default=None, help="Goal capturé par --save-goal")
    parser.add_argument("--goal-episode", type=int, default=None,
                        help="Goal = dernière frame de cet épisode du dataset")
    parser.add_argument("--save-goal", default=None,
                        help="Capture les images courantes comme goal et quitte")
    # planner (réglages : voir best.json de la campagne / 08)
    parser.add_argument("--horizon", type=int, default=4, help="Pas de frameskip frames")
    parser.add_argument("--n-samples", type=int, default=300,
                        help="300 ≈ 0,15 s par pas d'horizon sur 4090 (1000 : ~0,5 s)")
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--cost", default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--no-warm-start", action="store_true")
    # boucle
    parser.add_argument("--execute", action="store_true",
                        help="Envoie réellement les cibles au bras")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-step-deg", type=float, default=15.0,
                        help="Écart max cible − état par articulation et par pas")
    parser.add_argument("--stop-cost", type=float, default=None,
                        help="Arrêt quand le coût CEM passe sous ce seuil")
    parser.add_argument("--log-dir", default="results/robot_runs")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    hw = setup_hardware(cfg, verbose=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    robot = make_robot(args)
    robot.connect()
    try:
        if args.save_goal:
            obs = robot.get_observation()
            out = resolve(args.save_goal)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({cam: image_to_chw(obs[cam]) for cam in CAMERAS}, out)
            print(f"[Goal] images wrist + front sauvées : {out}")
            return

        from src.encoders import DINOv3Config, DINOv3Encoder
        encoder = DINOv3Encoder(DINOv3Config(
            family=cfg["encoder"].get("family", "dinov3"), size=cfg["encoder"]["size"],
            image_size=cfg["encoder"]["image_size"],
            dtype=cfg["encoder"].get("dtype", "auto"),
            attn_implementation=cfg["encoder"].get("attn_implementation", "sdpa")))
        predictor, fusion, meta = load_world_model(resolve(args.predictor_ckpt), device)
        if encoder.embed_dim != meta["embed_dim"]:
            sys.exit(f"[ERREUR] embed_dim encodeur ({encoder.embed_dim}) != checkpoint "
                     f"({meta['embed_dim']}) : mauvaise taille d'encodeur dans la config")
        stats = meta["action_stats"]
        if stats is None:
            sys.exit("[ERREUR] checkpoint sans action_stats : réentraîner 04")
        low, high = stats.normalized_bounds()
        planner = CEMPlanner(predictor, CEMConfig(
            horizon=args.horizon, n_samples=args.n_samples,
            n_elites=max(2, args.n_samples // 10), n_iterations=args.n_iter,
            action_dim=meta["action_dim"], action_low=low, action_high=high,
            cost_type=args.cost, precision=hw["precision"],
            rollout_chunk=min(args.n_samples, 500)))
        agent = WorldModelAgent(encoder.encode, fusion, planner, stats,
                                max_step_deg=args.max_step_deg,
                                warm_start=not args.no_warm_start)

        if args.goal_file:
            goal = torch.load(resolve(args.goal_file), weights_only=False)
            goal_wrist, goal_front = goal["wrist"], goal["front"]
        elif args.goal_episode is not None:
            goal_wrist, goal_front = goal_from_episode(cfg, args.goal_episode)
        else:
            sys.exit("[ERREUR] --goal-file ou --goal-episode requis (ou --save-goal)")
        agent.set_goal(goal_wrist, goal_front)

        frameskip = meta["frameskip"]
        print(f"[MPC] {Path(args.predictor_ckpt).name} — actions {stats.mode}, "
              f"1 pas = {frameskip} frames ({frameskip / args.fps:.2f} s), "
              f"horizon {args.horizon} pas, CEM N={args.n_samples} it={args.n_iter}")
        if not args.execute:
            print("[MPC] À BLANC : rien n'est envoyé au bras (--execute pour exécuter)")
        log_dir = resolve(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"mpc_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        try:
            log = run_mpc(robot, agent, frameskip=frameskip, fps=args.fps,
                          max_steps=args.max_steps, execute=args.execute,
                          stop_cost=args.stop_cost, log_path=log_path)
        except KeyboardInterrupt:
            print("\n[MPC] interrompu")
            log = []
        if log:
            ms = sorted(e["plan_ms"] for e in log)
            print(f"[MPC] {len(log)} pas, planification médiane {ms[len(ms) // 2]:.0f} ms "
                  f"(un pas du modèle dure {frameskip / args.fps * 1000:.0f} ms) → {log_path}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
