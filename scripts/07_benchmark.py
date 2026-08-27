"""
Benchmark chronométré sur le GPU local (RTX 4090) → estimation précise du
temps de chaque étape du pipeline pour un run complet.

Principe : on lance de VRAIS mini-entraînements (mêmes modules, même boucle,
même précision que 03/04/05/06) sur quelques dizaines de steps, on mesure le
temps par step et le pic VRAM, puis on extrapole au dataset entier et au
nombre d'epochs demandé.

Ce qui est mesuré :
    1. Encodeur DINOv3 (02) : img/s sur images synthétiques (borne GPU) et,
       si --dataset-id est donné, sur le VRAI DataLoader (décodage vidéo
       inclus — c'est le goulot réel).
    2. Fusion + probe (03) : s/step pour chaque stratégie.
    3. Predictor (04) : 3 mini-trains (batch 32 / 64 / 128 par défaut) →
       s/epoch, temps total, VRAM pic, batch max recommandé.
    4. LoRA (05) : s/step.
    5. CEM (06) : latence pour n_samples ∈ {200, 1000, 2000}.

Sources des tailles :
    - latents réels : --encoded-data (défaut : paths.encoded_data si présent) ;
    - sinon synthétiques : num_patches/embed_dim déduits de la config encodeur,
      N paires = --n-pairs (défaut 15000).

Usage:
    python scripts/07_benchmark.py [--config configs/default.yaml]
        [--encoded-data results/encoded/encoded_data.pt] [--n-pairs 15000]
        [--dataset-id divisio74/duck_dataset_v3]      # mesure le décodage réel
        [--batch-sizes 32,64,128] [--steps 30] [--n-epochs 30]
        [--lora-epochs 20] [--fusion-epochs 30] [--output results/benchmark.json]
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.fusion import make_fusion
from src.probes import ActionProbe
from src.predictor import WorldModelPredictor, PredictorConfig
from src.lora import LoRAConfig, inject_lora, get_lora_parameters
from src.planner import CEMPlanner, CEMConfig
from src.config import load_config, set_seed, log_environment, setup_hardware
from src.device import (resolve_amp_dtype, resolve_num_workers, place_tensors,
                        make_adamw, autocast_ctx, peak_vram_gb, reset_peak_vram,
                        gpu_summary)

EMBED_DIMS = {"small": 384, "base": 768, "large": 1024, "giant": 1536}
PATCHES = {"dinov3": 196, "dinov2": 256}     # à 224 px : patch 16 vs 14


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def time_steps(fn, n_warmup: int, n_steps: int) -> float:
    """Temps moyen (s) d'un appel fn() après warmup, synchronisé CUDA."""
    for _ in range(n_warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        fn()
    sync()
    return (time.perf_counter() - t0) / n_steps


# ────────────────────────────────────────────────────────────────
# 1. Encodeur
# ────────────────────────────────────────────────────────────────

def bench_encoder(cfg, args, device, n_pairs, hw):
    print("\n" + "─" * 64 + "\n[1/5] Encodeur (02_encode_dataset)\n" + "─" * 64)
    from src.encoders import DINOv3Config, DINOv3Encoder
    enc_cfg = DINOv3Config(
        family=cfg["encoder"].get("family", "dinov3"),
        size=cfg["encoder"]["size"],
        image_size=cfg["encoder"]["image_size"],
        dtype=cfg["encoder"].get("dtype", "auto"),
        attn_implementation=cfg["encoder"].get("attn_implementation", "sdpa"),
    )
    try:
        reset_peak_vram()
        encoder = DINOv3Encoder(enc_cfg)
    except Exception as e:
        print(f"[skip] encodeur non chargeable ({type(e).__name__}) — "
              f"{str(e).splitlines()[0]}")
        return None

    bs = args.encoder_batch
    imgs = (torch.rand(bs, 3, 224, 224) * 255).byte()
    dt = time_steps(lambda: encoder.encode(imgs), 3, max(3, args.steps // 3))
    gpu_ips = bs / dt
    n_images = 4 * n_pairs                     # wrist/global × t/t+1
    res = {
        "dtype": str(encoder.dtype_).replace("torch.", ""),
        "embed_dim": encoder.embed_dim, "num_patches": encoder.num_patches,
        "gpu_img_per_s": gpu_ips, "peak_vram_gb": peak_vram_gb(),
        "n_images": n_images,
        "estimate_gpu_bound_s": n_images / gpu_ips,
    }
    print(f"  GPU seul ({res['dtype']}, batch {bs}) : {gpu_ips:.0f} img/s  "
          f"| VRAM pic {res['peak_vram_gb']:.2f} GB")
    print(f"  → {n_images:,} images : {fmt(res['estimate_gpu_bound_s'])} "
          f"(borne inférieure, sans décodage vidéo)")

    if args.dataset_id:
        try:
            from torch.utils.data import DataLoader
            from src.data import LeRobotDataConfig, LeRobotPairsDataset
            data_cfg = LeRobotDataConfig(
                dataset_id=args.dataset_id,
                wrist_key=cfg["dataset"]["wrist_key"],
                global_key=cfg["dataset"]["global_key"],
                action_key=cfg["dataset"]["action_key"],
                proprio_key=cfg["dataset"]["proprio_key"],
                delta_timesteps=cfg["dataset"]["delta_timesteps"],
                cache_dir=cfg["dataset"]["cache_dir"],
            )
            ds = LeRobotPairsDataset(data_cfg)
            nw = resolve_num_workers(hw["num_workers"])
            loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                                pin_memory=device == "cuda",
                                persistent_workers=nw > 0)
            it = iter(loader)
            n_meas = max(3, args.steps // 5)

            def step():
                b = next(it)
                for k in ("wrist_t", "global_t", "wrist_t1", "global_t1"):
                    encoder.encode(b[k])
            dt_real = time_steps(step, 2, n_meas)
            real_pairs_s = bs / dt_real
            res.update({
                "real_pairs": len(ds), "num_workers": nw,
                "real_pairs_per_s": real_pairs_s,
                "estimate_real_s": len(ds) / real_pairs_s,
            })
            print(f"  Pipeline réel (décodage {nw} workers + encode) : "
                  f"{real_pairs_s:.1f} paires/s ({4 * real_pairs_s:.0f} img/s)")
            print(f"  → {len(ds):,} paires du dataset : {fmt(res['estimate_real_s'])}"
                  + ("   [goulot = décodage vidéo CPU]"
                     if 4 * real_pairs_s < 0.5 * gpu_ips else ""))
        except Exception as e:
            print(f"  [skip] mesure sur dataset réel impossible : "
                  f"{type(e).__name__}: {str(e).splitlines()[0]}")
    else:
        print("  (passer --dataset-id pour mesurer le décodage vidéo réel)")
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


# ────────────────────────────────────────────────────────────────
# Latents (réels ou synthétiques)
# ────────────────────────────────────────────────────────────────

def load_latents(cfg, args, device, hw):
    path = Path(args.encoded_data) if args.encoded_data else ROOT / cfg["paths"]["encoded_data"]
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        d = torch.load(path, weights_only=False, map_location="cpu")
        n = len(d["action"])
        print(f"[Latents] réels : {path.name} — {n} paires, "
              f"{d['z_wrist_t'].shape[1]} patches × {d['embed_dim']}")
        tensors = {k: d[k] for k in ("z_wrist_t", "z_global_t", "z_wrist_t1",
                                     "z_global_t1", "action")}
        source = "real"
    else:
        fam = cfg["encoder"].get("family", "dinov3")
        P, D = PATCHES.get(fam, 196), EMBED_DIMS[cfg["encoder"]["size"]]
        # On ne matérialise que ce qu'il faut pour les steps (max batch × 4)
        n = min(args.n_pairs, max(args.batch_sizes) * 4)
        g = torch.Generator().manual_seed(0)
        tensors = {k: torch.randn(n, P, D, generator=g) for k in
                   ("z_wrist_t", "z_global_t", "z_wrist_t1", "z_global_t1")}
        tensors["action"] = torch.randn(n, 6, generator=g) * 50
        print(f"[Latents] synthétiques : {fam}-{cfg['encoder']['size']} → "
              f"{P} patches × {D}, extrapolation sur --n-pairs {args.n_pairs}")
        source = "synthetic"
    tensors, data_device = place_tensors(tensors, hw["data_device"], device)
    return tensors, data_device, source


def sample_idx(n_avail, bs):
    return torch.randint(0, n_avail, (bs,))


# ────────────────────────────────────────────────────────────────
# 2. Fusion + probe (03)
# ────────────────────────────────────────────────────────────────

def bench_fusion(cfg, args, device, amp, tensors, n_pairs):
    print("\n" + "─" * 64 + "\n[2/5] Fusion + probe (03_compare_fusion)\n" + "─" * 64)
    probe_cfg = cfg.get("probe", {})
    bs = int(probe_cfg.get("batch_size", 32))
    strategies = cfg.get("fusion", {}).get("strategies_to_test",
                                           ["concat", "concat_view", "cross_attn_bd", "late_cls"])
    D = tensors["z_wrist_t"].shape[-1]
    A = tensors["action"].shape[-1]
    n_avail = tensors["action"].shape[0]
    n_train = int(0.8 * n_pairs)
    steps_per_epoch = math.ceil(n_train / bs)
    total = 0.0
    per = {}
    for name in strategies:
        kwargs = {"dim": D, **({"action_dim": A} if name == "proprio_guided" else {})}
        fusion = make_fusion(name, **kwargs).to(device).train()
        probe = ActionProbe(dim=D, action_dim=A,
                            hidden=int(probe_cfg.get("hidden_dim", 512))).to(device)
        opt = make_adamw(list(fusion.parameters()) + list(probe.parameters()),
                         1e-3, 1e-4, device)

        def step():
            idx = sample_idx(n_avail, bs)
            zw = tensors["z_wrist_t"][idx].to(device, non_blocking=True)
            zg = tensors["z_global_t"][idx].to(device, non_blocking=True)
            act = tensors["action"][idx].to(device, non_blocking=True)
            with autocast_ctx(device, amp):
                pred = probe(fusion(zw, zg))
            loss = nn.functional.l1_loss(pred.float(), act)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        reset_peak_vram()
        dt = time_steps(step, 5, args.steps)
        est = dt * steps_per_epoch * args.fusion_epochs * 1.25   # +25 % val
        per[name] = {"s_per_step": dt, "estimate_s": est, "peak_vram_gb": peak_vram_gb()}
        total += est
        print(f"  {name:<15} {dt * 1000:6.1f} ms/step  → {args.fusion_epochs} epochs : "
              f"{fmt(est):>9}  | VRAM pic {peak_vram_gb():.2f} GB")
    print(f"  TOTAL 03 ({len(strategies)} stratégies) : {fmt(total)}")
    return {"batch_size": bs, "steps_per_epoch": steps_per_epoch,
            "per_strategy": per, "estimate_total_s": total}


# ────────────────────────────────────────────────────────────────
# 3. Predictor (04) — 3 mini-trains
# ────────────────────────────────────────────────────────────────

def bench_predictor(cfg, args, device, amp, tensors, n_pairs):
    print("\n" + "─" * 64 + f"\n[3/5] Predictor (04_train_predictor) — "
          f"{len(args.batch_sizes)} mini-trains de {args.steps} steps\n" + "─" * 64)
    D = tensors["z_wrist_t"].shape[-1]
    A = tensors["action"].shape[-1]
    n_avail = tensors["action"].shape[0]
    n_train, n_val = int(0.8 * n_pairs), n_pairs - int(0.8 * n_pairs)
    results = {}
    best = None
    for bs in args.batch_sizes:
        set_seed(cfg["seed"])
        fusion = make_fusion(args.fusion, dim=D).to(device).train()
        pred_cfg = PredictorConfig(embed_dim=D, action_dim=A, n_layers=args.n_layers,
                                   n_heads=12 if D % 12 == 0 else 8, ffn_dim=D * 4)
        predictor = WorldModelPredictor(pred_cfg).to(device).train()
        params = list(fusion.parameters()) + list(predictor.parameters())
        opt = make_adamw(params, 1e-4, 1e-4, device)

        def fetch(idx):
            return tuple(tensors[k][idx].to(device, non_blocking=True) for k in
                         ("z_wrist_t", "z_global_t", "z_wrist_t1", "z_global_t1", "action"))

        def train_step():
            zw, zg, zw1, zg1, act = fetch(sample_idx(n_avail, bs))
            with autocast_ctx(device, amp):
                pred = predictor(fusion(zw, zg), act)
                z1 = fusion(zw1, zg1)
            loss = nn.functional.mse_loss(pred.float(), z1.float())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

        @torch.no_grad()
        def val_step():
            zw, zg, zw1, zg1, act = fetch(sample_idx(n_avail, bs))
            with autocast_ctx(device, amp):
                pred = predictor(fusion(zw, zg), act)
                nn.functional.mse_loss(pred.float(), fusion(zw1, zg1).float())

        try:
            reset_peak_vram()
            dt_train = time_steps(train_step, 5, args.steps)
            dt_val = time_steps(val_step, 2, max(3, args.steps // 3))
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs:>4} : OUT OF MEMORY → batch max < {bs}")
            results[bs] = {"oom": True}
            del fusion, predictor, opt
            torch.cuda.empty_cache()
            continue
        vram = peak_vram_gb()
        s_epoch = dt_train * math.ceil(n_train / bs) + dt_val * math.ceil(n_val / bs)
        est = s_epoch * args.n_epochs
        results[bs] = {"s_per_train_step": dt_train, "s_per_val_step": dt_val,
                       "s_per_epoch": s_epoch, "estimate_s": est,
                       "peak_vram_gb": vram, "samples_per_s": bs / dt_train}
        print(f"  batch {bs:>4} : {dt_train * 1000:6.1f} ms/step "
              f"({bs / dt_train:6.0f} ech/s) | epoch {fmt(s_epoch):>9} | "
              f"{args.n_epochs} epochs : {fmt(est):>9} | VRAM pic {vram:.1f} GB")
        if best is None or est < results[best]["estimate_s"]:
            best = bs
        del fusion, predictor, opt
        if device == "cuda":
            torch.cuda.empty_cache()
    if best is not None:
        print(f"  → batch le plus rapide : {best} "
              f"({fmt(results[best]['estimate_s'])} pour {args.n_epochs} epochs, "
              f"predictor {args.n_layers} couches, fusion {args.fusion})")
    return {"fusion": args.fusion, "n_layers": args.n_layers, "n_epochs": args.n_epochs,
            "per_batch_size": {str(k): v for k, v in results.items()},
            "fastest_batch_size": best,
            "estimate_s": results[best]["estimate_s"] if best is not None else None}


# ────────────────────────────────────────────────────────────────
# 4. LoRA (05)
# ────────────────────────────────────────────────────────────────

def bench_lora(cfg, args, device, amp, tensors, n_real_pairs):
    print("\n" + "─" * 64 + "\n[4/5] LoRA (05_train_lora)\n" + "─" * 64)
    D = tensors["z_wrist_t"].shape[-1]
    A = tensors["action"].shape[-1]
    n_avail = tensors["action"].shape[0]
    bs = args.lora_batch
    fusion = make_fusion(args.fusion, dim=D).to(device).eval()
    for p in fusion.parameters():
        p.requires_grad = False
    pred_cfg = PredictorConfig(embed_dim=D, action_dim=A, n_layers=args.n_layers,
                               n_heads=12 if D % 12 == 0 else 8, ffn_dim=D * 4)
    predictor = inject_lora(WorldModelPredictor(pred_cfg), LoRAConfig(rank=8),
                            verbose=False).to(device).train()
    lora_params = get_lora_parameters(predictor)
    opt = make_adamw(lora_params, 5e-4, 1e-4, device)

    def step():
        idx = sample_idx(n_avail, bs)
        zw, zg, zw1, zg1, act = (tensors[k][idx].to(device, non_blocking=True) for k in
                                 ("z_wrist_t", "z_global_t", "z_wrist_t1", "z_global_t1", "action"))
        with autocast_ctx(device, amp):
            with torch.no_grad():
                z, z1 = fusion(zw, zg), fusion(zw1, zg1)
            pred = predictor(z, act)
        loss = nn.functional.mse_loss(pred.float(), z1.float())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0); opt.step()
    reset_peak_vram()
    dt = time_steps(step, 5, args.steps)
    n_train = int(0.8 * n_real_pairs)
    s_epoch = dt * math.ceil(n_train / bs) * 1.25
    est = s_epoch * args.lora_epochs
    print(f"  batch {bs} : {dt * 1000:.1f} ms/step | epoch {fmt(s_epoch)} | "
          f"{args.lora_epochs} epochs sur {n_real_pairs} paires réelles : {fmt(est)} "
          f"| VRAM pic {peak_vram_gb():.1f} GB")
    del fusion, predictor, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"batch_size": bs, "s_per_step": dt, "s_per_epoch": s_epoch,
            "n_real_pairs": n_real_pairs, "estimate_s": est}


# ────────────────────────────────────────────────────────────────
# 5. CEM (06)
# ────────────────────────────────────────────────────────────────

def bench_cem(cfg, args, device, tensors, hw):
    print("\n" + "─" * 64 + "\n[5/5] Planner CEM (06_inference_demo)\n" + "─" * 64)
    D = tensors["z_wrist_t"].shape[-1]
    A = tensors["action"].shape[-1]
    fusion = make_fusion(args.fusion, dim=D).to(device).eval()
    pred_cfg = PredictorConfig(embed_dim=D, action_dim=A, n_layers=args.n_layers,
                               n_heads=12 if D % 12 == 0 else 8, ffn_dim=D * 4)
    predictor = WorldModelPredictor(pred_cfg).to(device).eval()
    with torch.no_grad():
        z = fusion(tensors["z_wrist_t"][:1].to(device), tensors["z_global_t"][:1].to(device))[0]
    res = {}
    for n_samples in args.cem_samples:
        chunk = min(n_samples, 500)
        planner = CEMPlanner(predictor, CEMConfig(
            horizon=10, n_samples=n_samples, n_elites=max(10, n_samples // 10),
            n_iterations=3, action_dim=A, precision=hw["precision"],
            rollout_chunk=chunk))
        try:
            reset_peak_vram()
            dt = time_steps(lambda: planner.plan(z, z), 2, max(3, args.steps // 5))
        except torch.cuda.OutOfMemoryError:
            print(f"  n_samples {n_samples:>5} : OOM (réduire --rollout-chunk)")
            torch.cuda.empty_cache()
            continue
        res[str(n_samples)] = {"latency_ms": dt * 1000, "peak_vram_gb": peak_vram_gb(),
                               "rollout_chunk": chunk}
        print(f"  n_samples {n_samples:>5} (horizon 10, 3 iter, chunk {chunk}) : "
              f"{dt * 1000:7.1f} ms/plan → {1 / dt:5.1f} Hz | VRAM pic {peak_vram_gb():.1f} GB")
    return res


# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data", default=None,
                        help="Latents réels (défaut : paths.encoded_data si présent)")
    parser.add_argument("--n-pairs", type=int, default=15000,
                        help="Nb de paires du dataset complet (si pas de latents réels)")
    parser.add_argument("--n-real-pairs", type=int, default=None,
                        help="Nb de paires des démos réelles pour le LoRA (défaut : n-pairs)")
    parser.add_argument("--dataset-id", default=None,
                        help="Mesurer aussi le pipeline d'encodage réel (décodage vidéo)")
    parser.add_argument("--batch-sizes", default="32,64,128",
                        help="Batch sizes des mini-trains du predictor")
    parser.add_argument("--steps", type=int, default=30,
                        help="Steps mesurés par mini-train (après 5 de warmup)")
    parser.add_argument("--n-epochs", type=int, default=30, help="Epochs visés pour 04")
    parser.add_argument("--fusion-epochs", type=int, default=None,
                        help="Epochs de 03 (défaut : probe.n_epochs)")
    parser.add_argument("--lora-epochs", type=int, default=20)
    parser.add_argument("--lora-batch", type=int, default=32)
    parser.add_argument("--fusion", default="cross_attn_bd")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--encoder-batch", type=int, default=64)
    parser.add_argument("--cem-samples", default="200,1000,2000")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    args.cem_samples = [int(x) for x in args.cem_samples.split(",")]

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    hw = setup_hardware(cfg)
    log_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = resolve_amp_dtype(device, args.precision or hw["precision"])
    args.fusion_epochs = args.fusion_epochs or int(cfg.get("probe", {}).get("n_epochs", 30))
    if device == "cpu":
        print("[ATTENTION] Pas de GPU : les temps mesurés sont ceux du CPU, "
              "pas d'une 4090.")

    print("=" * 64)
    print(f"BENCHMARK — {gpu_summary()}")
    print(f"  autocast : {amp or 'off (fp32)'} | TF32 : "
          f"{torch.backends.cuda.matmul.allow_tf32 if device == 'cuda' else 'n/a'}")
    print("=" * 64)

    report = {"gpu": gpu_summary(), "precision": str(amp or "fp32"),
              "config": {"encoder": cfg["encoder"], "n_epochs": args.n_epochs,
                         "fusion_epochs": args.fusion_epochs,
                         "lora_epochs": args.lora_epochs, "steps": args.steps}}

    tensors, data_device, source = load_latents(cfg, args, device, hw)
    n_pairs = tensors["action"].shape[0] if source == "real" else args.n_pairs
    n_real = args.n_real_pairs or n_pairs
    report["latents"] = {"source": source, "n_pairs": n_pairs,
                         "data_device": str(data_device)}

    t_start = time.time()
    if not args.skip_encoder:
        report["encoder"] = bench_encoder(cfg, args, device, n_pairs, hw)
    report["fusion"] = bench_fusion(cfg, args, device, amp, tensors, n_pairs)
    report["predictor"] = bench_predictor(cfg, args, device, amp, tensors, n_pairs)
    report["lora"] = bench_lora(cfg, args, device, amp, tensors, n_real)
    report["cem"] = bench_cem(cfg, args, device, tensors, hw)

    # ── Résumé ──
    enc = report.get("encoder") or {}
    enc_est = enc.get("estimate_real_s", enc.get("estimate_gpu_bound_s"))
    enc_note = ("réel" if "estimate_real_s" in enc else
                "borne GPU, sans décodage" if enc else "non mesuré")
    rows = [
        ("02 encodage", enc_est, enc_note),
        ("03 fusions", report["fusion"]["estimate_total_s"],
         f"{len(report['fusion']['per_strategy'])} strat. × {args.fusion_epochs} epochs"),
        ("04 predictor", report["predictor"]["estimate_s"],
         f"{args.n_epochs} epochs, batch {report['predictor']['fastest_batch_size']}"),
        ("05 LoRA", report["lora"]["estimate_s"],
         f"{args.lora_epochs} epochs, {n_real} paires réelles"),
    ]
    total = sum(r[1] for r in rows if r[1] is not None)
    print("\n" + "=" * 64)
    print(f"ESTIMATION RUN COMPLET — {n_pairs:,} paires, "
          f"{cfg['encoder'].get('family', 'dinov3')}-{cfg['encoder']['size']}, "
          f"latents {source}")
    print("=" * 64)
    for name, secs, note in rows:
        print(f"  {name:<14} {fmt(secs) if secs is not None else '   n/a':>10}   ({note})")
    print(f"  {'TOTAL':<14} {fmt(total):>10}")
    if report["cem"]:
        k = next(iter(report["cem"]))
        print(f"  CEM : {report['cem'][k]['latency_ms']:.0f} ms/plan à {k} candidats")
    print(f"\n(benchmark exécuté en {fmt(time.time() - t_start)})")
    report["summary"] = {name: secs for name, secs, _ in rows}
    report["summary"]["total_s"] = total

    out = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Rapport : {out}")


if __name__ == "__main__":
    main()
