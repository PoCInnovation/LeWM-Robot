"""
Diagnostic CEM : pourquoi le planner fait pire que l'action moyenne ?

Sur les mêmes départs que 08 (épisodes de val, goal = frame t+H·frameskip) :
  A. Le coût sait-il reconnaître les bonnes actions ?
     - rang des actions de la démo parmi 1000 candidats tirés comme au
       1er tour du CEM (N(0,1) bornés) : 0 = meilleur, 0,5 = hasard
     - % de départs où la démo bat l'action nulle (moyenne)
     - MAE (vs démo) des 10 % de candidats au meilleur coût / MAE de tous
       les candidats : < 1 → le coût oriente vers la démo
     - corrélation de Spearman entre distance à la démo et coût
     - coût démo vs coût identité (z_courant tel quel)
  B. Est-ce l'optimisation ? CEM 3 et 10 itérations : MAE ratio, et
     accord de signe plan/démo sur le 1er pas (articulations qui bougent).
  C. Par articulation : hausse du coût quand on décale la démo de ±1 σ.

Usage : python diag_cem.py <repo> <ckpt> <encoded> [H ...]
"""
import sys, json, time
import torch
import torch.nn.functional as F

repo, ckpt, enc = sys.argv[1], sys.argv[2], sys.argv[3]
horizons = [int(h) for h in sys.argv[4:]] or [1, 4]
sys.path.insert(0, repo)
sys.path.insert(0, repo + "/scripts")
from src.checkpoint import load_world_model
from src.config import load_config, set_seed
from src.data import split_train_val
from src.latents import LatentPairs, frameskip_pairs, load_encoded
from src.planner import CEMConfig, CEMPlanner
import importlib
plan_starts = importlib.import_module("08_eval_planner").plan_starts

dev = "cuda"
cfg = load_config(repo + "/configs/rtx4090.yaml")
set_seed(cfg["seed"])
pred, fusion, meta = load_world_model(ckpt, dev)
pred.eval(); fusion.eval()
stats = meta["action_stats"]
fs = meta["frameskip"]
data = frameskip_pairs(load_encoded(enc), fs)
_, val = split_train_val(data, seed=cfg["seed"], val_fraction=0.2)
lat = LatentPairs(data, "auto", dev, reserve_gb=5.0, verbose=False)
lat.normalize_actions(stats)
low, high = stats.normalized_bounds()
low, high = low.float().to(dev), high.float().to(dev)
std_deg = torch.tensor(stats.std)
NC = 1000
g = torch.Generator(device=dev).manual_seed(0)


def costs_of(z0, zg, acts):
    """acts (n, H, A) → (cos, mse) (n,) ; rollout bf16 comme le CEM."""
    out_c, out_m = [], []
    for s in range(0, acts.shape[0], 500):
        a = acts[s:s + 500]
        z = z0.unsqueeze(0).expand(a.shape[0], -1, -1).contiguous()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(a.shape[1]):
                z = pred(z, a[:, t])
        z = z.float()
        out_c.append(1 - F.cosine_similarity(z.flatten(1), zg.flatten().unsqueeze(0), dim=-1))
        out_m.append(((z - zg.unsqueeze(0)) ** 2).mean(dim=(-1, -2)))
    return torch.cat(out_c), torch.cat(out_m)


def spearman(x, y):
    rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])


report = {"ckpt": ckpt, "frameskip": fs, "mode": stats.mode}
for H in horizons:
    starts = plan_starts(data, val, H, stride=fs)
    gg = torch.Generator().manual_seed(cfg["seed"])
    pick = torch.randperm(len(starts), generator=gg)[:64].sort().values
    starts = [starts[i] for i in pick.tolist()]
    A = {k: {"rank": [], "beats_zero": [], "top10_mae_ratio": [], "spearman": [],
             "demo_vs_identity": []} for k in ("cos", "mse")}
    joint_sens = torch.zeros(meta["action_dim"])
    cem = {}
    planners = {it: CEMPlanner(pred, CEMConfig(
        horizon=H, n_samples=1000, n_elites=100, n_iterations=it,
        action_dim=meta["action_dim"], action_low=low.cpu(), action_high=high.cpu(),
        initial_std=1.0, cost_type="cosine", precision="auto", rollout_chunk=500))
        for it in (3, 10)}
    for it in planners:
        cem[it] = {"mae": [], "base": [], "sign": [], "ms": []}
    for rows in starts:
        zw, zg_, zw1, zg1, act = lat.fetch(torch.tensor(rows))
        with torch.no_grad():
            z0 = fusion(zw[:1], zg_[:1])[0]
            zG = fusion(zw1[-1:], zg1[-1:])[0]
        demo = act.unsqueeze(0)                                  # (1, H, A)
        cand = (torch.randn(NC, H, act.shape[-1], device=dev, generator=g)).clamp(low, high)
        allA = torch.cat([demo, torch.zeros_like(demo), cand])
        cc, mm = costs_of(z0, zG, allA)
        idc = 1 - F.cosine_similarity(z0.flatten(), zG.flatten(), dim=0)
        idm = ((z0 - zG) ** 2).mean()
        dist = (cand - demo).abs().mean(dim=(1, 2))
        for k, c, idv in (("cos", cc, idc), ("mse", mm, idm)):
            d, z, r = c[0], c[1], c[2:]
            A[k]["rank"].append(float((r < d).float().mean()))
            A[k]["beats_zero"].append(float(d < z))
            top = r.argsort()[:NC // 10]
            A[k]["top10_mae_ratio"].append(float(dist[top].mean() / dist.mean()))
            A[k]["spearman"].append(spearman(dist, r))
            A[k]["demo_vs_identity"].append(float(d / idv))
        # C : sensibilité par articulation (cosine), décalage ±1 σ du 1er pas
        base = cc[0]
        pert = []
        for j in range(act.shape[-1]):
            for sgn in (1, -1):
                p = demo.clone(); p[0, 0, j] += sgn
                pert.append(p.clamp(low, high))
        pc, _ = costs_of(z0, zG, torch.cat(pert))
        joint_sens += ((pc.view(-1, 2).mean(1) - base) / base).cpu()
        # B : CEM
        for it, pl in planners.items():
            torch.cuda.synchronize(); t0 = time.time()
            plan = pl.plan(z0, zG).float()
            torch.cuda.synchronize()
            cem[it]["ms"].append((time.time() - t0) * 1000)
            cem[it]["mae"].append(float((plan - act).abs().mean()))
            cem[it]["base"].append(float(act.abs().mean()))
            moving = act[0].abs() > 0.5
            if moving.any():
                cem[it]["sign"].append(float((torch.sign(plan[0][moving]) == torch.sign(act[0][moving])).float().mean()))
    mean = lambda x: sum(x) / len(x)
    rH = {"n_starts": len(starts)}
    for k in A:
        rH[k] = {m: round(mean(v), 4) for m, v in A[k].items()}
    rH["joint_cost_increase_pct"] = [round(float(x) * 100 / len(starts), 2) for x in joint_sens]
    for it in cem:
        rH[f"cem_it{it}"] = {"mae_ratio": round(mean(cem[it]["mae"]) / mean(cem[it]["base"]), 4),
                             "sign_agree_moving_joints": round(mean(cem[it]["sign"]), 4) if cem[it]["sign"] else None,
                             "ms_per_plan": round(mean(cem[it]["ms"]))}
    report[f"H{H}"] = rH
    print(f"== H={H}", json.dumps(rH, indent=1), flush=True)

print("REPORT", json.dumps(report))
