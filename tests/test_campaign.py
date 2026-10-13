"""Développement des sweeps de la campagne et points de départ de 08 — CPU only."""

import importlib.util
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = load_script("campaign")
eval_planner = load_script("08_eval_planner")


def done(name, val, ratio, **params):
    return {"name": name, "params": params,
            "result": {"best_val_loss": val, "best_ratio_vs_identity": ratio}}


def test_grid_is_cartesian_with_fixed():
    stage = {"fixed": {"batch_size": 64}, "grid": {"lr": [1e-4, 3e-4], "n_layers": [4, 6]}}
    runs = campaign.expand(stage, {})
    assert len(runs) == 4
    assert all(r["batch_size"] == 64 for r in runs)
    assert {(r["lr"], r["n_layers"]) for r in runs} == {(1e-4, 4), (1e-4, 6), (3e-4, 4), (3e-4, 6)}


def test_from_best_keeps_top_configs_beating_identity():
    previous = {"A": [
        done("a", 0.10, 1.20, lr=1e-4, n_layers=4),   # plus basse mais ne bat pas l'identité
        done("b", 0.20, 0.80, lr=3e-4, n_layers=6),
        done("c", 0.15, 0.60, lr=1e-4, n_layers=6),
        {"name": "d", "params": {"lr": 3e-5, "n_layers": 4}},   # pas terminé
    ]}
    stage = {"from_best": {"stage": "A", "top": 2, "keep": ["lr", "n_layers"]},
             "fixed": {"batch_size": 64},
             "variants": [{"weight_decay": 0.01}, {"weight_decay": 0.1}]}
    runs = campaign.expand(stage, previous)
    assert [(r["lr"], r["n_layers"], r["weight_decay"]) for r in runs] == [
        (1e-4, 6, 0.01), (1e-4, 6, 0.1), (3e-4, 6, 0.01), (3e-4, 6, 0.1)]


def test_flags_and_names():
    params = {"batch_size": 64, "lr": 3e-5, "n_layers": 4, "weight_decay": 0.01}
    assert campaign.Campaign.flags(params) == [
        "--batch-size", "64", "--lr", "3e-05", "--n-layers", "4", "--weight-decay", "0.01"]
    assert campaign.run_name(params) == "lr3e-05-L4-wd0.01"


def test_shipped_campaign_expands():
    with open(ROOT / "configs/campaign_rtx4090.yaml") as f:
        spec = yaml.safe_load(f)
    stages = {s["name"]: s for s in spec["stages"]}
    assert len(campaign.expand(stages["A_screen"], {})) == 6
    assert campaign.Campaign.flags({"residual": True, "lr": 1e-4}) == ["--residual", "--lr", "0.0001"]
    assert campaign.Campaign.flags({"residual": False}) == []
    assert len(campaign.expand(stages["D_cem"], {})) == 12
    # B : une seule différence par variante par rapport à la référence
    ref, *others = stages["B_regul"]["variants"]
    assert all(sum(v[k] != ref[k] for k in ref) == 1 for v in others)


def test_seed_stage_reruns_best_config(capsys):
    with open(ROOT / "configs/campaign_rtx4090.yaml") as f:
        stages = {s["name"]: s for s in yaml.safe_load(f)["stages"]}
    params = dict(lr=1e-4, residual=True, n_layers=6, weight_decay=0.01, dropout=0.1)
    previous = {"B_regul": [done("best", 0.100, 0.5, batch_size=64, **params),
                            done("second", 0.101, 0.5, batch_size=64,
                                 **dict(params, n_layers=4))]}
    runs = campaign.expand(stages["B_seeds"], previous)
    assert [r["seed"] for r in runs] == [1, 2]
    assert all({k: r[k] for k in params} == params for r in runs)
    assert campaign.run_name(runs[0]).endswith("-seed1")

    c = campaign.Campaign.__new__(campaign.Campaign)
    c.results = dict(previous, B_seeds=[
        done("s1", 0.103, 0.5, **runs[0]), done("s2", 0.099, 0.5, **runs[1])])
    out = c.seed_spread(stages["B_seeds"])
    assert abs(out["spread"] - 0.004) < 1e-9 and not out["significant"]
    assert "équivalentes" in capsys.readouterr().out


def test_plan_starts_need_horizon_consecutive_val_pairs():
    # 2 épisodes de val : frames 0..5 (paires 0..4) et 10..12 (paires 10..11)
    frames = [0, 1, 2, 3, 4, 10, 11]
    data = {"episode_idx": torch.tensor([0] * 5 + [1] * 2),
            "frame_idx": torch.tensor(frames)}
    starts = eval_planner.plan_starts(data, torch.arange(7), horizon=3)
    assert [[frames[r] for r in s] for s in starts] == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
