"""
Batterie de tests du pipeline (scope : adaptation machine) — données
synthétiques, zéro réseau, CPU-only.

Lancer :  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoded_data import (EncodedEpisodes, NormStats, PairsView,
                              SequenceView, merge_shard_metas, partial_stats,
                              split_episodes)
from src.fusion import FUSION_STRATEGIES, make_fusion
from src.lora import LoRAConfig, LoRALinear, get_lora_parameters, inject_lora, merge_lora
from src.planner import CEMConfig, CEMPlanner
from src.predictor import PredictorConfig, WorldModelPredictor


# ────────────────────────────────────────────────────────────────
# Helpers : dataset encodé synthétique
# ────────────────────────────────────────────────────────────────

def make_synthetic_encoded(tmp_path: Path, n_episodes=4, T=20, N=8, D=16,
                           A=6, P=6, seed=0) -> Path:
    """Écrit un dataset encodé v2 synthétique et renvoie son dossier."""
    g = torch.Generator().manual_seed(seed)
    out = tmp_path / "encoded"
    out.mkdir(parents=True, exist_ok=True)

    episodes, act_parts, prop_parts = [], [], []
    frame_start = 0
    for ep in range(n_episodes):
        action = torch.randn(T, A, generator=g) * 50 + 10   # échelle "degrés"
        proprio = torch.randn(T, P, generator=g) * 30
        z_wrist = torch.randn(T, N, D, generator=g)
        z_global = torch.randn(T, N, D, generator=g)
        torch.save({
            "episode_idx": ep,
            "z_wrist": z_wrist.half(),
            "z_global": z_global.half(),
            "action": action,
            "proprio": proprio,
            "frame_start": frame_start,
        }, out / f"ep_{ep:05d}.pt")
        episodes.append({"idx": ep, "length": T, "file": f"ep_{ep:05d}.pt"})
        act_parts.append(partial_stats(action))
        prop_parts.append(partial_stats(proprio))
        frame_start += T

    meta = {
        "embed_dim": D, "num_patches": N,
        "encoder_family": "synthetic", "encoder_size": "test",
        "image_size": 224, "dataset_id": "synthetic", "fps": 30, "seed": seed,
        "num_episodes": n_episodes, "total_frames": n_episodes * T,
        "episodes": episodes,
        "action_stats": NormStats.from_partials(act_parts).to_dict(),
        "proprio_stats": NormStats.from_partials(prop_parts).to_dict(),
        "storage_dtype": "float16", "format_version": 2,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f)
    return out


# ────────────────────────────────────────────────────────────────
# NormStats (stats sauvées dans meta.json par l'encodage shardé)
# ────────────────────────────────────────────────────────────────

class TestNormStats:
    def test_from_partials_matches_direct(self):
        a = torch.randn(100, 6) * 40 + 5
        b = torch.randn(50, 6) * 40 + 5
        stats = NormStats.from_partials([partial_stats(a), partial_stats(b)])
        full = torch.cat([a, b])
        assert torch.allclose(stats.mean, full.mean(0), atol=1e-4)
        assert torch.allclose(stats.std, full.std(0, unbiased=False), atol=1e-3)
        assert torch.allclose(stats.min, full.min(0).values)
        assert torch.allclose(stats.max, full.max(0).values)

    def test_roundtrip(self):
        x = torch.randn(30, 6) * 80
        stats = NormStats.from_partials([partial_stats(x)])
        assert torch.allclose(stats.denormalize(stats.normalize(x)), x, atol=1e-4)

    def test_serialization(self):
        x = torch.randn(30, 4)
        stats = NormStats.from_partials([partial_stats(x)])
        stats2 = NormStats.from_dict(stats.to_dict())
        assert torch.allclose(stats.mean, stats2.mean)
        assert torch.allclose(stats.std, stats2.std)


# ────────────────────────────────────────────────────────────────
# Split par épisode (utilitaire du format v2)
# ────────────────────────────────────────────────────────────────

class TestSplitEpisodes:
    def test_disjoint_and_complete(self):
        train, val = split_episodes(10, 0.2, seed=42)
        assert set(train) & set(val) == set()
        assert sorted(train + val) == list(range(10))
        assert len(val) == 2

    def test_deterministic(self):
        assert split_episodes(20, 0.25, 1) == split_episodes(20, 0.25, 1)
        assert split_episodes(20, 0.25, 1) != split_episodes(20, 0.25, 2)


# ────────────────────────────────────────────────────────────────
# Format encodé v2 : PairsView / SequenceView
# ────────────────────────────────────────────────────────────────

class TestEncodedViews:
    def test_pairs_alignment(self, tmp_path):
        """z_t1 de la paire (ep, t) == frame t+delta du même épisode."""
        root = make_synthetic_encoded(tmp_path)
        data = EncodedEpisodes(root)
        for delta in (1, 3):
            view = PairsView(data, [0, 2], delta=delta,
                             norm_actions=False, norm_proprio=False)
            item = view[5]
            pos, t = view.index[5]
            ep = data.episode(pos)
            assert torch.equal(item["z_wrist_t"], ep["z_wrist"][t].float())
            assert torch.equal(item["z_wrist_t1"],
                               ep["z_wrist"][t + delta].float())
            assert torch.equal(item["action"], ep["action"][t])

    def test_pairs_no_cross_episode(self, tmp_path):
        root = make_synthetic_encoded(tmp_path, n_episodes=3, T=10)
        data = EncodedEpisodes(root)
        view = PairsView(data, [0, 1, 2], delta=4)
        # 10 frames, delta 4 → 6 paires par épisode
        assert len(view) == 3 * 6
        for pos, t in view.index:
            assert t + 4 <= 9

    def test_raw_actions_by_request(self, tmp_path):
        """norm_actions=False renvoie bien les actions BRUTES (logique 04/05)."""
        root = make_synthetic_encoded(tmp_path)
        data = EncodedEpisodes(root)
        raw = PairsView(data, [0], delta=1, norm_actions=False)
        ep = data.episode(0)
        assert torch.equal(raw[0]["action"], ep["action"][0])
        # échelle "degrés" : les actions brutes ne sont PAS ~N(0,1)
        assert raw[0]["action"].abs().max() > 5

    def test_sequence_windows(self, tmp_path):
        root = make_synthetic_encoded(tmp_path, T=20)
        data = EncodedEpisodes(root)
        H, delta = 4, 2
        view = SequenceView(data, [0], horizon=H, delta=delta,
                            norm_actions=False)
        item = view[0]
        assert item["z_wrist"].shape[0] == H + 1
        assert item["actions"].shape[0] == H
        for pos, t in view.index:
            assert t + H * delta <= 19

    def test_missing_meta_message(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="meta.json"):
            EncodedEpisodes(tmp_path / "empty")


# ────────────────────────────────────────────────────────────────
# Merge de shards (job array → meta.json)
# ────────────────────────────────────────────────────────────────

class TestMergeShards:
    def _write_shard(self, out, idx, num, episodes, actions):
        parts = [partial_stats(a) for a in actions]
        merged = {
            "sum":   torch.stack([torch.tensor(p["sum"]) for p in parts]).sum(0).tolist(),
            "sumsq": torch.stack([torch.tensor(p["sumsq"]) for p in parts]).sum(0).tolist(),
            "min":   torch.stack([torch.tensor(p["min"]) for p in parts]).min(0).values.tolist(),
            "max":   torch.stack([torch.tensor(p["max"]) for p in parts]).max(0).values.tolist(),
            "count": sum(p["count"] for p in parts),
        }
        shard = {
            "shard_index": idx, "num_shards": num,
            "embed_dim": 16, "num_patches": 8,
            "encoder_family": "synthetic", "encoder_size": "test",
            "image_size": 224, "dataset_id": "synthetic", "fps": 30, "seed": 0,
            "storage_dtype": "float16",
            "episodes": episodes,
            "action_partial": merged, "proprio_partial": merged,
        }
        with open(out / f"shard_meta_{idx}.json", "w") as f:
            json.dump(shard, f)

    def test_merge_ok(self, tmp_path):
        a0, a1 = torch.randn(10, 6) * 30, torch.randn(15, 6) * 30
        self._write_shard(tmp_path, 0, 2,
                          [{"idx": 0, "length": 10, "file": "ep_00000.pt"}], [a0])
        self._write_shard(tmp_path, 1, 2,
                          [{"idx": 1, "length": 15, "file": "ep_00001.pt"}], [a1])
        meta = merge_shard_metas(tmp_path)
        assert meta["num_episodes"] == 2
        assert meta["total_frames"] == 25
        stats = NormStats.from_dict(meta["action_stats"])
        full = torch.cat([a0, a1])
        assert torch.allclose(stats.mean, full.mean(0), atol=1e-3)

    def test_merge_incomplete_fails(self, tmp_path):
        a0 = torch.randn(10, 6)
        self._write_shard(tmp_path, 0, 3,
                          [{"idx": 0, "length": 10, "file": "ep_00000.pt"}], [a0])
        with pytest.raises(RuntimeError, match="incomplets"):
            merge_shard_metas(tmp_path)


# ────────────────────────────────────────────────────────────────
# LoRA (comportement d'origine : toutes les Linear)
# ────────────────────────────────────────────────────────────────

class TestLoRA:
    def _predictor(self):
        return WorldModelPredictor(PredictorConfig(
            embed_dim=32, action_dim=6, n_layers=2, n_heads=4, ffn_dim=64))

    def test_wraps_all_linears(self):
        """Comportement d'origine : LoRA injecté dans toutes les nn.Linear."""
        pred = self._predictor()
        n_linear = sum(isinstance(m, torch.nn.Linear) for m in pred.modules())
        inject_lora(pred, LoRAConfig(rank=4), verbose=False)
        n_wrapped = sum(isinstance(m, LoRALinear) for m in pred.modules())
        assert n_wrapped == n_linear

    def test_init_is_identity(self):
        """A gaussien, B zéro → au début, LoRA ne change rien."""
        torch.manual_seed(0)
        pred = self._predictor()
        z, a = torch.randn(2, 10, 32), torch.randn(2, 6)
        pred.eval()
        with torch.no_grad():
            before = pred(z, a)
        inject_lora(pred, LoRAConfig(rank=4), verbose=False)
        with torch.no_grad():
            after = pred(z, a)
        assert torch.allclose(before, after, atol=1e-6)

    def test_merge_equivalence(self):
        torch.manual_seed(0)
        pred = self._predictor()
        inject_lora(pred, LoRAConfig(rank=4), verbose=False)
        for p in get_lora_parameters(pred):
            torch.nn.init.normal_(p, std=0.05)
        pred.eval()
        z, a = torch.randn(2, 10, 32), torch.randn(2, 6)
        with torch.no_grad():
            with_lora = pred(z, a)
        merged = merge_lora(pred)
        assert not any(isinstance(m, LoRALinear) for m in merged.modules())
        with torch.no_grad():
            after_merge = merged(z, a)
        assert torch.allclose(with_lora, after_merge, atol=1e-5)

    def test_state_dict_roundtrip(self):
        """Sauvegarde 05 → rechargement 06 (inject puis load)."""
        torch.manual_seed(0)
        pred = self._predictor()
        inject_lora(pred, LoRAConfig(rank=4), verbose=False)
        for p in get_lora_parameters(pred):
            torch.nn.init.normal_(p, std=0.05)
        sd = pred.state_dict()

        pred2 = self._predictor()
        inject_lora(pred2, LoRAConfig(rank=4), verbose=False)
        pred2.load_state_dict(sd)
        z, a = torch.randn(2, 10, 32), torch.randn(2, 6)
        pred.eval(); pred2.eval()
        with torch.no_grad():
            assert torch.allclose(pred(z, a), pred2(z, a))


# ────────────────────────────────────────────────────────────────
# Fusion
# ────────────────────────────────────────────────────────────────

class TestFusion:
    @pytest.mark.parametrize("name", list(FUSION_STRATEGIES))
    def test_shapes(self, name):
        B, N, D = 2, 8, 16
        kwargs = {"dim": D}
        if name == "proprio_guided":
            kwargs["action_dim"] = 6
        fusion = make_fusion(name, **kwargs)
        out = fusion(torch.randn(B, N, D), torch.randn(B, N, D),
                     torch.randn(B, 6))
        assert out.dim() == 3 and out.shape[0] == B and out.shape[2] == D


# ────────────────────────────────────────────────────────────────
# CEM planner (comportement d'origine)
# ────────────────────────────────────────────────────────────────

class _LinearWorld(torch.nn.Module):
    """Monde jouet : z' = z + W·a (l'action pousse le latent linéairement)."""

    def __init__(self, N=8, D=16, A=6):
        super().__init__()
        torch.manual_seed(7)
        self.W = torch.nn.Parameter(torch.randn(A, N * D) * 0.05,
                                    requires_grad=False)
        self.N, self.D = N, D

    def forward(self, z, a):
        delta = (a @ self.W).reshape(-1, self.N, self.D)
        return z + delta


class TestCEMPlanner:
    def test_bounds_respected(self):
        world = _LinearWorld()
        cfg = CEMConfig(horizon=3, n_samples=32, n_elites=4, n_iterations=2,
                        action_dim=6, device="cpu")
        planner = CEMPlanner(world, cfg)
        actions = planner.plan(torch.randn(8, 16), torch.randn(8, 16))
        assert (actions >= cfg.action_low - 1e-5).all()
        assert (actions <= cfg.action_high + 1e-5).all()

    def test_cost_decreases_toward_reachable_goal(self):
        """Sur un monde linéaire, le CEM doit faire décroître le coût."""
        N, D, A = 8, 16, 6
        world = _LinearWorld(N, D, A)
        z0 = torch.randn(N, D)
        a_star = torch.tensor([0.8, -0.5, 0.7, 0.2, -0.9, 0.4])
        with torch.no_grad():
            z_goal = world(z0.unsqueeze(0), a_star.unsqueeze(0))[0]

        cfg = CEMConfig(horizon=1, n_samples=256, n_elites=16, n_iterations=4,
                        action_dim=A, cost_type="mse", device="cpu")
        planner = CEMPlanner(world, cfg)
        torch.manual_seed(0)
        actions, diag = planner.plan(z0, z_goal, return_diagnostics=True)
        costs = diag["best_cost"]
        assert costs[-1] <= costs[0], "le coût CEM ne décroît pas"
        assert (actions[0] - a_star).abs().mean() < 0.5


# ────────────────────────────────────────────────────────────────
# Predictor (comportement d'origine)
# ────────────────────────────────────────────────────────────────

class TestPredictor:
    def test_forward_shape(self):
        pred = WorldModelPredictor(PredictorConfig(
            embed_dim=32, action_dim=6, n_layers=2, n_heads=4, ffn_dim=64))
        out = pred(torch.randn(3, 20, 32), torch.randn(3, 6))
        assert out.shape == (3, 20, 32)

    def test_rollout_shape(self):
        from src.predictor import rollout
        pred = WorldModelPredictor(PredictorConfig(
            embed_dim=32, action_dim=6, n_layers=2, n_heads=4, ffn_dim=64))
        traj = rollout(pred, torch.randn(2, 10, 32), torch.randn(2, 5, 6))
        assert traj.shape == (2, 6, 10, 32)
