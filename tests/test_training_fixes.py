"""Split par épisode, normalisation des actions, bornes CEM, ciblage LoRA — CPU only."""

import torch
import torch.nn as nn

from src.data import split_train_val
from src.latents import ActionStats, train_action_stats
from src.lora import LoRAConfig, LoRALinear, get_lora_parameters, inject_lora
from src.planner import CEMConfig, CEMPlanner
from src.predictor import PredictorConfig, WorldModelPredictor


def episodes_dataset(n_episodes=10, length=20, aug=True):
    """Paires consécutives (t, t+1) par épisode, + une passe augmentée."""
    ep, frame = [], []
    start = 0
    for e in range(n_episodes):
        for f in range(start, start + length - 1):
            ep.append(e)
            frame.append(f)
        start += length
    n = len(ep)
    reps = 2 if aug else 1
    return {
        "action": torch.randn(n * reps, 6),
        "pair_id": torch.arange(n).repeat(reps),
        "episode_idx": torch.tensor(ep).repeat(reps),
        "frame_idx": torch.tensor(frame).repeat(reps),
        "is_augmented": torch.cat([torch.zeros(n), torch.ones(n)][:reps]).long(),
    }


class TestEpisodeSplit:
    def test_no_frame_shared_between_train_and_val(self):
        data = episodes_dataset()
        train_idx, val_idx = split_train_val(data, seed=0)
        f = data["frame_idx"]
        train_frames = set(f[train_idx].tolist()) | set((f[train_idx] + 1).tolist())
        val_frames = set(f[val_idx].tolist()) | set((f[val_idx] + 1).tolist())
        assert not train_frames & val_frames

    def test_whole_episodes_and_clean_val(self):
        data = episodes_dataset(n_episodes=10)
        train_idx, val_idx = split_train_val(data, seed=0, val_fraction=0.2)
        ep = data["episode_idx"]
        assert len(torch.unique(ep[val_idx])) == 2
        assert not set(ep[train_idx].tolist()) & set(ep[val_idx].tolist())
        assert data["is_augmented"][val_idx].sum() == 0
        # tout le monde est quelque part sauf les augmentés des épisodes de val
        assert len(train_idx) + len(val_idx) == len(data["action"]) - len(val_idx)

    def test_reproducible(self):
        data = episodes_dataset()
        a = split_train_val(data, seed=3)
        b = split_train_val(data, seed=3)
        assert all(torch.equal(x, y) for x, y in zip(a, b))

    def test_legacy_without_episode(self):
        data = episodes_dataset()
        del data["episode_idx"]
        train_idx, val_idx = split_train_val(data, seed=0)
        assert len(train_idx) and len(val_idx)


class TestActionStats:
    def test_roundtrip_and_constant_joint(self):
        a = torch.randn(200, 6) * torch.tensor([6., 26., 27., 9., 10., 0.]) + 40
        stats = ActionStats.from_actions(a)
        n = stats.normalize(a)
        assert n[:, :5].std(0, unbiased=False).sub(1).abs().max() < 1e-4
        assert torch.isfinite(n).all()                       # pince figée : std 0
        assert torch.allclose(stats.denormalize(n), a, atol=1e-3)
        assert ActionStats.from_dict(stats.to_dict()) == stats

    def test_bounds_cover_the_data(self):
        a = torch.randn(500, 6) * 30
        stats = ActionStats.from_actions(a)
        low, high = stats.normalized_bounds()
        n = stats.normalize(a)
        assert (n >= low - 1e-5).all() and (n <= high + 1e-5).all()

    def test_train_stats_ignore_augmented_rows(self):
        data = episodes_dataset()
        n = len(data["action"]) // 2
        data["action"][n:] += 1000.0                         # bruit capteur extrême
        stats = train_action_stats(data, torch.arange(len(data["action"])))
        assert max(abs(m) for m in stats.mean) < 1.0


def test_cem_respects_per_joint_bounds():
    class ToyPredictor(nn.Module):
        def forward(self, z, a):
            return z + a.mean(-1, keepdim=True).unsqueeze(-1)

    low = torch.tensor([-1.0, -2.0, -0.5, 0.0, -3.0, 0.0])
    high = torch.tensor([1.0, 0.5, 2.0, 0.0, 3.0, 0.0])
    cfg = CEMConfig(horizon=3, n_samples=64, n_elites=8, n_iterations=3,
                    action_dim=6, action_low=low, action_high=high,
                    device="cpu", precision="fp32")
    actions = CEMPlanner(ToyPredictor(), cfg).plan(torch.zeros(4, 8), torch.ones(4, 8))
    assert (actions >= low - 1e-6).all() and (actions <= high + 1e-6).all()


def test_lora_targets_attention_only():
    pred = WorldModelPredictor(PredictorConfig(embed_dim=32, action_dim=6, n_layers=2,
                                               n_heads=4, ffn_dim=64))
    pred = inject_lora(pred, LoRAConfig(rank=2), verbose=False)
    wrapped = {name.rsplit(".", 1)[-1] for name, m in pred.named_modules()
               if isinstance(m, LoRALinear)}
    assert wrapped == {"q_proj", "k_proj", "v_proj", "o_proj"}
    n_wrapped = sum(isinstance(m, LoRALinear) for m in pred.modules())
    assert n_wrapped == 2 * 4                                  # 2 couches × 4 projections
    assert len(get_lora_parameters(pred)) == 2 * n_wrapped
    trainable = {id(p) for p in pred.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in get_lora_parameters(pred)}
