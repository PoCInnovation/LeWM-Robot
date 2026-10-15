"""Perte multi-step : séquences de paires, fetch_chain et rollout — CPU only."""

import torch

from src.latents import LatentPairs, frameskip_pairs, rollout_chains
from src.losses import multi_step_mse
from src.predictor import PredictorConfig, WorldModelPredictor
from test_latents import make_v2


def test_chains_follow_consecutive_pairs_within_an_episode():
    data, _ = make_v2()                       # épisodes : paires 0..3 et 4..7, + 3 augmentées
    chains = rollout_chains(data, torch.arange(len(data["action"])), 3)
    assert chains.tolist() == [[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]]
    # la frame d'arrivée d'un pas est la frame de départ du suivant
    assert torch.equal(data["idx_t1"][chains[:, :-1]], data["idx_t"][chains[:, 1:]])
    assert rollout_chains(data, torch.arange(11), 1).tolist() == [[i] for i in range(8)]
    assert len(rollout_chains(data, torch.arange(11), 5)) == 0


def test_chains_stay_inside_the_given_split():
    data, _ = make_v2()
    chains = rollout_chains(data, torch.tensor([0, 1, 2, 5, 6, 7]), 2)
    assert chains.tolist() == [[0, 1], [1, 2], [5, 6], [6, 7]]


def test_chains_with_frameskip_step_k_frames():
    data, _ = make_v2()
    fs = frameskip_pairs(dict(data, episode_idx=torch.tensor([0] * 4 + [1] * 4 + [0] * 3),
                              frame_idx=torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 0, 1, 2])), 2)
    # paires (t, t+2) : 0→2, 1→3, 2→4 puis 5→7, 6→8, 7→9 ; pas suivant = t+2
    chains = rollout_chains(fs, torch.arange(len(fs["action"])), 2)
    assert chains.tolist() == [[0, 2], [3, 5]]


def test_fetch_chain_matches_pairwise_fetch():
    data, _ = make_v2()
    lat = LatentPairs(data, "cpu", "cpu", verbose=False)
    chains = rollout_chains(data, torch.arange(11), 3)
    zw0, zg0, zw, zg, act = lat.fetch_chain(chains)
    assert zw.shape == (len(chains), 3, 4, 8) and act.shape == (len(chains), 3, 6)
    first = lat.fetch(chains[:, 0])
    assert torch.equal(zw0, first[0]) and torch.equal(zg0, first[1])
    for h in range(3):
        step = lat.fetch(chains[:, h])
        assert torch.equal(zw[:, h], step[2]) and torch.equal(zg[:, h], step[3])
        assert torch.equal(act[:, h], step[4])


def _predictor():
    torch.manual_seed(0)
    p = WorldModelPredictor(PredictorConfig(embed_dim=16, action_dim=6, n_layers=2,
                                            n_heads=4, ffn_dim=32, dropout=0.0,
                                            residual=True))
    for m in p.out_head.parameters():         # sortir de « rien ne bouge »
        torch.nn.init.normal_(m, std=0.1)
    return p


def test_rollout_feeds_predictions_back_and_one_step_is_plain_mse():
    p = _predictor()
    z0, act, tgt = torch.randn(3, 5, 16), torch.randn(3, 2, 6), torch.randn(3, 2, 5, 16)
    steps = multi_step_mse(p, z0, act, tgt, per_step=True)
    z1 = p(z0, act[:, 0])
    z2 = p(z1, act[:, 1])
    assert torch.allclose(steps, torch.stack([((z1 - tgt[:, 0]) ** 2).mean(),
                                              ((z2 - tgt[:, 1]) ** 2).mean()]))
    one = multi_step_mse(p, z0, act[:, :1], tgt[:, :1])
    assert torch.allclose(one, torch.nn.functional.mse_loss(z1, tgt[:, 0]))


def test_gradient_checkpointing_gives_the_same_gradients():
    z0, act, tgt = torch.randn(3, 5, 16), torch.randn(3, 4, 6), torch.randn(3, 4, 5, 16)
    grads = []
    for ckpt in (False, True):
        p = _predictor()
        loss = multi_step_mse(p, z0, act, tgt, grad_checkpoint=ckpt)
        loss.backward()
        grads.append((loss.detach(), [q.grad.clone() for q in p.parameters()]))
    assert torch.allclose(grads[0][0], grads[1][0])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(grads[0][1], grads[1][1]))
