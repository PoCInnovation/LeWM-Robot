"""Format de latents v2 (frames uniques, bf16) et chargement mmap — CPU only."""

import torch
import pytest

from src.config import load_config
from src.data import LeRobotPairsDataset
from src.latents import LEGACY_KEYS, LatentPairs, build_frame_index, load_encoded


def make_v2(n_frames=10, n_aug_pairs=3, P=4, D=8, seed=0):
    """Latents v2 : une passe propre sur des paires consécutives + une passe augmentée."""
    g = torch.Generator().manual_seed(seed)
    # 2 épisodes : frames 0-4 et 5-9 → paires (0..3) et (5..8)
    pair_frames = [0, 1, 2, 3, 5, 6, 7, 8]
    frames, idx_t, idx_t1 = build_frame_index(pair_frames, delta=1)
    n_clean = len(pair_frames)
    rows = len(frames) + 2 * n_aug_pairs
    aug_t = torch.arange(len(frames), len(frames) + n_aug_pairs)
    data = {
        "format_version": 2,
        "z_wrist": torch.randn(rows, P, D, generator=g).bfloat16(),
        "z_global": torch.randn(rows, P, D, generator=g).bfloat16(),
        "idx_t": torch.cat([idx_t, aug_t]),
        "idx_t1": torch.cat([idx_t1, aug_t + n_aug_pairs]),
        "action": torch.randn(n_clean + n_aug_pairs, 6, generator=g),
        "pair_id": torch.cat([torch.arange(n_clean), torch.arange(n_aug_pairs)]),
        "is_augmented": torch.cat([torch.zeros(n_clean), torch.ones(n_aug_pairs)]).long(),
        "embed_dim": D,
    }
    return data, frames


class TestBuildFrameIndex:
    def test_consecutive_pairs_share_frames(self):
        frames, idx_t, idx_t1 = build_frame_index([0, 1, 2, 3, 5, 6, 7, 8], delta=1)
        assert frames == list(range(10))           # 8 paires → 10 frames, pas 16
        assert idx_t.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
        assert idx_t1.tolist() == [1, 2, 3, 4, 6, 7, 8, 9]

    def test_delta_and_subsampling(self):
        frames, idx_t, idx_t1 = build_frame_index([0, 10, 20], delta=2)
        assert frames == [0, 2, 10, 12, 20, 22]
        assert [frames[i] for i in idx_t] == [0, 10, 20]
        assert [frames[i] for i in idx_t1] == [2, 12, 22]


class TestLatentPairs:
    def test_v2_fetch_matches_expanded_v1(self):
        data, _ = make_v2()
        v2 = LatentPairs(data, "cpu", "cpu", verbose=False)
        v1_data = {
            "z_wrist_t": data["z_wrist"][data["idx_t"]],
            "z_global_t": data["z_global"][data["idx_t"]],
            "z_wrist_t1": data["z_wrist"][data["idx_t1"]],
            "z_global_t1": data["z_global"][data["idx_t1"]],
            "action": data["action"],
        }
        v1 = LatentPairs(v1_data, "cpu", "cpu", verbose=False)
        assert len(v2) == len(v1) == 11
        idx = torch.tensor([0, 3, 7, 8, 10])
        for a, b in zip(v2.fetch(idx), v1.fetch(idx)):
            assert a.dtype == torch.float32
            assert torch.equal(a, b)
        for a, b in zip(v2.fetch_t(idx), v1.fetch_t(idx)):
            assert torch.equal(a, b)

    def test_t1_of_a_pair_is_t_of_the_next(self):
        data, _ = make_v2()
        pairs = LatentPairs(data, "cpu", "cpu", verbose=False)
        _, _, zw_t1, zg_t1, _ = pairs.fetch(torch.tensor([0]))
        zw_t, zg_t, _ = pairs.fetch_t(torch.tensor([1]))
        assert torch.equal(zw_t1, zw_t) and torch.equal(zg_t1, zg_t)

    def test_metadata(self):
        data, _ = make_v2(P=4, D=8)
        pairs = LatentPairs(data, "cpu", "cpu", verbose=False)
        assert (pairs.num_patches, pairs.embed_dim, pairs.action_dim) == (4, 8, 6)

    def test_materialize_is_v1(self):
        data, _ = make_v2()
        out = LatentPairs(data, "cpu", "cpu", verbose=False).materialize(torch.arange(4))
        assert set(out) == set(LEGACY_KEYS) | {"action"}
        assert out["z_wrist_t"].shape == (4, 4, 8)


def test_save_and_mmap_roundtrip(tmp_path):
    data, _ = make_v2()
    path = tmp_path / "enc.pt"
    torch.save(data, path)
    loaded = load_encoded(path)
    assert loaded["z_wrist"].dtype == torch.bfloat16
    assert torch.equal(loaded["z_wrist"], data["z_wrist"])
    a = LatentPairs(loaded, "cpu", "cpu", verbose=False).fetch(torch.arange(5))
    b = LatentPairs(data, "cpu", "cpu", verbose=False).fetch(torch.arange(5))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_uint8_conversion_is_lossless_for_video_frames():
    u8 = torch.arange(256, dtype=torch.uint8).repeat(3, 1, 1)
    as_float = u8.float() / 255.0                  # ce que renvoie lerobot
    assert torch.equal(LeRobotPairsDataset._to_uint8(as_float), u8)
    assert LeRobotPairsDataset._to_uint8(u8) is u8


def test_config_base_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "hardware:\n  tf32: true\n  vram_reserve_gb: 5\nseed: 42\n")
    (tmp_path / "machine.yaml").write_text(
        "base: base.yaml\nhardware:\n  vram_reserve_gb: 3\n")
    cfg = load_config(tmp_path / "machine.yaml")
    assert cfg == {"hardware": {"tf32": True, "vram_reserve_gb": 3}, "seed": 42}


def test_rtx4090_config_loads():
    from pathlib import Path
    from src.config import hardware_config
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "configs/rtx4090.yaml")
    hw = hardware_config(cfg)
    assert cfg["dataset"]["hf_id"]                 # hérité de default.yaml
    assert hw["power_limit_required"] is False
    assert hw["vram_reserve_gb"] == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU requis")
def test_reserve_decides_vram_hosting():
    from src.device import choose_data_device, vram_free_gb
    free = vram_free_gb() * 1e9
    assert choose_data_device(int(free * 0.5), "auto", "cuda", reserve_gb=0).type == "cuda"
    assert choose_data_device(int(free * 0.9), "auto", "cuda",
                              reserve_gb=free * 0.2 / 1e9).type == "cpu"
