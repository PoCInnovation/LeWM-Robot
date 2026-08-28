"""
Tests de l'adaptation GPU (RTX 4090) — CPU-only sauf les tests marqués GPU.

Lancer :  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.device import (autocast_ctx, choose_data_device, configure_backend,
                        make_adamw, maybe_compile, place_tensors,
                        resolve_amp_dtype, resolve_num_workers, unwrap)
from src.config import hardware_config
from src.planner import CEMConfig, CEMPlanner
from src.predictor import PredictorConfig, WorldModelPredictor
from src.lora import LoRAConfig, inject_lora, merge_lora


# ────────────────────────────────────────────────────────────────
# Helpers device
# ────────────────────────────────────────────────────────────────

class TestDeviceHelpers:
    def test_amp_dtype_cpu_is_none(self):
        assert resolve_amp_dtype("cpu", "auto") is None
        assert resolve_amp_dtype("cpu", "bf16") is None
        assert resolve_amp_dtype("cpu", "fp32") is None

    def test_amp_dtype_invalid(self):
        with pytest.raises(ValueError):
            resolve_amp_dtype("cpu", "int8")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU requis")
    def test_amp_dtype_cuda(self):
        assert resolve_amp_dtype("cuda", "fp32") is None
        assert resolve_amp_dtype("cuda", "fp16") is torch.float16
        if torch.cuda.is_bf16_supported():
            assert resolve_amp_dtype("cuda", "auto") is torch.bfloat16

    def test_autocast_ctx_disabled_on_cpu(self):
        x = torch.randn(4, 4)
        with autocast_ctx("cpu", None):
            assert (x @ x).dtype == torch.float32

    def test_num_workers(self):
        assert resolve_num_workers(3) == 3
        assert resolve_num_workers("5") == 5
        assert 0 <= resolve_num_workers("auto") <= 8
        assert resolve_num_workers(None) == resolve_num_workers("auto")

    def test_choose_data_device_cpu_fallbacks(self):
        assert choose_data_device(10 ** 9, "auto", "cpu").type == "cpu"
        assert choose_data_device(10 ** 9, "cpu", "cpu").type == "cpu"
        assert choose_data_device(10 ** 9, "cuda", "cpu").type == "cpu"

    def test_place_tensors_cpu(self):
        t = {"a": torch.randn(10, 4), "b": torch.randn(10)}
        out, dev = place_tensors(t, "auto", "cpu", verbose=False)
        assert dev.type == "cpu"
        assert torch.equal(out["a"], t["a"])
        assert torch.equal(out["a"][torch.tensor([1, 3])], t["a"][[1, 3]])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU requis")
    def test_place_tensors_cuda(self):
        t = {"a": torch.randn(10, 4)}
        out, dev = place_tensors(t, "cuda", "cuda", verbose=False)
        assert dev.type == "cuda" and out["a"].device.type == "cuda"
        # indexation avec un index CPU → reste sur le GPU
        assert out["a"][torch.tensor([0, 2])].device.type == "cuda"

    def test_configure_backend_noop_on_cpu(self):
        configure_backend(verbose=False)   # ne doit pas lever sans GPU

    def test_maybe_compile_disabled_returns_same(self):
        m = torch.nn.Linear(2, 2)
        assert maybe_compile(m, enabled=False) is m
        assert unwrap(m) is m

    def test_make_adamw_cpu(self):
        m = torch.nn.Linear(4, 4)
        opt = make_adamw(m.parameters(), 1e-3, 1e-4, "cpu")
        assert isinstance(opt, torch.optim.AdamW)

    def test_hardware_config_defaults_and_override(self):
        hw = hardware_config({})
        assert hw["precision"] == "auto" and hw["tf32"] is True
        hw = hardware_config({"hardware": {"precision": "fp32", "compile": True}})
        assert hw["precision"] == "fp32" and hw["compile"] is True
        assert hw["data_device"] == "auto"       # défaut conservé


# ────────────────────────────────────────────────────────────────
# CEM planner : chunks + autocast
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


class TestCEMGpuOptions:
    def test_chunked_rollout_matches_unchunked(self):
        """Le découpage en chunks ne change pas le résultat."""
        N, D, A = 8, 16, 6
        world = _LinearWorld(N, D, A)
        z0, zg = torch.randn(N, D), torch.randn(N, D)
        outs = []
        for chunk in (0, 7):
            cfg = CEMConfig(horizon=3, n_samples=32, n_elites=4, n_iterations=2,
                            action_dim=A, device="cpu", rollout_chunk=chunk,
                            precision="fp32")
            torch.manual_seed(0)
            outs.append(CEMPlanner(world, cfg).plan(z0, zg))
        assert torch.allclose(outs[0], outs[1], atol=1e-6)

    def test_precision_auto_is_off_on_cpu(self):
        planner = CEMPlanner(_LinearWorld(), CEMConfig(device="cpu", precision="auto"))
        assert planner.amp_dtype is None

    def test_cost_decreases_toward_reachable_goal(self):
        N, D, A = 8, 16, 6
        world = _LinearWorld(N, D, A)
        z0 = torch.randn(N, D)
        a_star = torch.tensor([1.0, -0.5, 0.7, 0.2, -1.0, 0.4])
        with torch.no_grad():
            z_goal = world(z0.unsqueeze(0), a_star.unsqueeze(0))[0]
        cfg = CEMConfig(horizon=1, n_samples=256, n_elites=16, n_iterations=4,
                        action_dim=A, action_low=-2.0, action_high=2.0,
                        cost_type="mse", device="cpu", rollout_chunk=100)
        torch.manual_seed(0)
        actions, diag = CEMPlanner(world, cfg).plan(z0, z_goal, return_diagnostics=True)
        assert diag["best_cost"][-1] <= diag["best_cost"][0]
        assert (actions[0] - a_star).abs().mean() < 0.5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU requis")
    def test_bf16_rollout_close_to_fp32(self):
        N, D, A = 8, 16, 6
        world = _LinearWorld(N, D, A).cuda()
        z0, zg = torch.randn(N, D), torch.randn(N, D)
        res = {}
        for prec in ("fp32", "auto"):
            cfg = CEMConfig(horizon=3, n_samples=64, n_elites=8, n_iterations=2,
                            action_dim=A, device="cuda", precision=prec)
            torch.manual_seed(0)
            res[prec] = CEMPlanner(world, cfg).plan(z0, zg).cpu()
        assert (res["fp32"] - res["auto"]).abs().mean() < 0.2


# ────────────────────────────────────────────────────────────────
# Predictor + LoRA sous autocast (chemin d'entraînement de 04/05)
# ────────────────────────────────────────────────────────────────

class TestPredictorTrainingPath:
    def _predictor(self):
        return WorldModelPredictor(PredictorConfig(
            embed_dim=32, action_dim=6, n_layers=2, n_heads=4, ffn_dim=64))

    def test_forward_backward_cpu(self):
        pred = self._predictor()
        opt = make_adamw(pred.parameters(), 1e-3, 1e-4, "cpu")
        z, a = torch.randn(3, 20, 32), torch.randn(3, 6)
        with autocast_ctx("cpu", None):
            out = pred(z, a)
        loss = torch.nn.functional.mse_loss(out.float(), z)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        assert out.shape == (3, 20, 32)

    def test_unwrap_state_dict_keys_clean(self):
        pred = self._predictor()
        keys = unwrap(maybe_compile(pred, enabled=False)).state_dict().keys()
        assert not any(k.startswith("_orig_mod.") for k in keys)

    def test_lora_merge_equivalence(self):
        torch.manual_seed(0)
        pred = self._predictor()
        inject_lora(pred, LoRAConfig(rank=4), verbose=False)
        for m in pred.modules():
            if hasattr(m, "lora_A"):
                torch.nn.init.normal_(m.lora_A, std=0.05)
                torch.nn.init.normal_(m.lora_B, std=0.05)
        pred.eval()
        z, a = torch.randn(2, 10, 32), torch.randn(2, 6)
        with torch.no_grad():
            before = pred(z, a)
            after = merge_lora(pred)(z, a)
        assert torch.allclose(before, after, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU requis")
    def test_bf16_autocast_training_step_cuda(self):
        pred = self._predictor().cuda()
        amp = resolve_amp_dtype("cuda", "auto")
        opt = make_adamw(pred.parameters(), 1e-3, 1e-4, "cuda")
        z, a = torch.randn(4, 20, 32, device="cuda"), torch.randn(4, 6, device="cuda")
        with autocast_ctx("cuda", amp):
            out = pred(z, a)
        loss = torch.nn.functional.mse_loss(out.float(), z)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        assert torch.isfinite(loss)
