"""Tests de src/tracking.py.

Tournent sans wandb installé. Le cas où wandb est présent est couvert avec un
faux module injecté dans sys.modules.
"""

import argparse
import sys
import types

import pytest

from src import tracking
from src.tracking import (
    DEFAULT_PROJECT,
    RunTracker,
    Stopwatch,
    add_wandb_args,
    environment_config,
    gpu_peak_memory_gb,
    parse_tags,
    tracker_from_args,
)


# --- faux wandb ----------------------------------------------------------- #

class FakeRun:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.logged = []
        self.summary = {}
        self.artifacts = []
        self.finished = 0
        self.name = "fake-run"
        self.url = "https://wandb.test/fake-run"

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self):
        self.finished += 1


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.files = []

    def add_file(self, path):
        self.files.append(path)


def install_fake_wandb(monkeypatch, init=None):
    """Injecte un faux module wandb et renvoie le dernier run créé."""
    holder = {}

    def _init(**kwargs):
        if init is not None:
            return init(**kwargs)
        holder["run"] = FakeRun(**kwargs)
        return holder["run"]

    module = types.ModuleType("wandb")
    module.init = _init
    module.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", module)
    return holder


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Neutralise les credentials de la machine."""
    for name in ("WANDB_API_KEY", "WANDB_MODE", "WANDB_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tracking, "_has_netrc_entry", lambda: False)


def make_args(**overrides):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=30)
    add_wandb_args(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# --- wandb absent ou en échec ---------------------------------- #

def test_no_wandb_installed_degrades_to_a_no_op(monkeypatch):
    """sys.modules["wandb"] = None fait lever ImportError à l'import."""
    monkeypatch.setitem(sys.modules, "wandb", None)

    tracker = RunTracker(project="x")
    assert tracker.active is False
    assert "non installé" in tracker.disabled_reason
    tracker.log({"a": 1.0})
    tracker.log_summary({"b": 2.0})
    tracker.log_artifact(__file__)
    tracker.finish()
    assert tracker.url is None


def test_explicitly_disabled_says_so():
    tracker = RunTracker(enabled=False)
    assert tracker.active is False
    assert "--no-wandb" in tracker.disabled_reason


def test_a_failing_init_never_propagates(monkeypatch):
    """Serveur injoignable : on continue sans logging."""
    def boom(**kwargs):
        raise RuntimeError("network unreachable")

    install_fake_wandb(monkeypatch, init=boom)
    tracker = RunTracker(project="x")
    assert tracker.active is False
    assert "init échouée" in tracker.disabled_reason
    tracker.log({"a": 1.0})  # ne lève pas


def test_a_failing_log_never_propagates(monkeypatch):
    holder = install_fake_wandb(monkeypatch)
    tracker = RunTracker(project="x", mode="offline")

    def boom(payload, step=None):
        raise RuntimeError("disk full")

    holder["run"].log = boom
    tracker.log({"a": 1.0})          # avalé
    tracker.log_summary({"b": 2.0})  # toujours fonctionnel


def test_the_context_manager_closes_the_run(monkeypatch):
    holder = install_fake_wandb(monkeypatch)
    with RunTracker(project="x", mode="offline") as tracker:
        assert tracker.active
    assert holder["run"].finished == 1
    assert tracker.active is False


def test_finish_is_idempotent(monkeypatch):
    """atexit, le with et un appel explicite peuvent tous le déclencher."""
    holder = install_fake_wandb(monkeypatch)
    tracker = RunTracker(project="x", mode="offline")
    tracker.finish()
    tracker.finish()
    tracker.finish()
    assert holder["run"].finished == 1


# --- le chemin actif ------------------------------------------------------ #

def test_metrics_reach_wandb(monkeypatch):
    holder = install_fake_wandb(monkeypatch)
    tracker = RunTracker(project="p", name="n", mode="offline")

    tracker.log({"train/loss": 0.5}, step=7)
    tracker.log_summary({"best": 0.1})

    assert holder["run"].logged == [({"train/loss": 0.5}, 7)]
    assert holder["run"].summary == {"best": 0.1}
    assert holder["run"].init_kwargs["project"] == "p"
    assert holder["run"].init_kwargs["name"] == "n"


def test_none_values_are_dropped(monkeypatch):
    """gpu_peak_gb vaut None en CPU."""
    holder = install_fake_wandb(monkeypatch)
    tracker = RunTracker(project="p", mode="offline")

    tracker.log({"a": 1.0, "perf/gpu_peak_gb": None})
    assert holder["run"].logged[0][0] == {"a": 1.0}

    tracker.log({"only": None})
    assert len(holder["run"].logged) == 1, "un payload vide ne doit rien envoyer"


def test_artifacts_are_opt_in(monkeypatch):
    """Opt-in : téléverser à chaque epoch ralentirait le training."""
    holder = install_fake_wandb(monkeypatch)

    off = RunTracker(project="p", mode="offline", log_artifacts=False)
    off.log_artifact(__file__)
    assert holder["run"].artifacts == []

    holder2 = install_fake_wandb(monkeypatch)
    on = RunTracker(project="p", mode="offline", log_artifacts=True)
    on.log_artifact(__file__, name="ckpt", metadata={"epoch": 3})
    assert len(holder2["run"].artifacts) == 1
    artifact = holder2["run"].artifacts[0]
    assert artifact.name == "ckpt"
    assert artifact.metadata == {"epoch": 3}
    assert artifact.files == [__file__]


def test_a_missing_checkpoint_is_skipped_silently(monkeypatch):
    holder = install_fake_wandb(monkeypatch)
    tracker = RunTracker(project="p", mode="offline", log_artifacts=True)
    tracker.log_artifact("results/checkpoints/does_not_exist.pt")
    assert holder["run"].artifacts == []


def test_without_credentials_it_goes_offline_rather_than_prompting(monkeypatch):
    """Une init interactive bloquerait un run non surveillé."""
    holder = install_fake_wandb(monkeypatch)
    RunTracker(project="p")
    assert holder["run"].init_kwargs["mode"] == "offline"


def test_an_api_key_lets_it_stay_online(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    holder = install_fake_wandb(monkeypatch)
    RunTracker(project="p")
    assert holder["run"].init_kwargs["mode"] is None, "mode None = online par défaut"


# --- construction depuis la CLI ------------------------------------------ #

def test_the_run_config_captures_hyperparameters_and_environment(monkeypatch):
    holder = install_fake_wandb(monkeypatch)
    args = make_args(lr=3e-4, n_epochs=12)
    tracker_from_args(args, "run-name", cfg={"seed": 42}, extra_config={"phase": "A"})

    config = holder["run"].init_kwargs["config"]
    assert config["lr"] == 3e-4 and config["n_epochs"] == 12
    assert config["yaml_config"] == {"seed": 42}
    assert config["phase"] == "A"
    assert "torch_version" in config and "cuda_available" in config


def test_the_run_config_excludes_the_wandb_plumbing(monkeypatch):
    holder = install_fake_wandb(monkeypatch)
    tracker_from_args(make_args(wandb_tags="a,b"), "run-name")
    config = holder["run"].init_kwargs["config"]
    assert not [key for key in config if key.startswith("wandb_")]
    assert "no_wandb" not in config


def test_tags_and_group_are_forwarded(monkeypatch):
    holder = install_fake_wandb(monkeypatch)
    tracker_from_args(make_args(wandb_tags=" pilot, 5000ep "), "n", group="grp")
    assert holder["run"].init_kwargs["tags"] == ["pilot", "5000ep"]
    assert holder["run"].init_kwargs["group"] == "grp"


def test_no_wandb_flag_wins_over_everything(monkeypatch):
    install_fake_wandb(monkeypatch)
    tracker = tracker_from_args(make_args(no_wandb=True), "n", cfg={"wandb": {"project": "p"}})
    assert tracker.active is False


@pytest.mark.parametrize("cli,env,yaml_value,expected", [
    (None, None, None, DEFAULT_PROJECT),
    (None, None, "from-yaml", "from-yaml"),
    (None, "from-env", "from-yaml", "from-env"),
    ("from-cli", "from-env", "from-yaml", "from-cli"),
])
def test_project_precedence_is_cli_then_env_then_yaml(
    monkeypatch, cli, env, yaml_value, expected
):
    """CLI > variables d'env > config YAML."""
    if env:
        monkeypatch.setenv("WANDB_PROJECT", env)
    holder = install_fake_wandb(monkeypatch)
    cfg = {"wandb": {"project": yaml_value}} if yaml_value else {}
    tracker_from_args(make_args(wandb_project=cli), "n", cfg=cfg)
    assert holder["run"].init_kwargs["project"] == expected


def test_yaml_can_turn_artifact_logging_on(monkeypatch):
    install_fake_wandb(monkeypatch)
    tracker = tracker_from_args(
        make_args(), "n", cfg={"wandb": {"log_artifacts": True, "mode": "offline"}}
    )
    assert tracker.log_artifacts is True


# --- utilitaires ----------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    (None, []), ("", []), ("a", ["a"]), (" a , b ,, c ", ["a", "b", "c"]),
])
def test_parse_tags(raw, expected):
    assert parse_tags(raw) == expected


def test_environment_config_reports_versions_and_the_commit():
    config = environment_config(".")
    assert "torch_version" in config
    assert config["cuda_available"] in (True, False)
    assert len(config.get("git_commit", "")) == 40
    assert isinstance(config["git_dirty"], bool)


def test_gpu_peak_memory_is_none_on_cpu():
    import torch

    if torch.cuda.is_available():
        pytest.skip("machine avec GPU")
    assert gpu_peak_memory_gb() is None


def test_the_stopwatch_measures_a_rate():
    import time

    watch = Stopwatch()
    time.sleep(0.02)
    assert watch.elapsed >= 0.02
    assert watch.rate(100) > 0
    watch.reset()
    assert watch.elapsed < 0.02
