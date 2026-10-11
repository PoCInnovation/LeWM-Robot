"""Suivi d'expériences Weights & Biases, optionnel.

Si wandb est absent, mal configuré ou injoignable, tout dégrade en no-op et
l'entraînement continue.

Usage:
    from src.tracking import add_wandb_args, tracker_from_args

    add_wandb_args(parser)
    args = parser.parse_args()
    run = tracker_from_args(args, "mon-run", cfg)
    run.log({"train/loss": 0.5}, step=step)
    run.log_summary({"best_val_loss": best})
    run.finish()
"""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch

DEFAULT_PROJECT = "lewm-robot-wm"


def git_revision(root: str | Path = ".") -> dict[str, Any]:
    """Commit courant et propreté de l'arbre."""
    info: dict[str, Any] = {}
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root),
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(root),
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip())
    except Exception:
        pass
    return info


def environment_config(root: str | Path = ".") -> dict[str, Any]:
    """Versions, GPU et commit git, pour la config du run."""
    config: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        config["gpu_name"] = properties.name
        config["gpu_vram_gb"] = round(properties.total_memory / 1e9, 1)
        config["cuda_version"] = torch.version.cuda
    config.update(git_revision(root))
    return config


def gpu_peak_memory_gb() -> float | None:
    """Pic de VRAM allouée depuis le dernier reset, None en CPU."""
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 3)


def reset_gpu_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class Stopwatch:
    """Chronomètre wall-clock."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def rate(self, n_items: int) -> float:
        """Débit en items/s depuis le dernier reset."""
        seconds = self.elapsed
        return n_items / seconds if seconds > 0 else 0.0


class RunTracker:
    """Un run W&B, ou un no-op si le suivi est indisponible.

    :attr:`active` dit lequel des deux. Toutes les méthodes sont sûres à
    appeler dans les deux cas.
    """

    def __init__(
        self,
        project: str | None = None,
        name: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        mode: str | None = None,
        enabled: bool = True,
        log_artifacts: bool = False,
        run_dir: str | Path | None = None,
    ):
        self._run = None
        self._wandb = None
        self.log_artifacts = log_artifacts
        self.disabled_reason: str | None = None

        if not enabled:
            self.disabled_reason = "désactivé (--no-wandb)"
            self._announce()
            return

        try:
            import wandb
        except ImportError:
            self.disabled_reason = "wandb non installé (pip install wandb)"
            self._announce()
            return

        # Sans credentials, rester en local : une init interactive bloquerait
        # un run non surveillé.
        resolved_mode = mode or os.environ.get("WANDB_MODE")
        if resolved_mode is None and not (
            os.environ.get("WANDB_API_KEY") or _has_netrc_entry()
        ):
            resolved_mode = "offline"
            print("[W&B] Pas de credentials trouvés — passage en mode offline. "
                  "Synchronisation ultérieure : wandb sync <dossier>")

        if run_dir is not None:
            run_dir = Path(run_dir)
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                run_dir = None

        try:
            self._run = wandb.init(
                project=project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT,
                dir=str(run_dir) if run_dir is not None else None,
                name=name,
                config=dict(config or {}),
                tags=tags or None,
                group=group,
                mode=resolved_mode,
                reinit=True,
            )
            self._wandb = wandb
        except Exception as exc:
            self.disabled_reason = f"init échouée ({type(exc).__name__}: {exc})"
            self._run = None
            self._announce()
            return

        atexit.register(self.finish)
        print(f"[W&B] Run actif : {self.url or self._run.name}")

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None) if self._run else None

    def _announce(self) -> None:
        print(f"[W&B] Suivi inactif : {self.disabled_reason}. "
              "L'entraînement continue normalement.")

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Enregistre des métriques. Les valeurs None sont omises."""
        if not self.active:
            return
        payload = {key: value for key, value in metrics.items() if value is not None}
        if not payload:
            return
        try:
            self._run.log(payload, step=step)
        except Exception as exc:
            print(f"[W&B] log ignoré ({type(exc).__name__}: {exc})")

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        """Valeurs finales du run."""
        if not self.active:
            return
        try:
            for key, value in metrics.items():
                if value is not None:
                    self._run.summary[key] = value
        except Exception as exc:
            print(f"[W&B] summary ignoré ({type(exc).__name__}: {exc})")

    def log_artifact(
        self,
        path: str | Path,
        name: str | None = None,
        artifact_type: str = "model",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Téléverse un checkpoint. Sans effet si log_artifacts est False."""
        if not self.active or not self.log_artifacts:
            return
        path = Path(path)
        if not path.exists():
            return
        try:
            artifact = self._wandb.Artifact(
                name=name or path.stem,
                type=artifact_type,
                metadata=dict(metadata or {}),
            )
            artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        except Exception as exc:
            print(f"[W&B] artifact ignoré ({type(exc).__name__}: {exc})")

    def finish(self) -> None:
        """Clôt le run. Idempotent."""
        if not self.active:
            return
        try:
            self._run.finish()
        except Exception:
            pass
        finally:
            self._run = None

    def __enter__(self) -> "RunTracker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.finish()
        return False


def _has_netrc_entry() -> bool:
    """True si un ~/.netrc mentionne W&B (ce qu'écrit `wandb login`)."""
    for candidate in (Path.home() / ".netrc", Path.home() / "_netrc"):
        try:
            if candidate.exists() and "wandb" in candidate.read_text(errors="ignore"):
                return True
        except Exception:
            pass
    return False


def add_wandb_args(parser) -> None:
    """Ajoute les options W&B communes aux scripts d'entraînement."""
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument("--no-wandb", action="store_true",
                       help="Désactive complètement le suivi W&B.")
    group.add_argument("--wandb-project", default=None,
                       help=f"Projet W&B (défaut : {DEFAULT_PROJECT}).")
    group.add_argument("--wandb-name", default=None,
                       help="Nom du run (défaut : généré par W&B).")
    group.add_argument("--wandb-tags", default=None,
                       help="Tags séparés par des virgules.")
    group.add_argument("--wandb-group", default=None,
                       help="Groupe, pour comparer plusieurs runs liés.")
    group.add_argument("--wandb-mode", default=None,
                       choices=["online", "offline", "disabled"],
                       help="Forcer le mode (défaut : online si credentials).")
    group.add_argument("--wandb-log-artifacts", action="store_true",
                       help="Téléverse les checkpoints comme artifacts W&B.")


def parse_tags(raw: str | None) -> list[str]:
    """Découpe une liste de tags séparés par des virgules."""
    if not raw:
        return []
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def tracker_from_args(
    args,
    default_name: str,
    cfg: Mapping[str, Any] | None = None,
    extra_config: Mapping[str, Any] | None = None,
    root: str | Path = ".",
    group: str | None = None,
) -> RunTracker:
    """Construit un RunTracker depuis les arguments CLI.

    La config du run réunit les arguments CLI, la config YAML, l'environnement
    et le commit git.

    Priorité du projet et du mode : CLI > variables d'env > config YAML.
    """
    config: dict[str, Any] = {
        key: value for key, value in vars(args).items()
        if not key.startswith("wandb_") and key != "no_wandb"
    }
    if cfg:
        config["yaml_config"] = dict(cfg)
    config.update(environment_config(root))
    if extra_config:
        config.update(extra_config)

    yaml_wandb = dict((cfg or {}).get("wandb") or {})
    project = getattr(args, "wandb_project", None)
    if project is None and not os.environ.get("WANDB_PROJECT"):
        project = yaml_wandb.get("project")
    mode = getattr(args, "wandb_mode", None)
    if mode is None and not os.environ.get("WANDB_MODE"):
        mode = yaml_wandb.get("mode")

    return RunTracker(
        project=project,
        name=getattr(args, "wandb_name", None) or default_name,
        config=config,
        tags=parse_tags(getattr(args, "wandb_tags", None)),
        group=getattr(args, "wandb_group", None) or group,
        mode=mode,
        enabled=not getattr(args, "no_wandb", False),
        log_artifacts=(getattr(args, "wandb_log_artifacts", False)
                       or bool(yaml_wandb.get("log_artifacts", False))),
        run_dir=Path(root) / "results",
    )
