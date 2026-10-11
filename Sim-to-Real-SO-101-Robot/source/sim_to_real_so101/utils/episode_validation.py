"""Tell a *failed* episode apart from a *degenerate* one.

Failures are the point of several collection modes: an object that slips out of
the gripper, a stack that collapses, a grasp that misses. All of that is contact
dynamics the world model needs, and none of it is discarded.

What must be discarded is an episode with no usable dynamics at all — a state
machine that crashed after three frames, a diverging IK, an arm that never
moved. On a 5000-episode run those slip in silently and are near-impossible to
spot afterwards, so they are caught at record time.

The checker is deliberately reluctant to reject. Anything that merely *looks*
odd is reported as a warning and carried into the side-car metadata, where the
end-of-collection audit can weigh it. Only unusable episodes are rejected.

numpy only — no torch, no isaaclab — so it runs at record time inside Isaac and
again afterwards when auditing the finished dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Rejection reasons, as stable identifiers for the side-car and the audit.
TOO_SHORT = "too_short"
NOT_FINITE = "not_finite"
BAD_SHAPE = "bad_shape"
MOTIONLESS = "motionless"
DIVERGED = "diverged"
JOINT_PINNED = "joint_pinned"

# Warnings — recorded, never a reason to drop the episode.
WARN_DEAD_DIMENSION = "dead_dimension"
WARN_SATURATED = "saturated_joint"
WARN_MAYBE_TRUNCATED = "maybe_truncated"
WARN_VERY_SHORT = "very_short"


@dataclass
class EpisodeCheckConfig:
    """Thresholds for :func:`check_episode`.

    Defaults are tuned to reject only the unusable. ``max_joint_step`` is
    generous on purpose: at 30 Hz a fast SO-101 motion covers well under
    0.5 rad per control step, so anything above that is the IK blowing up
    rather than a brisk move.
    """

    min_frames: int = 30
    short_frames: int = 60          # below this, worth a look but still kept
    max_joint_step: float = 0.5     # rad between two consecutive control frames
    # Dimensions commanded as set-points rather than as a tracked trajectory.
    # The scripted policy flips the jaw straight from JAW_OPEN to JAW_CLOSED,
    # a measured 0.77 rad step in a single frame — perfectly legitimate, and it
    # must not be mistaken for a diverging solver. Arm joints, by contrast, move
    # at most DQ_MAX (0.05 rad) per frame.
    setpoint_dims: tuple = (5,)
    min_motion_std: float = 1e-3    # rad; below this the arm simply never moved
    dead_dimension_std: float = 1e-3
    pinned_fraction: float = 0.99   # a joint welded to a stop for the whole episode
    saturated_fraction: float = 0.5  # worth reporting well before it is fatal
    limit_tolerance: float = 1e-3
    buffer_capacity: int | None = None  # recorder capacity, to spot truncation


@dataclass
class EpisodeReport:
    """Verdict plus the statistics the audit needs later."""

    valid: bool = True
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def reject(self, reason, detail):
        self.valid = False
        self.reasons.append({"code": reason, "detail": detail})

    def warn(self, code, detail):
        self.warnings.append({"code": code, "detail": detail})

    def summary(self):
        if self.valid:
            head = f"valid ({self.stats.get('n_frames', 0)} frames)"
        else:
            head = "REJECTED: " + "; ".join(item["detail"] for item in self.reasons)
        if self.warnings:
            head += " | warnings: " + "; ".join(item["detail"] for item in self.warnings)
        return head


def _as_2d(array, name, report):
    """Coerce to ``(n_frames, n_joints)`` float array, or reject."""
    try:
        values = np.asarray(array, dtype=np.float64)
    except Exception as exc:
        report.reject(BAD_SHAPE, f"{name} is not numeric: {exc}")
        return None
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        report.reject(BAD_SHAPE, f"{name} has shape {values.shape}, expected (frames, joints)")
        return None
    return values


def check_episode(actions, states=None, joint_limits=None, config=None, fps=30):
    """Judge one episode.

    Args:
        actions: ``(n_frames, n_joints)`` commanded joint positions.
        states: optional ``(n_frames, n_joints)`` measured joint positions.
        joint_limits: optional ``(n_joints, 2)`` lower/upper stops, needed for
            the saturation checks. In the same units as ``actions``.
        config: :class:`EpisodeCheckConfig`, or None for the defaults.
        fps: used only to report a duration.

    Returns:
        :class:`EpisodeReport`.
    """
    config = config or EpisodeCheckConfig()
    report = EpisodeReport()

    actions = _as_2d(actions, "actions", report)
    if actions is None:
        return report

    n_frames, n_joints = actions.shape
    report.stats["n_frames"] = int(n_frames)
    report.stats["n_joints"] = int(n_joints)
    report.stats["duration_s"] = round(n_frames / float(fps), 3) if fps else None

    if states is not None:
        states = _as_2d(states, "states", report)
        if states is None:
            return report
        if states.shape != actions.shape:
            report.reject(
                BAD_SHAPE,
                f"states {states.shape} does not match actions {actions.shape}",
            )
            return report

    # -- Non-finite values -------------------------------------------------
    # Nothing downstream survives a NaN, and it propagates through statistics.
    for name, values in (("actions", actions), ("states", states)):
        if values is None:
            continue
        if not np.isfinite(values).all():
            count = int((~np.isfinite(values)).sum())
            report.reject(NOT_FINITE, f"{count} non-finite value(s) in {name}")
            return report

    # -- Length ------------------------------------------------------------
    if n_frames < config.min_frames:
        report.reject(
            TOO_SHORT, f"{n_frames} frames, under the {config.min_frames} minimum"
        )
    elif n_frames < config.short_frames:
        report.warn(WARN_VERY_SHORT, f"only {n_frames} frames")

    if config.buffer_capacity and n_frames >= config.buffer_capacity:
        # The recorder drops frames past its capacity with only a print.
        report.warn(
            WARN_MAYBE_TRUNCATED,
            f"{n_frames} frames reaches the recorder capacity ({config.buffer_capacity}) — "
            "the episode may be truncated",
        )

    # -- Per-dimension statistics -----------------------------------------
    stds = actions.std(axis=0)
    report.stats["action_std"] = [round(float(v), 6) for v in stds]
    report.stats["action_min"] = [round(float(v), 6) for v in actions.min(axis=0)]
    report.stats["action_max"] = [round(float(v), 6) for v in actions.max(axis=0)]

    if float(stds.max()) < config.min_motion_std:
        report.reject(
            MOTIONLESS,
            f"no dimension moves (max std {float(stds.max()):.2e})",
        )

    dead = [int(i) for i, value in enumerate(stds) if value < config.dead_dimension_std]
    report.stats["dead_dimensions"] = dead
    if dead and len(dead) < n_joints:
        # Exactly how defect #2 shows up: wrist_roll flat across every episode.
        report.warn(
            WARN_DEAD_DIMENSION,
            f"dimension(s) {dead} are constant over the episode",
        )

    # -- Divergence --------------------------------------------------------
    # Measured joint positions are the honest signal: physics cannot teleport a
    # joint, so a jump there really is the solver blowing up. Commands can jump
    # legitimately (a set-point change), so when only commands are available the
    # set-point dimensions are excluded rather than the threshold relaxed.
    if n_frames >= 2:
        report.stats["max_command_step"] = round(float(np.abs(np.diff(actions, axis=0)).max()), 6)

        if states is not None:
            checked, source, columns = states, "states", list(range(n_joints))
        else:
            columns = [i for i in range(n_joints) if i not in set(config.setpoint_dims)]
            checked, source = actions, "actions"
        report.stats["divergence_checked_on"] = source

        if columns:
            steps = np.abs(np.diff(checked[:, columns], axis=0))
            largest = float(steps.max())
            report.stats["max_joint_step"] = round(largest, 6)
            if largest > config.max_joint_step:
                joint = columns[int(np.unravel_index(steps.argmax(), steps.shape)[1])]
                report.reject(
                    DIVERGED,
                    f"joint {joint} jumps {largest:.3f} rad in one frame "
                    f"(limit {config.max_joint_step}, checked on {source})",
                )
        else:
            report.stats["max_joint_step"] = 0.0
    else:
        report.stats["max_joint_step"] = 0.0
        report.stats["max_command_step"] = 0.0

    # -- Joints stuck against their stops ---------------------------------
    if joint_limits is not None:
        limits = np.asarray(joint_limits, dtype=np.float64)
        if limits.shape == (n_joints, 2):
            at_low = actions <= limits[:, 0] + config.limit_tolerance
            at_high = actions >= limits[:, 1] - config.limit_tolerance
            fractions = (at_low | at_high).mean(axis=0)
            report.stats["saturation_fraction"] = [round(float(v), 4) for v in fractions]

            pinned = [int(i) for i, value in enumerate(fractions) if value >= config.pinned_fraction]
            if pinned:
                report.reject(
                    JOINT_PINNED,
                    f"joint(s) {pinned} sit against a stop for the whole episode",
                )

            saturated = [
                int(i)
                for i, value in enumerate(fractions)
                if config.saturated_fraction <= value < config.pinned_fraction
            ]
            if saturated:
                # Defect #3 territory: commands the arm cannot physically follow.
                report.warn(
                    WARN_SATURATED,
                    f"joint(s) {saturated} spend most of the episode against a stop",
                )
        else:
            report.warn(
                WARN_SATURATED,
                f"joint_limits has shape {limits.shape}, expected ({n_joints}, 2) — "
                "saturation not checked",
            )

    return report


def summarize_episodes(reports):
    """Aggregate many :class:`EpisodeReport` into collection-level figures."""
    total = len(reports)
    if total == 0:
        return {"total": 0}

    rejected = [report for report in reports if not report.valid]
    reason_counts = {}
    for report in rejected:
        for item in report.reasons:
            reason_counts[item["code"]] = reason_counts.get(item["code"], 0) + 1

    warning_counts = {}
    for report in reports:
        for item in report.warnings:
            warning_counts[item["code"]] = warning_counts.get(item["code"], 0) + 1

    kept = [report for report in reports if report.valid]
    frames = [report.stats.get("n_frames", 0) for report in kept]

    summary = {
        "total": total,
        "kept": len(kept),
        "rejected": len(rejected),
        "rejection_rate": round(len(rejected) / total, 4),
        "reasons": reason_counts,
        "warnings": warning_counts,
        "frames_total": int(sum(frames)),
        "frames_mean": round(float(np.mean(frames)), 1) if frames else 0.0,
        "frames_min": int(min(frames)) if frames else 0,
        "frames_max": int(max(frames)) if frames else 0,
    }

    # Per-dimension spread across the whole collection. A dimension that is flat
    # here is flat in the dataset — the check that would have caught defect #2
    # before 200 episodes were recorded.
    per_dim = [report.stats.get("action_std") for report in kept if report.stats.get("action_std")]
    if per_dim:
        stacked = np.asarray(per_dim, dtype=np.float64)
        summary["action_std_mean"] = [round(float(v), 6) for v in stacked.mean(axis=0)]
        summary["dimensions_always_dead"] = [
            int(i) for i in range(stacked.shape[1]) if float(stacked[:, i].max()) < 1e-3
        ]
    return summary
