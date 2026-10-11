"""Load the calibration dump, when there is one, and sharpen the harness with it.

Until a real machine has run ``scripts/isaac_probe.py``, the surrogate uses
approximate link geometry and the tests that depend on it are skipped rather
than asserted. The moment the dump lands, everything derived from measurement —
joint order, body names, real joint stops, the home pose, the timing settings —
replaces the assumptions, and the geometry checks start running.

Drop the file the binome sends back at either of::

    outputs/isaac_probe.json
    tests/calibration/isaac_probe.json

or point ``SO101_PROBE_JSON`` at it. Nothing else to do.
"""

import json
import os

import torch

from so101_surrogate import RobotSurrogate, SO101Kinematics

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)

SEARCH_PATHS = [
    os.environ.get("SO101_PROBE_JSON"),
    os.path.join(REPO_DIR, "outputs", "isaac_probe.json"),
    os.path.join(TESTS_DIR, "calibration", "isaac_probe.json"),
]


def probe_path():
    """First calibration dump found, or None."""
    for path in SEARCH_PATHS:
        if path and os.path.isfile(path):
            return path
    return None


def load_probe(path=None):
    """Parsed dump, or None when no machine has produced one yet."""
    path = path or probe_path()
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def has_probe():
    return probe_path() is not None


def surrogate_from_probe(probe, kinematics=None, **kwargs):
    """Build a surrogate using the measured layout, limits and home pose.

    Link geometry stays whatever ``kinematics`` provides — fitting the chain to
    the FK samples is a separate step. Everything the dump measures directly is
    taken from it.
    """
    robot_info = probe.get("robot") or {}
    limits = robot_info.get("soft_joint_pos_limits") or robot_info.get("joint_pos_limits")

    surrogate = RobotSurrogate(
        kinematics=kinematics or SO101Kinematics(),
        joint_names=robot_info.get("joint_names"),
        body_names=robot_info.get("body_names"),
        default_joint_pos=robot_info.get("default_joint_pos"),
        **kwargs,
    )
    if limits:
        surrogate._limits = torch.tensor(limits, dtype=surrogate.dtype)
    return surrogate


def fk_error_against_probe(probe, kinematics=None, body="gripper"):
    """How far the surrogate's geometry sits from the measured arm, in metres.

    Returns ``None`` when the dump carries no samples for ``body``. The numbers
    are informational: they say how much the offline geometry can be trusted for
    anything distance-based, which is precisely what the grasp offsets are.
    """
    samples = probe.get("fk_samples") or []
    if not samples:
        return None

    kinematics = kinematics or SO101Kinematics()
    body_names = (probe.get("robot") or {}).get("body_names") or []
    if body not in body_names:
        return None
    body_index = body_names.index(body)
    if body_index >= len(kinematics.chain) + 1:
        return None

    errors = []
    for sample in samples:
        measured = (sample.get("body_pos") or {}).get(body)
        if measured is None:
            continue
        q = torch.tensor(sample["q"], dtype=torch.float64)
        predicted = SO101Kinematics(chain=kinematics.chain, dtype=torch.float64).body_pos(
            q, body_index
        )
        errors.append(float(torch.linalg.norm(predicted - torch.tensor(measured, dtype=torch.float64))))

    if not errors:
        return None
    errors_tensor = torch.tensor(errors)
    return {
        "n_samples": len(errors),
        "mean_m": float(errors_tensor.mean()),
        "median_m": float(errors_tensor.median()),
        "max_m": float(errors_tensor.max()),
    }
