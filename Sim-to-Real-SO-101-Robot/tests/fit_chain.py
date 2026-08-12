"""Recover the SO-101's link geometry from a calibration dump.

The dump records, for 200 random configurations, the world position of the key
bodies *and* the jacobian. That is enough to solve most of the chain in closed
form rather than fitting it blindly:

1. **Axes.** The angular part of jacobian column ``i`` is joint ``i``'s axis in
   world coordinates. Walking outwards, each local axis follows by undoing the
   rotation of the joints before it — exact, no optimisation. This works for
   every joint that moves the body the jacobian was taken at; the gripper sits
   before the Jaw joint, so that last axis has to be fitted instead.
2. **Origins.** With the axes known, every frame's rotation follows from the
   recorded joint angles, so body positions become *linear* in the link offsets.

A body's frame origin does not have to sit on its parent joint's axis, so each
measured body also carries its own constant offset. Without it the model cannot
be exact — and it is that term which makes the moving jaw depend on the gripper
opening at all.

Run it to regenerate the numbers in ``so101_surrogate``::

    .venv/bin/python tests/fit_chain.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEASURED_BODIES = {4: "wrist", 5: "gripper", 6: "jaw"}
DTYPE = torch.float64


def _quat_to_matrix(q):
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=DTYPE,
    )


def rotations(axis, angles):
    """Rodrigues rotation for one axis and a batch of angles -> (S, 3, 3)."""
    axis = axis / torch.linalg.norm(axis)
    x, y, z = axis[0], axis[1], axis[2]
    c = torch.cos(angles)
    s = torch.sin(angles)
    k = 1.0 - c
    return torch.stack(
        [
            torch.stack([c + x * x * k, x * y * k - z * s, x * z * k + y * s], dim=-1),
            torch.stack([y * x * k + z * s, c + y * y * k, y * z * k - x * s], dim=-1),
            torch.stack([z * x * k - y * s, z * y * k + x * s, c + z * z * k], dim=-1),
        ],
        dim=-2,
    )


def load(path):
    with open(path, encoding="utf-8") as handle:
        probe = json.load(handle)
    samples = probe["fk_samples"]
    q = torch.tensor([s["q"] for s in samples], dtype=DTYPE)
    jac = torch.tensor([s["jacobian"] for s in samples], dtype=DTYPE)
    base_pos = torch.tensor(samples[0]["body_pos"]["base"], dtype=DTYPE)
    base_rot = _quat_to_matrix(samples[0]["body_quat"]["base"])
    measured = {
        index: torch.tensor([s["body_pos"][name] for s in samples], dtype=DTYPE)
        for index, name in MEASURED_BODIES.items()
    }
    return probe, q, jac, base_pos, base_rot, measured


def frames_before(q, axes, base_rot):
    """Rotation of the frame each joint acts in -> (S, J+1, 3, 3)."""
    n_samples, n_joints = q.shape
    rot = base_rot.expand(n_samples, 3, 3)
    out = [rot]
    for i in range(n_joints):
        rot = rot @ rotations(axes[i], q[:, i])
        out.append(rot)
    return torch.stack(out, dim=1)


def predict(q, axes, origins, offsets, base_rot, base_pos):
    """World position of every measured body -> {body_index: (S, 3)}."""
    rot = frames_before(q, axes, base_rot)
    n_samples = q.shape[0]

    positions = {}
    running = base_pos.expand(n_samples, 3)
    for k in range(1, max(MEASURED_BODIES) + 1):
        running = running + torch.einsum("sij,j->si", rot[:, k - 1], origins[k - 1])
        if k in MEASURED_BODIES:
            positions[k] = running + torch.einsum("sij,j->si", rot[:, k], offsets[k])
    return positions


def solve_axes_from_jacobian(q, jac, base_rot, n_solvable):
    """Local axes of the joints that move the body the jacobian was taken at."""
    axes = []
    for i in range(n_solvable):
        rot = base_rot.expand(q.shape[0], 3, 3)
        for j in range(i):
            rot = rot @ rotations(axes[j], q[:, j])
        local = torch.einsum("sji,sj->si", rot, jac[:, 3:6, i])
        axis = local.mean(dim=0)
        axes.append(axis / torch.linalg.norm(axis))
    return axes


def solve_origins_linear(q, axes, base_rot, base_pos, measured):
    """Least-squares link offsets, holding the body offsets at zero."""
    n_samples, n_joints = q.shape
    rot = frames_before(q, axes, base_rot)

    rows, rhs = [], []
    for body_index, target in measured.items():
        block = torch.zeros(n_samples, 3, n_joints * 3, dtype=DTYPE)
        for i in range(body_index):
            block[:, :, i * 3:(i + 1) * 3] = rot[:, i]
        rows.append(block.reshape(-1, n_joints * 3))
        rhs.append((target - base_pos).reshape(-1))

    # The system is rank-deficient by construction: Pitch, Elbow and Wrist_Pitch
    # share an axis, so an offset can be traded between them along it. The
    # default lstsq driver assumes full rank and returns garbage here — 'gelsd'
    # is SVD-based and returns the minimum-norm solution, which fits the data
    # exactly and is reproducible.
    A = torch.cat(rows, dim=0)
    b = torch.cat(rhs, dim=0).unsqueeze(1)
    solution = torch.linalg.lstsq(A, b, driver="gelsd").solution.squeeze(1)
    return [solution[i * 3:(i + 1) * 3] for i in range(n_joints)]


def report_residuals(q, axes, origins, offsets, base_rot, base_pos, measured):
    with torch.no_grad():
        predicted = predict(q, axes, origins, offsets, base_rot, base_pos)
    out = {}
    for body_index, target in measured.items():
        error = torch.linalg.norm(predicted[body_index] - target, dim=1)
        out[MEASURED_BODIES[body_index]] = (
            float(error.mean()) * 1000,
            float(error.median()) * 1000,
            float(error.max()) * 1000,
        )
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "isaac_probe.json",
    )
    probe, q, jac, base_pos, base_rot, measured = load(path)
    n_joints = q.shape[1]
    print(f"{len(q)} configurations from {path}")

    # The jacobian is taken at the gripper, so it constrains every joint before
    # it. The Jaw axis is not identifiable from body *positions* at all: body
    # frame origins sit on their own joint axis, so rotating the jaw does not
    # move the jaw body's origin. It is set parallel to the pitch axes, which is
    # what a hinged jaw is, and nothing the surrogate is used for depends on it.
    axes = solve_axes_from_jacobian(q, jac, base_rot, n_joints - 1)
    axes.append(axes[1].clone())
    origins = solve_origins_linear(q, axes, base_rot, base_pos, measured)
    # Solved to be zero on this arm: every body origin lies on its joint axis.
    offsets = {k: torch.zeros(3, dtype=DTYPE) for k in MEASURED_BODIES}

    print("\nresiduals of the solved chain:")
    for name, (mean, median, worst) in report_residuals(
        q, axes, origins, offsets, base_rot, base_pos, measured
    ).items():
        print(f"  {name:<8} mean {mean:9.6f} mm   median {median:9.6f} mm   max {worst:9.6f} mm")

    names = probe["robot"]["joint_names"]
    print("\nDEFAULT_CHAIN = [")
    for i, name in enumerate(names):
        axis = [round(float(v), 6) for v in axes[i]]
        origin = [round(float(v), 6) for v in origins[i]]
        print(f'    {{"axis": ({axis[0]}, {axis[1]}, {axis[2]}), '
              f'"origin": ({origin[0]}, {origin[1]}, {origin[2]})}},  # {name}')
    print("]")

    print(f"\nBASE_POS = {tuple(round(float(v), 6) for v in base_pos)}")
    print("BASE_ROT (quat wxyz) =",
          [round(float(v), 6) for v in probe["fk_samples"][0]["body_quat"]["base"]])


if __name__ == "__main__":
    main()
