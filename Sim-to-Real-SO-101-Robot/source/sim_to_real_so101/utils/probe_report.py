"""Machine and robot report produced by ``scripts/isaac_probe.py``.

Kept apart from the launcher script so it can be exercised without Isaac Sim:
the script has to build the Omniverse app at import time, which makes it
untestable on a machine without a GPU — and that is exactly how a wrong
assumption about ``find_bodies`` once cost a full round trip.

Everything here takes already-built objects as arguments and imports nothing
from isaaclab, so the offline harness can drive it against a robot surrogate.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time

import torch

SCHEMA_VERSION = 1

# Bodies the controllers may look for. Names absent from a given articulation
# are skipped, never probed — see :func:`body_ids`.
BODY_CANDIDATES = ["gripper", "jaw", "moving_jaw", "Jaw", "wrist", "Wrist", "base"]


def body_ids(robot, candidates):
    """Resolve body indices by name, skipping names this robot does not have.

    ``find_bodies`` treats its argument as a regular expression and **raises**
    when nothing matches; it does not return an empty list. Candidates are
    therefore filtered against the articulation's own body list.
    """
    names = list(robot.data.body_names)
    return {name: names.index(name) for name in candidates if name in names}


def _package_version(name):
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        try:
            module = __import__(name)
            return getattr(module, "__version__", "unknown")
        except Exception:
            return None


def probe_environment(disk_check_path):
    """Versions, GPU, ffmpeg and disk space of the host machine."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in [
                "torch", "isaacsim", "isaaclab", "isaaclab_tasks",
                "lerobot", "numpy", "gymnasium",
            ]
        },
    }

    gpu = {"cuda_available": torch.cuda.is_available()}
    if gpu["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        gpu.update(
            {
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
            }
        )
    info["gpu"] = gpu

    ffmpeg = {"available": shutil.which("ffmpeg") is not None}
    if ffmpeg["available"]:
        try:
            out = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=30
            ).stdout
            ffmpeg["version"] = out.splitlines()[0] if out else None
        except Exception as exc:
            ffmpeg["version_error"] = str(exc)
        try:
            encoders = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            # The dataset is written in AV1; h264 is the fallback if AV1 encoding
            # turns out to dominate the collection time.
            ffmpeg["encoders"] = {
                name: (name in encoders)
                for name in ["libsvtav1", "libaom-av1", "librav1e", "libx264", "libx265"]
            }
        except Exception as exc:
            ffmpeg["encoders_error"] = str(exc)
    info["ffmpeg"] = ffmpeg

    # Walk up until an existing directory is found, so the check works before
    # the dataset folder has been created.
    path = os.path.abspath(disk_check_path)
    probe_path = path
    while probe_path and not os.path.isdir(probe_path):
        parent = os.path.dirname(probe_path)
        if parent == probe_path:
            break
        probe_path = parent
    try:
        usage = shutil.disk_usage(probe_path)
        info["disk"] = {
            "requested_path": path,
            "measured_on": probe_path,
            "total_gb": round(usage.total / 1024**3, 2),
            "free_gb": round(usage.free / 1024**3, 2),
        }
    except Exception as exc:
        info["disk"] = {"error": str(exc)}

    return info


def probe_registered_tasks():
    """Gym ids exposed by this repo, to confirm the package is importable."""
    import gymnasium as gym

    return sorted(spec.id for spec in gym.registry.values() if "Lerobot-" in spec.id)


def probe_sim_settings(env_cfg):
    """Timing settings — this is what the 30 Hz / 60 Hz decision hangs on."""
    dt = float(env_cfg.sim.dt)
    decimation = int(env_cfg.decimation)
    return {
        "sim_dt": dt,
        "decimation": decimation,
        "render_interval": int(env_cfg.sim.render_interval),
        "episode_length_s": float(env_cfg.episode_length_s),
        "physics_hz": round(1.0 / dt, 3),
        "control_hz": round(1.0 / (dt * decimation), 3),
        "num_envs": int(env_cfg.scene.num_envs),
    }


def probe_robot(robot):
    """Articulation layout and limits — the offline surrogate is built on this."""
    info = {
        "joint_names": list(robot.data.joint_names),
        "body_names": list(robot.data.body_names),
        "num_joints": int(robot.num_joints),
        "num_bodies": int(robot.num_bodies),
        "is_fixed_base": bool(robot.is_fixed_base),
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "joint_pos_limits": robot.data.joint_pos_limits[0].tolist(),
    }

    default_root = getattr(robot.data, "default_root_state", None)
    info["default_root_state"] = default_root[0].tolist() if default_root is not None else None

    soft = getattr(robot.data, "soft_joint_pos_limits", None)
    info["soft_joint_pos_limits"] = soft[0].tolist() if soft is not None else None

    for attr in ["joint_vel_limits", "joint_effort_limits"]:
        values = getattr(robot.data, attr, None)
        info[attr] = values[0].tolist() if values is not None else None

    info["body_ids"] = body_ids(robot, BODY_CANDIDATES)

    ee_body_id = robot.find_bodies(["gripper"])[0][0]
    info["ee_body_id"] = int(ee_body_id)
    # The jacobian array skips the root link on a fixed-base articulation.
    info["jacobian_body_id"] = int(ee_body_id - 1 if robot.is_fixed_base else ee_body_id)

    return info


def sample_kinematics(env, robot, n_samples, seed, step_sim=None):
    """Record FK + jacobian over configurations spanning the joint space.

    The commanded configuration is not recorded — the configuration actually
    reached after the step is, so each sample stays self-consistent even if the
    actuators drift slightly during the step.

    ``step_sim`` advances the simulation by one physics step; it is injected so
    the offline harness can drive this without an Omniverse app.
    """
    device = robot.device
    generator = torch.Generator(device="cpu").manual_seed(seed)

    limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if limits is None:
        limits = robot.data.joint_pos_limits
    lo = limits[0, :, 0].to("cpu")
    hi = limits[0, :, 1].to("cpu")
    num_joints = robot.num_joints

    env_origin = env.scene.env_origins[0]
    if step_sim is None:
        dt = float(env.cfg.sim.dt)

        def step_sim():
            env.sim.step(render=False)
            robot.update(dt)

    ids = body_ids(robot, BODY_CANDIDATES)
    jac_body_id = robot.find_bodies(["gripper"])[0][0]
    if robot.is_fixed_base:
        jac_body_id -= 1

    samples = []
    for i in range(n_samples):
        # First sample is the default pose, so the dump always contains the
        # exact configuration the robot spawns in.
        if i == 0:
            q = robot.data.default_joint_pos[0].to("cpu").clone()
        else:
            u = torch.rand(num_joints, generator=generator)
            q = lo + u * (hi - lo)

        q_sim = q.to(device).unsqueeze(0)
        zeros = torch.zeros_like(q_sim)
        robot.write_joint_state_to_sim(q_sim, zeros)
        # Hold the target on the written position so the actuators do not pull
        # the arm away during the step.
        robot.set_joint_position_target(q_sim)
        robot.write_data_to_sim()
        step_sim()

        reached = robot.data.joint_pos[0]
        jacobian = robot.root_physx_view.get_jacobians()[0, jac_body_id]

        samples.append(
            {
                "q": reached.tolist(),
                "body_pos": {
                    name: (robot.data.body_pos_w[0, idx] - env_origin).tolist()
                    for name, idx in ids.items()
                },
                "body_quat": {
                    name: robot.data.body_quat_w[0, idx].tolist()
                    for name, idx in ids.items()
                },
                "jacobian": jacobian.tolist(),
            }
        )

    return samples


def probe_scene(env):
    """Scene entity names and camera geometry, as seen at runtime."""
    info = {"entities": sorted(env.scene.keys())}
    try:
        info["env_origin"] = env.scene.env_origins[0].tolist()
    except Exception as exc:
        info["env_origin_error"] = str(exc)

    cameras = {}
    for name in env.scene.keys():
        if not name.startswith("camera_"):
            continue
        cfg = getattr(env.scene.cfg, name, None)
        if cfg is None:
            continue
        entry = {"height": int(cfg.height), "width": int(cfg.width), "prim_path": cfg.prim_path}
        spawn = getattr(cfg, "spawn", None)
        if spawn is not None:
            for attr in ["focal_length", "f_stop", "focus_distance"]:
                value = getattr(spawn, attr, None)
                if value is not None:
                    entry[attr] = float(value)
        offset = getattr(cfg, "offset", None)
        if offset is not None:
            entry["offset_pos"] = list(offset.pos)
            entry["offset_rot"] = list(offset.rot)
            entry["convention"] = getattr(offset, "convention", None)
        cameras[name] = entry
    info["cameras"] = cameras
    return info


def new_report(task, seed):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task,
        "seed": seed,
        "errors": {},
    }


def run_section(report, name, fn):
    """Run one probe; record the failure instead of losing the whole run."""
    try:
        report[name] = fn()
        return True
    except Exception as exc:
        report["errors"][name] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR]: section '{name}' failed: {type(exc).__name__}: {exc}")
        return False


def write_report(report, out_path):
    out_path = os.path.abspath(out_path)
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    return out_path


def print_summary(report):
    """Human-readable recap, so the terminal output alone is already useful."""
    def line(label, value):
        print(f"  {label:<26} {value}")

    print("\n" + "=" * 72)
    print("ISAAC PROBE — SUMMARY")
    print("=" * 72)

    env_info = report.get("environment", {})
    print("\n[Environment]")
    line("python", env_info.get("python"))
    for name, version in (env_info.get("packages") or {}).items():
        line(name, version or "NOT FOUND")

    gpu = env_info.get("gpu", {})
    print("\n[GPU]")
    if gpu.get("cuda_available"):
        line("device", gpu.get("name"))
        line("VRAM (GB)", gpu.get("total_memory_gb"))
        line("CUDA", gpu.get("cuda_version"))
    else:
        line("cuda_available", "NO  <-- blocking for collection")

    ffmpeg = env_info.get("ffmpeg", {})
    print("\n[ffmpeg]")
    line("available", ffmpeg.get("available"))
    for name, present in (ffmpeg.get("encoders") or {}).items():
        line(name, "yes" if present else "no")

    disk = env_info.get("disk", {})
    print("\n[Disk]")
    if "error" in disk:
        line("error", disk["error"])
    else:
        line("path", disk.get("measured_on"))
        free = disk.get("free_gb")
        line("free (GB)", free)
        # ~8-13 GB expected for 5000 episodes at 30 Hz, plus room to breathe.
        if isinstance(free, (int, float)) and free < 30:
            line("", "WARNING: under 30 GB free, tight for 5000 episodes")

    sim_info = report.get("sim", {})
    print("\n[Timing]")
    line("physics (Hz)", sim_info.get("physics_hz"))
    line("decimation", sim_info.get("decimation"))
    line("control (Hz)", sim_info.get("control_hz"))
    line("episode_length_s", sim_info.get("episode_length_s"))

    robot = report.get("robot", {})
    print("\n[Robot]")
    line("joints", ", ".join(robot.get("joint_names", [])))
    line("bodies", ", ".join(robot.get("body_names", [])))
    line("fixed base", robot.get("is_fixed_base"))
    line("ee_body_id", robot.get("ee_body_id"))
    line("jacobian_body_id", robot.get("jacobian_body_id"))
    for name, limits in zip(robot.get("joint_names", []), robot.get("soft_joint_pos_limits") or []):
        line(f"  limit {name}", f"[{limits[0]:+.4f}, {limits[1]:+.4f}] rad")

    print("\n[Kinematics]")
    line("FK samples", len(report.get("fk_samples", [])))

    scene = report.get("scene", {})
    print("\n[Scene]")
    line("entities", ", ".join(scene.get("entities", [])))
    for name, cam in (scene.get("cameras") or {}).items():
        line(name, f"{cam.get('width')}x{cam.get('height')}")

    errors = report.get("errors", {})
    print("\n[Errors]")
    if errors:
        for section_name, message in errors.items():
            line(section_name, message)
    else:
        line("none", "all sections completed")
    print("=" * 72 + "\n")
