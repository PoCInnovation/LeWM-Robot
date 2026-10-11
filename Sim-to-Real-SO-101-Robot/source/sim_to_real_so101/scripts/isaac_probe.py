"""One-shot probe of an Isaac Sim machine: self-test + robot calibration dump.

Run this ONCE on the machine that has the GPU. It produces a single JSON that
lets the rest of the project be developed and tested on a machine without any
NVIDIA GPU:

* **self-test** — versions, GPU, ffmpeg encoders, free disk space. Catches
  environment problems before a 10-hour dataset collection, not during.
* **calibration dump** — the articulation's joint/body layout, its real limits,
  and forward-kinematics + jacobian samples over the whole joint space. That
  data drives an offline surrogate of the robot, so the controllers and the
  trajectory generators can be exercised in closed loop under pytest, on CPU.

Usage (inside the Isaac Sim environment, e.g. the ``leisaac`` conda env)::

    python -m sim_to_real_so101.scripts.isaac_probe

Then send back the generated ``outputs/isaac_probe.json`` plus the terminal
output. Nothing else is needed.

This file is only the launcher: everything it reports lives in
``sim_to_real_so101.utils.probe_report``, which imports nothing from isaaclab
and is covered by the offline test suite.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Isaac Sim self-test + SO-101 calibration dump (run once, on the GPU machine)."
)
parser.add_argument(
    "--task",
    type=str,
    default="Lerobot-So101-Teleop-Task",
    help="Task to instantiate. The default carries the cameras and the floor.",
)
parser.add_argument(
    "--samples",
    type=int,
    default=200,
    help="Number of random joint configurations to record FK + jacobian for.",
)
parser.add_argument(
    "--out",
    type=str,
    default="outputs/isaac_probe.json",
    help="Where to write the JSON report.",
)
parser.add_argument(
    "--disk_check_path",
    type=str,
    default="datasets",
    help="Path whose free space is reported (where the dataset will be written).",
)
parser.add_argument("--seed", type=int, default=101, help="Sampling seed.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The probe never needs to display anything, and headless starts much faster.
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils import probe_report


def main():
    report = probe_report.new_report(args_cli.task, args_cli.seed)

    def section(name, fn):
        return probe_report.run_section(report, name, fn)

    section("environment", lambda: probe_report.probe_environment(args_cli.disk_check_path))
    section("registered_tasks", probe_report.probe_registered_tasks)

    env = None
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.seed = args_cli.seed
        section("sim", lambda: probe_report.probe_sim_settings(env_cfg))

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()

        robot = env.scene["robot"]
        section("robot", lambda: probe_report.probe_robot(robot))
        section("scene", lambda: probe_report.probe_scene(env))
        section(
            "fk_samples",
            lambda: probe_report.sample_kinematics(
                env, robot, args_cli.samples, args_cli.seed
            ),
        )
    except Exception as exc:
        report["errors"]["env"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR]: could not build the environment: {type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    out_path = probe_report.write_report(report, args_cli.out)
    probe_report.print_summary(report)
    print(f"[INFO]: report written to {out_path}")
    print(f"[INFO]: size {os.path.getsize(out_path) / 1024:.0f} KB — send this file back.")


if __name__ == "__main__":
    main()
    simulation_app.close()
