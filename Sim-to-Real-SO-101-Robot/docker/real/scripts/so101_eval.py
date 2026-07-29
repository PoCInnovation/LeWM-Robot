# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SO100 Real-Robot Gr00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the SO100 / SO101 robots
using the GR00T Policy API.

Major responsibilities:
    • Initialize robot hardware from a RobotConfig (LeRobot)
    • Convert robot observations into GR00T VLA inputs
    • Query the GR00T policy server (PolicyClient)
    • Decode multi-step (temporal) model actions back into robot motor commands
    • Stream actions to the real robot in real time

This file is meant to be a simple, readable reference
for real-world policy debugging and demos.
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import asdict, dataclass
from datetime import datetime

import logging
from pathlib import Path
from pprint import pformat
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import draccus
from gr00t.policy.server_client import PolicyClient

# Importing various robot configs ensures CLI autocompletion works
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.utils import init_logging, log_say
import numpy as np

from so101_control import SO101Control


def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dim to arrays or scalars.

    GR00T Policy Server expects:
        obs: (batch=1, time=1, ...)
    Calling this function twice achieves that.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  # scalar → [scalar]
    return obs


class So100Adapter:
    """
    Adapter between:
        • Raw robot observation dictionary
        • GR00T VLA input format
        • GR00T action chunk → robot joint commands

    Responsible for:
        • Packaging camera frames as obs["video"]
        • Building obs["state"] for arm + gripper
        • Adding language instruction
        • Adding batch/time dimensions
        • Decoding model action chunks into real robot actions
    """

    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client

        # SO100 joint ordering used for BOTH training + robot execution
        self.robot_state_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        self.camera_keys = ["front", "wrist"]

    # -------------------------------------------------------------------------
    # Observation → Model Input
    # -------------------------------------------------------------------------
    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        """
        Convert raw robot observation dict into the structured GR00T VLA input.
        """
        model_obs = {}

        # (1) Cameras
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}

        # (2) Arm + gripper state
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs["state"] = {
            "single_arm": state[:5],  # (5,)
            "gripper": state[5:6],  # (1,)
        }

        # (3) Language
        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    # -------------------------------------------------------------------------
    # Model Action Chunk → Robot Motor Commands
    # -------------------------------------------------------------------------
    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        """
        chunk["single_arm"]: (B, T, 5)
        chunk["gripper"]:    (B, T, 1)

        Convert to:
            {
                "shoulder_pan.pos": val,
                ...
            }
        for timestep t.
        """
        single_arm = chunk["single_arm"][0][t]  # (5,)
        gripper = chunk["gripper"][0][t]  # (1,)

        full = np.concatenate([single_arm, gripper], axis=0)  # (6,)

        return {joint_name: float(full[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: Dict) -> List[Dict[str, float]]:
        """
        Returns a list of robot motor commands (one per model timestep).
        """
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        # Determine horizon
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) → T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]



def save_eval_plot(
    obs_buffer: List[Dict[str, float]],
    action_buffer: List[Dict[str, float]],
    joint_keys: List[str],
    out_dir: Path = Path("outputs/plots"),
) -> None:
    """Save recorded obs/action buffers as JSON and a per-joint trajectory plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"eval_{ts}"

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for ax, key in zip(axes.flatten(), joint_keys):
        obs_vals = [s[key] for s in obs_buffer]
        act_vals = [s[key] for s in action_buffer]
        ax.plot(obs_vals, label="obs", linewidth=0.8)
        ax.plot(act_vals, label="action", linewidth=0.8, alpha=0.8)
        ax.set_title(key)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.supxlabel("Timestep")
    fig.supylabel("Joint value (deg)")
    fig.suptitle(f"Eval joint trajectory \u2013 {ts}")
    fig.tight_layout()

    png_path = out_dir / f"{base}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logging.info("Saved plot image to %s", png_path)


@dataclass
class EvalConfig:
    """
    Command-line configuration for real-robot policy evaluation.
    """

    robot: RobotConfig | None = None
    policy_host: str = "0.0.0.0"
    policy_port: int = 5555
    action_horizon: int = 16
    lang_instruction: str = "Grab markers and place into pen holder."
    play_sounds: bool = False
    timeout: int = 60
    rerun: bool = False
    passive_mode: bool = False
    plot: bool = False



@draccus.wrap()
def eval(cfg: EvalConfig):
    """
    Main entry point for real-robot policy evaluation.
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))

    so101_control = SO101Control(cfg)
    so101_control.connect()

    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = So100Adapter(policy_client)

    joint_keys = policy.robot_state_keys
    obs_buffer: List[Dict[str, float]] = []
    action_buffer: List[Dict[str, float]] = []

    try:
        if cfg.rerun:
            so101_control.start_logging_thread()

        while True:
            obs = so101_control.get_observation()
            obs["lang"] = cfg.lang_instruction

            actions = policy.get_action(obs)

            for i, action_dict in enumerate(actions[: cfg.action_horizon]):
                tic = time.time()
                so101_control.send_action(action_dict)
                so101_control.update_log_action(action_dict)

                if cfg.plot:
                    step_obs = so101_control.get_observation()
                    obs_buffer.append({k: float(step_obs[k]) for k in joint_keys})
                    action_buffer.append({k: v for k, v in action_dict.items()})

                toc = time.time()
                if toc - tic < 1.0 / 30:
                    time.sleep(1.0 / 30 - (toc - tic))

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down...")
    finally:
        so101_control.stop_logging_thread()

        if cfg.plot and (obs_buffer or action_buffer):
            save_eval_plot(obs_buffer, action_buffer, joint_keys)

        so101_control.disconnect()

if __name__ == "__main__":
    eval()
