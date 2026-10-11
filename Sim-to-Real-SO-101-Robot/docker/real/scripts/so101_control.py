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
import draccus
import logging
import threading
import time
import os
import numpy as np
import subprocess
import uuid
from typing import Any, Dict
from pprint import pformat
from dataclasses import asdict, dataclass
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.utils.utils import init_logging
import rerun as rr


def init_rerun(session_name: str | None = None, window_size: str = "1280x720") -> None:
    """Initialize and spawn the Rerun viewer for live rollout visualization."""
    if session_name is None:
        session_name = f"so100_eval_{uuid.uuid4().hex[:8]}"
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("RERUN_MEMORY_LIMIT", "10%")
    port = 9876
    subprocess.Popen([
        "rerun",
        "--port", str(port),
        "--memory-limit", memory_limit,
        "--window-size", window_size,
        "--expect-data-soon", "true",
    ])
    rr.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")


def kill_rerun():
    subprocess.run(["pkill", "-f", "rerun"], capture_output=True)


@dataclass
class SetupConfig:
    robot: RobotConfig | None = None
    play_sounds: bool = False
    passive_mode: bool = False
    rerun: bool = False


class SO101Control:

    initial_pose = {
        "shoulder_pan.pos": -6.3602,
        "shoulder_lift.pos": -51.9492,
        "elbow_flex.pos": 16.7276,
        "wrist_flex.pos": 89.2483,
        "wrist_roll.pos": -51.8499,
        "gripper.pos": 0.0000,
    }

    home_pose = {
        "shoulder_pan.pos": -6.2835,
        "shoulder_lift.pos": -91.4407,
        "elbow_flex.pos": 93.1444,
        "wrist_flex.pos": 69.1434,
        "wrist_roll.pos": -51.9027,
        "gripper.pos": 0.0707,
    }

    def __init__(self, cfg: SetupConfig):
        self.robot = make_robot_from_config(cfg.robot)
        self.play_sounds = cfg.play_sounds
        self.passive_mode = cfg.passive_mode
        self.rerun = cfg.rerun
        self._hw_lock = threading.Lock()

    def connect(self):
        logging.info("Initializing robot")
        self.robot.connect()
        if self.rerun:
            init_rerun()
        if self.passive_mode:
            self.robot.bus.disable_torque()
            logging.info("Robot in passive mode. Move the arm to desired poses.")
        else:
            self.move_to_initial_pose()
        

    def disconnect(self):
        if self.passive_mode:
            logging.info("Final pose:")
            obs = self.robot.get_observation()
            joint_obs = {k: v for k, v in obs.items() if k.endswith(".pos")}
            for k, v in joint_obs.items():
                logging.info(f"  {k}: {v:.4f}")
        else:
            self.move_to_home_pose()

        self.robot.disconnect()

        if self.rerun:
            kill_rerun()

        logging.info("Robot disconnected")

    def move_to_pose(self, target_pose: dict, duration: float = 3.0, fps: float = 30.0, rerun: bool = False):
        """Smoothly move the robot to a target pose over the given duration."""
        current_obs = self.robot.get_observation()
        current_pose = {k: current_obs[k] for k in target_pose}

        keys = list(target_pose.keys())
        start = np.array([current_pose[k] for k in keys])
        end = np.array([target_pose[k] for k in keys])

        num_steps = int(duration * fps)
        for i in range(1, num_steps + 1):
            t = i / num_steps
            # Smooth ease-in-ease-out via cosine interpolation
            alpha = (1 - np.cos(t * np.pi)) / 2
            interp = start + alpha * (end - start)
            action = {k: float(interp[j]) for j, k in enumerate(keys)}
            self.robot.send_action(action)
            time.sleep(1.0 / fps)
            if rerun:
                self.log_rerun_data(observation=self.robot.get_observation(), action=action)

    def move_to_initial_pose(self):
        self.move_to_pose(self.initial_pose, rerun=self.rerun)
        logging.info("Moved to initial pose")

    def move_to_home_pose(self):
        self.move_to_pose(self.home_pose, rerun=self.rerun)
        logging.info("Moved to home pose")


    # -----------------------------------------------------------------
    # Thread-safe hardware access
    # -----------------------------------------------------------------
    def get_observation(self) -> Dict[str, Any]:
        with self._hw_lock:
            return self.robot.get_observation()

    def send_action(self, action: Dict[str, Any]) -> None:
        with self._hw_lock:
            self.robot.send_action(action)

    # -----------------------------------------------------------------
    # Threaded Rerun logging
    # -----------------------------------------------------------------
    def start_logging_thread(self, log_fps: float = 30.0) -> None:
        self._log_stop = threading.Event()
        self._log_action: Dict[str, Any] = {}

        def worker():
            while not self._log_stop.wait(1.0 / log_fps):
                try:
                    self.log_rerun_data(self.get_observation(), self._log_action)
                except Exception as e:
                    logging.warning("Rerun log error: %s", e)

        self._log_thread = threading.Thread(target=worker, daemon=True)
        self._log_thread.start()

    def update_log_action(self, action: Dict[str, Any]) -> None:
        self._log_action = action

    def stop_logging_thread(self) -> None:
        if hasattr(self, "_log_stop"):
            self._log_stop.set()
            self._log_thread.join(timeout=2.0)

    def log_rerun_data(
        self,
        observation: Dict[str, Any] | None = None,
        action: Dict[str, Any] | None = None,
    ) -> None:
        """Log observation images/scalars and action scalars to the Rerun viewer."""
        if observation:
            for k, v in observation.items():
                if v is None:
                    continue
                key = f"observation.{k}"
                if isinstance(v, (float, int, np.integer, np.floating)):
                    rr.log(key, rr.Scalars(float(v)))
                elif isinstance(v, np.ndarray):
                    if (
                        v.ndim == 3
                        and v.shape[0] in (1, 3, 4)
                        and v.shape[-1] not in (1, 3, 4)
                    ):
                        v = np.transpose(v, (1, 2, 0))
                    if v.ndim >= 2:
                        rr.log(key, rr.Image(v), static=True)
                    else:
                        for i, vi in enumerate(v):
                            rr.log(f"{key}_{i}", rr.Scalars(float(vi)))

        if action:
            for k, v in action.items():
                if v is None:
                    continue
                key = f"action.{k}"
                if isinstance(v, (float, int, np.integer, np.floating)):
                    rr.log(key, rr.Scalars(float(v)))
                elif isinstance(v, np.ndarray):
                    for i, vi in enumerate(v.flatten()):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


@draccus.wrap()
def main(cfg: SetupConfig):
    init_logging()

    logging.info(pformat(asdict(cfg)))

    so101_control = SO101Control(cfg)
    so101_control.connect()

    try:
        while True:
            obs = so101_control.robot.get_observation()
            so101_control.log_rerun_data(observation=obs, action={})
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down...")
    finally:
        so101_control.disconnect()


if __name__ == "__main__":
    main()
