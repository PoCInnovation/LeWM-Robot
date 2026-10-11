#!/usr/bin/env python
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
Manual joint control script for SO Follower robot.

Usage:
    python -m lerobot.robots.so_follower.manual_joint_control \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=follower \\
        --step-size=1.0

Controls:
- LEFT/RIGHT arrow keys: Decrease/increase current joint position
- UP/DOWN arrow keys: Switch between joints
- 'z' key: Send all joints to 0 position
- 'q' or ESC: Quit
"""

import argparse
import sys
import threading
import time
import numpy as np
from pathlib import Path

try:
    from pynput import keyboard
    from pynput.keyboard import Key
except ImportError:
    print("Error: pynput library not found. Install with: pip install pynput")
    sys.exit(1)

from lerobot.robots.so101_follower import SO101Follower as SOFollower
from lerobot.robots.so101_follower import SO101FollowerConfig as SOFollowerRobotConfig


class ManualJointController:
    """Controller for manually adjusting robot joints via keyboard."""

    home_pose = {
        "shoulder_pan": -6.2835,
        "shoulder_lift": -91.4407,
        "elbow_flex": 93.1444,
        "wrist_flex": 69.1434,
        "wrist_roll": -51.9027,
        "gripper": 0.0707,
    }

    def __init__(self, robot: SOFollower, step_size: float = 1.0):
        self.robot = robot
        self.step_size = step_size
        self.joint_names = list(robot.bus.motors.keys())
        self.current_joint_idx = 0
        self.target_positions = {}
        self.running = True
        self._moving = False
        
        # Initialize target positions from current state
        current_state = self.robot.bus.sync_read("Present_Position")
        self.target_positions = {name: float(pos) for name, pos in current_state.items()}
        
        print("\n" + "="*60)
        print("Manual Joint Control for SO Follower")
        print("="*60)
        print("\nControls:")
        print("  LEFT/RIGHT arrows: Decrease/increase current joint")
        print("  UP/DOWN arrows:    Switch between joints")
        print("  'h' key:           Move to home position (smooth)")
        print("  'z' key:           Send all joints to 0")
        print("  'q' or ESC:        Quit")
        print("="*60 + "\n")
        
        self._display_status()

    def _display_status(self):
        """Display current joint positions and selection."""
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        print("="*60)
        print("Manual Joint Control for SO Follower")
        print("="*60)
        print(f"\nStep size: {self.step_size}")
        print("\nJoint Positions:")
        print("-"*60)
        
        for idx, joint_name in enumerate(self.joint_names):
            target = self.target_positions[joint_name]
            indicator = ">>>" if idx == self.current_joint_idx else "   "
            print(f"{indicator} {joint_name:20s}: {target:8.2f}")
        
        print("-"*60)
        print(f"\nCurrently selected: {self.joint_names[self.current_joint_idx]}")
        status = " [MOVING TO HOME...]" if self._moving else ""
        print(f"\nControls: LEFT/RIGHT=adjust, UP/DOWN=switch joint, H=home, Z=zero all, Q/ESC=quit{status}")

    def _send_action(self):
        """Send current target positions to robot."""
        action = {f"{name}.pos": pos for name, pos in self.target_positions.items()}
        self.robot.send_action(action)

    def adjust_joint(self, delta: float):
        """Adjust current joint by delta."""
        joint_name = self.joint_names[self.current_joint_idx]
        self.target_positions[joint_name] += delta
        
        # Clamp to reasonable ranges based on motor type
        if joint_name == "gripper":
            self.target_positions[joint_name] = max(0, min(100, self.target_positions[joint_name]))
        else:
            # For body joints, clamp based on normalization mode
            if self.robot.config.use_degrees:
                self.target_positions[joint_name] = max(-180, min(180, self.target_positions[joint_name]))
            else:
                self.target_positions[joint_name] = max(-100, min(100, self.target_positions[joint_name]))
        
        self._send_action()
        self._display_status()

    def switch_joint(self, direction: int):
        """Switch to next/previous joint."""
        self.current_joint_idx = (self.current_joint_idx + direction) % len(self.joint_names)
        self._display_status()

    def move_to_home(self, duration: float = 3.0, fps: float = 30.0):
        """Smoothly move the robot to the home pose in a background thread."""
        if self._moving:
            return

        def _run():
            self._moving = True
            self._display_status()
            keys = [k for k in self.home_pose if k in self.target_positions]
            start = np.array([self.target_positions[k] for k in keys])
            end = np.array([self.home_pose[k] for k in keys])
            num_steps = int(duration * fps)
            for i in range(1, num_steps + 1):
                if not self.running:
                    break
                alpha = (1 - np.cos(i / num_steps * np.pi)) / 2
                interp = start + alpha * (end - start)
                for j, k in enumerate(keys):
                    self.target_positions[k] = float(interp[j])
                self._send_action()
                time.sleep(1.0 / fps)
            self._moving = False
            self._display_status()

        threading.Thread(target=_run, daemon=True).start()

    def zero_all_joints(self):
        """Send all joints to 0 position."""
        print("\nSending all joints to 0...")
        for name in self.joint_names:
            self.target_positions[name] = 0.0
        self._send_action()
        self._display_status()

    def on_press(self, key):
        """Handle keyboard press events."""
        try:
            if key == Key.left:
                self.adjust_joint(-self.step_size)
            elif key == Key.right:
                self.adjust_joint(self.step_size)
            elif key == Key.up:
                self.switch_joint(-1)
            elif key == Key.down:
                self.switch_joint(1)
            elif key == Key.esc:
                print("\nExiting...")
                self.running = False
                return False
            elif hasattr(key, 'char'):
                if key.char == 'q':
                    print("\nExiting...")
                    self.running = False
                    return False
                elif key.char == 'h':
                    self.move_to_home()
                elif key.char == 'z':
                    self.zero_all_joints()
        except AttributeError:
            pass

    def run(self):
        """Start the manual control loop."""
        # Start keyboard listener
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        
        try:
            # Keep the main thread alive and periodically update display
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            listener.stop()
            print("Control loop stopped.")


def main():
    parser = argparse.ArgumentParser(description="Manual joint control for SO Follower robot")
    parser.add_argument(
        "--robot.type",
        type=str,
        required=True,
        help="Robot type (e.g., so100_follower, so101_follower)",
    )
    parser.add_argument(
        "--robot.port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for robot communication (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--robot.id",
        type=str,
        required=True,
        help="Robot ID for calibration lookup",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1.0,
        help="Step size for joint adjustments (default: 1.0)",
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help="Use degrees for joint positions instead of normalized range",
    )
    
    args = parser.parse_args()
    
    # Parse arguments with dot notation
    robot_type = getattr(args, 'robot.type')
    robot_port = getattr(args, 'robot.port')
    robot_id = getattr(args, 'robot.id')
    
    # Create robot config
    config = SOFollowerRobotConfig(
        id=robot_id,
        port=robot_port,
        use_degrees=args.use_degrees,
    )
    
    # Initialize robot
    print("Initializing robot...")
    robot = SOFollower(config)
    
    try:
        robot.connect()
        print("Robot connected successfully!")
        
        # Start manual control
        controller = ManualJointController(robot, step_size=args.step_size)
        controller.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Disconnecting robot...")
        if robot.is_connected:
            robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()