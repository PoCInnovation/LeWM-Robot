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
import os
import threading
import queue
import subprocess

import torch
import numpy as np
from tqdm import tqdm

import omni.kit.app
from carb.eventdispatcher import get_eventdispatcher, Event


from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotRecorder:

    STOP_RECORDING_EVENT: str = "lerobot_so101_teleop.stop_recording"
    CANCEL_RECORDING_EVENT: str = "lerobot_so101_teleop.cancel_recording"

    def __init__(
        self,
        task_name: str,
        repo_id: str,
        dataset_root: str,
        fps: int,
        device: str,
        cameras: dict,
        save_mp4: bool = False,
        depth: bool = False, # not saved in lerobot dataset
        instance_id_seg: bool = False, # not saved in lerobot dataset
        robot_type: str = "so101_follower",
    ):

        self.robot_type = robot_type
        self.fps = fps
        self.dt = 1 / self.fps
        self.save_mp4 = save_mp4
        self.rgb = True # default to save rgb videos
        self.depth = depth
        self.instance_id_seg = instance_id_seg

        self.camera_features_template = {
            "dtype": "video",
            "shape": (0, 0, 3),
            "names": ["height", "width", "channels"],
        }

        self.FOLLOWER_OBS_FEATURES = {
            "observation.state": {
                "dtype": "float32",
                "fps": self.fps,
                "shape": (6,),
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            }
        }

        self.cameras = cameras

        for camera_name in self.cameras.keys():
            camera = self.cameras[camera_name]

            features = self.camera_features_template.copy()

            features["shape"] = (camera["height"], camera["width"], 3)

            self.FOLLOWER_OBS_FEATURES.update(
                {
                    f"observation.images.{camera_name}": features,
                }
            )

        self.LEADER_ACTION_FEATURES = {
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "fps": self.fps,
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            }
        }

        self.SO101_ACTION_NAMES = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.dataset_features = {
            **self.FOLLOWER_OBS_FEATURES,
            **self.LEADER_ACTION_FEATURES,
        }
        self.num_recorded_episodes = 0

        self.device = device
        self.capcity = 40 * self.fps
        self.current_frame = 0

        self.action_buffers_tensor = None
        self.observation_buffer_tensor = None

        self.rgb_buffer_tensors = {}
        self.depth_buffer_tensors = {}
        self.instance_id_seg_buffers_tensors = {}

        self.episode_queue = queue.Queue()
        self.episode_processor_stop_event = threading.Event()

        self.stop_recording_sub = get_eventdispatcher().observe_event(
            observer_name="stop_recording_observer",
            event_name=self.STOP_RECORDING_EVENT,
            on_event=self.save_episode,
        )

        self.cancel_recording_sub = get_eventdispatcher().observe_event(
            observer_name="cancel_recording_observer",
            event_name=self.CANCEL_RECORDING_EVENT,
            on_event=self.cancel_recording,
        )

        self.episode_processor_thread = threading.Thread(
            target=self.async_episode_processor,
            daemon=True,
        )
        self.episode_processor_thread.start()

    def check_dataset_exists(self):
        if os.path.exists(self.dataset_root):
            return True
        else:
            return False

    def _init_existing_dataset(self):
        self.dataset = LeRobotDataset(
            self.repo_id,
            root=self.dataset_root,
        )
        print(f"[INFO]: Existing dataset initialized - {self.dataset.root}")
        return

    def init_dataset(self):
        if self.check_dataset_exists():
            try:
                self._init_existing_dataset()
                return
            except:
                raise ValueError(
                    f"[ERROR]: Dataset folder exists but cannot be initialized at {self.dataset_root}"
                )

        self.dataset = LeRobotDataset.create(
            self.repo_id,
            fps=self.fps,
            features=self.dataset_features,
            root=self.dataset_root,
            robot_type=self.robot_type,
        )

        print(f"[INFO]: New dataset initialized - {self.dataset.root}")

    def allocate_buffers(self):
        buf_device = "cpu"
        self.action_buffers_tensor = torch.zeros(
            (self.capcity, 6), dtype=torch.float32, device=buf_device
        )
        self.observation_buffer_tensor = torch.zeros(
            (self.capcity, 6), dtype=torch.float32, device=buf_device
        )

        for camera_name in self.cameras.keys():
            if self.rgb:
                self.rgb_buffer_tensors[camera_name] = torch.zeros(
                    (
                        self.capcity,
                        self.cameras[camera_name]["height"],
                        self.cameras[camera_name]["width"],
                        3,
                    ),
                    dtype=torch.uint8,
                    device=buf_device,
                )
            if self.depth:
                self.depth_buffer_tensors[camera_name] = torch.zeros(
                    (self.capcity, self.cameras[camera_name]["height"], self.cameras[camera_name]["width"], 1),
                    dtype=torch.float32,
                    device=buf_device,
                )
            if self.instance_id_seg:
                self.instance_id_seg_buffers_tensors[camera_name] = torch.zeros(
                    (
                        self.capcity,
                        self.cameras[camera_name]["height"],
                        self.cameras[camera_name]["width"],
                        3,
                    ),
                    dtype=torch.uint8,
                    device=buf_device,
                )

    def push_frame_to_buffer(self, action, observation, visual_buffers, depth_buffers, instance_id_seg_buffers):
        if self.current_frame >= self.capcity:
            # TODO: extand tensors to increase the buffer capacity if reached
            print(
                f"[INFO]: Reached the maximum capacity of the buffer. Skipping frame {self.current_frame}"
            )
            return

        if self.observation_buffer_tensor is None:
            self.allocate_buffers()

        self.action_buffers_tensor[self.current_frame] = action.detach().to("cpu")
        self.observation_buffer_tensor[self.current_frame] = observation.detach().to("cpu")

        for camera_name in self.cameras.keys():
            if self.rgb:
                self.rgb_buffer_tensors[camera_name][self.current_frame] = visual_buffers[
                    camera_name
                ].detach().to("cpu")
            if self.depth:
                self.depth_buffer_tensors[camera_name][self.current_frame] = depth_buffers[
                    camera_name
                ].detach().to("cpu")
            if self.instance_id_seg:
                self.instance_id_seg_buffers_tensors[camera_name][self.current_frame] = instance_id_seg_buffers[
                    camera_name
                ].detach().to("cpu")

        self.current_frame += 1

    def add_dataset_frame(self, action, observation, rgb_buffers, frame_index):
        frame = {
            "action": action,
            "observation.state": observation,
            "task": self.task_name,
        }
        for camera_name in self.cameras.keys():
            frame[f"observation.images.{camera_name}"] = rgb_buffers[camera_name]

        self.dataset.add_frame(frame)

    def save_episode(self, event: Event):
        if event.event_name == self.STOP_RECORDING_EVENT:

            print(f"[INFO]: Copy episode to CPU...")
            n = self.current_frame

            cpu_action_buffers_tensor = self.action_buffers_tensor[:n].to("cpu").numpy()
            cpu_obs_buffer_tensor = self.observation_buffer_tensor[:n].to("cpu").numpy()

            cpu_rgb_buffer_tensors = {}

            for camera_name in self.cameras.keys():
                cpu_rgb_buffer_tensors[camera_name] = (
                    self.rgb_buffer_tensors[camera_name][:n].to("cpu").numpy()
                )

            episode_data = {
                "action_buffers": cpu_action_buffers_tensor.copy(),
                "observation_buffer_tensor": cpu_obs_buffer_tensor.copy(),
                "rgb_buffer_tensors": cpu_rgb_buffer_tensors.copy(),
                "total_frames": self.current_frame,
            }

            if self.depth:
                cpu_depth_buffer_tensors = {}
                for camera_name in self.cameras.keys():
                    cpu_depth_buffer_tensors[camera_name] = (
                        self.depth_buffer_tensors[camera_name][:n].to("cpu").numpy()
                    )
                episode_data["depth_buffer_tensors"] = cpu_depth_buffer_tensors.copy()

            if self.instance_id_seg:
                cpu_instance_id_seg_buffers_tensors = {}
                for camera_name in self.cameras.keys():
                    cpu_instance_id_seg_buffers_tensors[camera_name] = (
                        self.instance_id_seg_buffers_tensors[camera_name][:n].to("cpu").numpy()
                    )
                episode_data["instance_id_seg_buffers_tensors"] = cpu_instance_id_seg_buffers_tensors.copy()

            self.episode_queue.put(episode_data)

            print(f"[INFO]: Episode added to queue.")

            self.action_buffers_tensor = None
            self.observation_buffer_tensor = None
            self.rgb_buffer_tensor = {}
            self.depth_buffer_tensors = {}
            self.instance_id_seg_buffers_tensors = {}
            self.current_frame = 0
            print("[INFO]: Cleared buffers")

    def cancel_recording(self, event: Event):
        if event.event_name == self.CANCEL_RECORDING_EVENT:
            print(f"[INFO]: Cancelled recording.")
            self.action_buffers_tensor = None
            self.observation_buffer_tensor = None
            self.rgb_buffer_tensor = {}
            self.current_frame = 0
            print("[INFO]: Cleared buffers")
            print(f"[INFO]: Episode cancelled.")

    def async_episode_processor(self):
        while not self.episode_processor_stop_event.is_set():
            try:
                episode = self.episode_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                print(f"[INFO]: [ASYNC] received episode from queue...")

                action_buffers = episode["action_buffers"]
                observation_buffer_tensor = episode["observation_buffer_tensor"]
                rgb_buffer_tensors = episode["rgb_buffer_tensors"]
                
                total_frames = episode["total_frames"]

                for frame_index in range(total_frames):
                    rgb_buffers = {}
                    for camera_name in self.cameras.keys():
                        rgb_buffers[camera_name] = rgb_buffer_tensors[camera_name][frame_index]

                    self.add_dataset_frame(
                        action_buffers[frame_index],
                        observation_buffer_tensor[frame_index],
                        rgb_buffers,
                        frame_index,
                    )

                # Save depth and RGB videos for each camera (once per episode, after processing all frames)
                this_episode_index = self.dataset.meta.total_episodes + 1

                if self.save_mp4:
                    print(f"[INFO]: Saving mp4videos...")
                    depth_buffer_tensors = episode["depth_buffer_tensors"]
                    instance_id_seg_buffers_tensors = episode["instance_id_seg_buffers_tensors"]

                    for camera_name in self.cameras.keys():
                        depth_frames = depth_buffer_tensors[camera_name][:total_frames]
                        rgb_frames = rgb_buffer_tensors[camera_name][:total_frames]
                        instance_id_seg_frames = instance_id_seg_buffers_tensors[camera_name][:total_frames]

                        self.save_depth_video(depth_frames, camera_name, this_episode_index)
                        self.save_rgb_video(rgb_frames, camera_name, this_episode_index)
                        self.save_instance_id_segmentation_video(instance_id_seg_frames, camera_name, this_episode_index)

                self.dataset.save_episode()
                self.dataset.finalize()
                self._init_existing_dataset()

                self.num_recorded_episodes += 1
                print(f"[INFO]: Episode {self.num_recorded_episodes} saved.")

                if self.episode_queue.empty():
                    print(f"[INFO]: No more episodes in queue. Stopping processor thread...")
                else:
                    print(f"[INFO]: Additional {self.episode_queue.qsize()} episodes in queue.")

            except Exception as e:
                print(f"Error in async processing: {e}")
            finally:
                self.episode_queue.task_done()

    def _save_video(self, frames_rgb, camera_name, data_type, episode_index):
        """
        Save RGB frames as an MP4 video using ffmpeg.

        Args:
            frames_rgb: numpy array of shape (num_frames, height, width, 3), uint8
            camera_name: name of the camera
            data_type: type of data for filename (e.g., "depth", "rgb")
            episode_index: episode number for filename
        """
        fps = self.fps
        root = self.dataset_root
        filename = f"{camera_name}_{data_type}_{episode_index:03d}.mp4"
        filepath = os.path.join(root, "mp4", self.repo_id.split("/")[-1],camera_name, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        num_frames, height, width, _ = frames_rgb.shape
        frames_rgb = np.ascontiguousarray(frames_rgb)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if exists
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",  # Read from stdin
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            filepath,
        ]

        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            frame_data = b"".join(frame.tobytes() for frame in frames_rgb)
            stdout, stderr = process.communicate(input=frame_data)

            if process.returncode != 0:
                print(f"[ERROR]: ffmpeg failed with return code {process.returncode}")
                print(f"[ERROR]: ffmpeg stderr: {stderr.decode()}")
            else:
                print(f"[INFO]: Saved {data_type} video to {filepath}")

        except FileNotFoundError:
            print("[ERROR]: ffmpeg not found. Please install ffmpeg to save videos.")
        except Exception as e:
            print(f"[ERROR]: Failed to save video: {e}")

    def save_depth_video(self, frames, camera_name, episode_index):
        """Save depth frames (float32) as grayscale MP4."""
        num_frames, height, width, channels = frames.shape

        if channels == 1:
            frames = frames.squeeze(-1)  # (num_frames, height, width)

        # Normalize depth values to 0-255 using robust percentile normalization
        valid_mask = np.isfinite(frames)
        if valid_mask.any():
            min_val = np.percentile(frames[valid_mask], 1)
            max_val = np.percentile(frames[valid_mask], 99)
        else:
            min_val, max_val = 0.0, 1.0

        if max_val - min_val < 1e-6:
            max_val = min_val + 1.0

        frames_normalized = np.clip((frames - min_val) / (max_val - min_val), 0, 1)
        frames_uint8 = (frames_normalized * 255).astype(np.uint8)

        # Convert grayscale to RGB for codec compatibility
        frames_rgb = np.stack([frames_uint8] * 3, axis=-1)

        self._save_video(frames_rgb, camera_name, "depth", episode_index)

    def save_rgb_video(self, frames, camera_name, episode_index):
        """Save RGB frames (uint8) as MP4."""
        frames_rgb = frames.astype(np.uint8) if frames.dtype != np.uint8 else frames
        self._save_video(frames_rgb, camera_name, "rgb", episode_index)

    def save_instance_id_segmentation_video(self, frames, camera_name, episode_index):
        """Save instance ID segmentation frames (already colorized RGB) as MP4."""
        frames_rgb = frames.astype(np.uint8) if frames.dtype != np.uint8 else frames
        self._save_video(frames_rgb, camera_name, "instance_id_segmentation", episode_index)

    def _del__(self):
        # stop the episode processor thread
        self.episode_processor_stop_event.set()
        self.episode_processor_thread.join(timeout=0.1)
        self.episode_processor_thread = None
        print("processor thread joined")
