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
import argparse


from lerobot.datasets.lerobot_dataset import LeRobotDataset


def push_dataset_to_hub(
    repo_id: str, root: str, private: bool = True, tags: list[str] = None
):
    try:
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
    except Exception as e:
        print(f"[ERROR]: Failed to initialize dataset: {e}")
        return

    print(f"[INFO]: Pushing dataset to HuggingFace Hub...")
    dataset.push_to_hub(private=private, tags=tags)


def main():
    parser = argparse.ArgumentParser(description="LeRobot Utils.")
    parser.add_argument(
        "--repo-id", type=str, default=None, help="Repository ID to store the dataset."
    )
    parser.add_argument(
        "--root", type=str, default=None, help="Repository root to store the dataset."
    )
    parser.add_argument(
        "--private", action="store_true", default=False, help="Private dataset."
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags to add to the dataset. Can specify multiple tags.",
    )
    args_cli = parser.parse_args()

    push_dataset_to_hub(
        repo_id=args_cli.repo_id,
        root=args_cli.root,
        private=args_cli.private,
        tags=args_cli.tags,
    )


if __name__ == "__main__":
    main()
