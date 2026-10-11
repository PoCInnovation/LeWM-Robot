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
Statistical analysis of SO101 calibration files.

Reads all .json calibration files from a directory, computes per-joint
mean and std of the range width (range_max - range_min), saves a stats
JSON, and produces a box-and-whisker plot.

Usage:
    python real_robot/so101_calibration_stats.py \
        --calib-dir real_robot/sample_callibrations \
        --output-json real_robot/calibration_stats.json \
        --output-plot real_robot/calibration_stats.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def load_all(calib_dir: Path, exclude: list[str] = None) -> dict[str, list[int]]:
    """Return per-joint lists of motion ranges (abs(range_max - range_min)) across all calibration files."""
    motion_ranges: dict[str, list[int]] = {j: [] for j in JOINTS}
    exclude_stems = set(exclude or [])
    files = [f for f in sorted(calib_dir.glob("*.json")) if f.stem not in exclude_stems]
    if not files:
        raise FileNotFoundError(f"No JSON files found in {calib_dir} (after exclusions)")

    if exclude_stems:
        print(f"Excluding: {', '.join(exclude_stems)}")
    print(f"Using {len(files)} calibration file(s):")
    for f in files:
        print(f"  {f.name}")
        with open(f) as fh:
            data = json.load(fh)
        for joint in JOINTS:
            if joint in data:
                motion_range = abs(data[joint]["range_max"] - data[joint]["range_min"])
                motion_ranges[joint].append(motion_range)

    return motion_ranges, files


def compute_stats(motion_ranges: dict[str, list[int]]) -> dict:
    stats = {}
    for joint, values in motion_ranges.items():
        arr = np.array(values, dtype=float)
        stats[joint] = {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "n": len(arr),
        }
    return stats


def save_stats(stats: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to {path}")


def plot(motion_ranges: dict[str, list[int]], stats: dict, files, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [motion_ranges[j] for j in JOINTS]
    robot_names = [f.stem for f in files]
    colors = plt.cm.tab10.colors

    # Box plots — grey, no fliers (individual points shown separately)
    bp = ax.boxplot(
        data,
        tick_labels=range(1, len(JOINTS) + 1),
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="lightgrey", alpha=0.6),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # One point per calibration file per joint, coloured by file
    for robot_idx, robot_name in enumerate(robot_names):
        color = colors[robot_idx % len(colors)]
        for joint_idx, joint in enumerate(JOINTS, start=1):
            y = motion_ranges[joint][robot_idx]
            ax.scatter(joint_idx, y, color=color, edgecolors="black",
                       linewidths=0.5, zorder=3, s=60)

    ax.set_title(
        "SO101 Calibration: Motion Range per Joint\n(abs(range_max − range_min), encoder counts)",
        fontsize=13,
    )
    ax.set_ylabel("Motion range (encoder counts)")
    ax.set_xlabel("Joint index")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=colors[i % len(colors)], label=name)
        for i, name in enumerate(robot_names)
    ]
    fig.legend(handles=handles, title="Robot", loc="upper right", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Plot saved to {output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-dir", default="real_robot/sample_callibrations")
    parser.add_argument("--output-json", default="real_robot/calibration_stats.json")
    parser.add_argument("--output-plot", default="real_robot/calibration_stats.png")
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        metavar="NAME",
        help="Calibration file stem(s) to exclude (e.g. --exclude follower_arm_1)",
    )
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir)
    motion_ranges, files = load_all(calib_dir, exclude=args.exclude)
    stats = compute_stats(motion_ranges)

    print("\nPer-joint motion range statistics (encoder counts):")
    print(f"  {'Joint':<18} {'Mean':>7} {'Std':>7} {'Min':>6} {'Max':>6} {'N':>4}")
    print("  " + "-" * 52)
    for joint, s in stats.items():
        print(f"  {joint:<18} {s['mean']:>7.1f} {s['std']:>7.1f} {s['min']:>6} {s['max']:>6} {s['n']:>4}")

    save_stats(stats, Path(args.output_json))
    plot(motion_ranges, stats, files, Path(args.output_plot))


if __name__ == "__main__":
    main()
