#!/usr/bin/env python3
"""End-to-end automated smoke test and pipeline validation for LeWM-Robot.

This script executes the entire pipeline sequentially:
  1. Precompute features over a tiny slice of LeRobot (temporal mode enabled)
  2. Train the behavior cloning head (BC policy) for a few fast epochs
  3. Train the future feature predictor (dynamic world model) for a few fast epochs
  4. Generate behavior cloning policy execution plots (demo)
  5. Generate future frame prediction plots via nearest-neighbor (demo_prediction)

If any step fails, the script reports the failure and aborts with a non-zero exit code.
"""

from __future__ import annotations
import os
import sys
import time
import subprocess
import shutil
from pathlib import Path


def log_header(title: str):
    print("=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)


def run_command(cmd: list[str], cwd: Path | None = None) -> float:
    print(f"Executing: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print("\n❌ Command Failed!")
        print("Stdout:")
        print(result.stdout)
        print("Stderr:")
        print(result.stderr)
        raise RuntimeError(f"Command {' '.join(cmd)} failed with code {result.returncode}")
    
    print(f"✅ Completed successfully in {elapsed:.1f}s\n")
    return elapsed


def main():
    root_dir = Path(__file__).resolve().parent
    cache_dir = root_dir / "cached_features" / "smoke_test_cache"
    checkpoints_dir = root_dir / "checkpoints" / "smoke_test_runs"
    
    # 0. Clean up previous smoke test runs to ensure a fresh test
    if cache_dir.exists():
        print(f"Cleaning existing cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    if checkpoints_dir.exists():
        print(f"Cleaning existing checkpoints: {checkpoints_dir}")
        shutil.rmtree(checkpoints_dir)
        
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Check hardware
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # To run extremely fast on GPU or CPU without blowing up memory/compute
    dtype = "float16" if device == "cuda" else "float32"
    
    print(f"Starting End-to-End Smoke Test on Device: {device} (Dtype: {dtype})")
    
    timings = {}
    
    try:
        # ---- STEP 1: PRECOMPUTE TEMPORAL FEATURES ----
        log_header("STEP 1: Precomputing Temporal V-JEPA 2 Features")
        precompute_cmd = [
            sys.executable, "precompute_features.py",
            "--src_repo", "lerobot/libero_object_image",
            "--dst_dir", str(cache_dir),
            "--vjepa2_repo", "facebook/vjepa2-vitl-fpc64-256",
            "--batch_size", "4",
            "--dtype", dtype,
            "--device", device,
            "--num_workers", "0",  # 0 workers is safest across different OS environments
            "--max_episodes", "2",  # Minimum episodes to allow training/val split
            "--frame_stride", "20",  # Extremely sparse to run in seconds
        ]
        timings["Precompute"] = run_command(precompute_cmd, cwd=root_dir)
        
        # Validate that the cache metadata and files exist
        metadata_file = cache_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Cache metadata file {metadata_file} was not generated!")
        print("✔ Metadata file successfully generated and verified.")
        
        # ---- STEP 2: TRAIN BEHAVIOR CLONING POLICY HEAD ----
        log_header("STEP 2: Training Behavior Cloning Policy Head")
        train_bc_cmd = [
            sys.executable, "train.py",
            "--cache_dir", str(cache_dir),
            "--output_dir", str(checkpoints_dir / "bc"),
            "--val_episodes", "1",
            "--epochs", "2",
            "--batch_size", "2",
            "--num_workers", "0",
            "--device", device,
        ]
        timings["Train BC"] = run_command(train_bc_cmd, cwd=root_dir)
        
        bc_checkpoint = checkpoints_dir / "bc" / "best.pt"
        if not bc_checkpoint.exists():
            raise FileNotFoundError(f"BC checkpoint {bc_checkpoint} was not found!")
        print("✔ BC policy checkpoint successfully trained and verified.")
        
        # ---- STEP 3: TRAIN FUTURE FEATURE PREDICTOR ----
        log_header("STEP 3: Training Future Feature Predictor (World Model)")
        train_pred_cmd = [
            sys.executable, "train_predictor.py",
            "--cache_dir", str(cache_dir),
            "--output_dir", str(checkpoints_dir / "predictor"),
            "--horizon", "1",  # Horizon 1 is safest for sparse stride
            "--val_episodes", "1",
            "--epochs", "2",
            "--batch_size", "2",
            "--num_workers", "0",
            "--device", device,
        ]
        timings["Train Predictor"] = run_command(train_pred_cmd, cwd=root_dir)
        
        pred_checkpoint = checkpoints_dir / "predictor" / "best.pt"
        if not pred_checkpoint.exists():
            raise FileNotFoundError(f"Predictor checkpoint {pred_checkpoint} was not found!")
        print("✔ Future feature predictor checkpoint successfully trained and verified.")
        
        # ---- STEP 4: GENERATE BEHAVIOR CLONING DEMO PLOT ----
        log_header("STEP 4: Generating Behavior Cloning Demo Plot")
        bc_demo_output = checkpoints_dir / "bc_demo.png"
        demo_bc_cmd = [
            sys.executable, "demo.py",
            "--cache_dir", str(cache_dir),
            "--checkpoint", str(bc_checkpoint),
            "--output", str(bc_demo_output),
            "--device", device,
        ]
        timings["Demo BC"] = run_command(demo_bc_cmd, cwd=root_dir)
        
        if not bc_demo_output.exists():
            raise FileNotFoundError(f"BC demo plot {bc_demo_output} was not found!")
        print("✔ BC demo plot successfully generated and verified.")
        
        # ---- STEP 5: GENERATE FUTURE FRAME PREDICTION DEMO PLOT ----
        log_header("STEP 5: Generating Future Frame Prediction Plot")
        pred_demo_output = checkpoints_dir / "prediction_demo.png"
        demo_pred_cmd = [
            sys.executable, "demo_prediction.py",
            "--cache_dir", str(cache_dir),
            "--checkpoint", str(pred_checkpoint),
            "--src_repo", "lerobot/libero_object_image",
            "--output", str(pred_demo_output),
            "--num_steps", "1",  # Minimal rows to make it ultra-fast
            "--device", device,
        ]
        timings["Demo Predictor"] = run_command(demo_pred_cmd, cwd=root_dir)
        
        if not pred_demo_output.exists():
            raise FileNotFoundError(f"Predictor demo plot {pred_demo_output} was not found!")
        print("✔ Predictor demo plot successfully generated and verified.")
        
        # ---- FINAL SUCCESS SUMMARY ----
        print("\n" + "=" * 80)
        print("🎉 END-TO-END PIPELINE SMOKE TEST: GLORIOUS SUCCESS!")
        print("=" * 80)
        for stage, t in timings.items():
            print(f"  - {stage:<25}: {t:>6.1f}s")
        print("-" * 80)
        print(f"Total validation time: {sum(timings.values()):.1f}s")
        print(f"All outputs successfully written to: {checkpoints_dir}")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline Smoke Test FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
