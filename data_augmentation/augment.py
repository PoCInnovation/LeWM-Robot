import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_augmentation import transforms as T

DEFAULT_SRC = "duck_dataset"
DEFAULT_DST = "duck_dataset_augmented"
DEFAULT_FPS = 30
CAMERAS = ["observation.images.wrist", "observation.images.front"]

AUG_REGISTRY = {
    "crop_resize":   {"flag": "crop",          "kind": "visual",   "help": "Recadrage + zoom (fixe sur l'episode)"},
    "gaussian_blur": {"flag": "blur",          "kind": "visual",   "help": "Flou gaussien (constant sur l'episode)"},
    "color_jitter":  {"flag": "color",         "kind": "visual",   "help": "Luminosite/contraste/saturation (constants sur l'episode)"},
    "speed_slow":    {"flag": "speed-slow",    "kind": "temporal", "help": "Ralentit l'episode (x0.85)"},
    "speed_fast":    {"flag": "speed-fast",    "kind": "temporal", "help": "Accelere l'episode (x1.15)"},
    "frame_drop":    {"flag": "frame-drop",    "kind": "temporal", "help": "Supprime quelques frames"},
    "temporal_crop": {"flag": "temporal-crop", "kind": "temporal", "help": "Garde une sous-fenetre temporelle"},
    "motor_noise":   {"flag": "motor-noise",   "kind": "temporal", "help": "Bruit gaussien sur action + etat moteur"},
}

QUANTILES = [("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)]
STATS_FRAMES_PER_EP = 4
STATS_IMG_W = 160
STATS_IMG_H = 120


def extract_episode_frames(video_path, from_ts, to_ts):
    duration = round(to_ts - from_ts, 6)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(from_ts), "-i", str(video_path),
            "-t", str(duration), "-pix_fmt", "bgr24", "-vf", "scale=640:480",
            f"{tmp}/%06d.png",
        ], check=True, capture_output=True)
        return [cv2.imread(str(p)) for p in sorted(Path(tmp).glob("*.png"))]


def write_clip(frames, out_path, fps):
    if not frames:
        raise ValueError(f"Aucune frame pour {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            cv2.imwrite(f"{tmp}/{i:06d}.png", f)
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps), "-i", f"{tmp}/%06d.png",
            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(out_path),
        ], check=True, capture_output=True)


def concat_clips(clip_paths, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
        concat_file = f.name
    for codec, extra in [("libsvtav1", ["-crf", "35"]), ("libx264", ["-crf", "23"])]:
        r = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c:v", codec] + extra + ["-pix_fmt", "yuv420p", str(out_path)],
            capture_output=True)
        if r.returncode == 0:
            Path(concat_file).unlink(missing_ok=True)
            return
    raise RuntimeError(f"Concat ffmpeg echouee pour {out_path}")


def aug_crop_resize(fw, ff, rng):
    h, w = fw[0].shape[:2]
    params = T.sample_crop(rng, h, w)
    return ([T.apply_crop(a, params) for a in fw],
            [T.apply_crop(b, params) for b in ff])


def aug_gaussian_blur(fw, ff, rng):
    params = T.sample_blur(rng)
    return ([T.apply_blur(a, params) for a in fw],
            [T.apply_blur(b, params) for b in ff])


def aug_color_jitter(fw, ff, rng, brightness=0.2, contrast=0.2, saturation=0.2):
    params = T.sample_color(rng, T.Strength(color_brightness=brightness,
                                            color_contrast=contrast,
                                            color_saturation=saturation))
    return ([T.apply_color(a, params) for a in fw],
            [T.apply_color(b, params) for b in ff])


def aug_speed(fw, ff, df, rng, factor):
    n = len(fw)
    n_new = max(2, int(n / factor))
    t_o = np.linspace(0, 1, n)
    t_n = np.linspace(0, 1, n_new)
    idx = np.clip((t_n * (n - 1)).astype(int), 0, n - 1)
    nw = [fw[i] for i in idx]
    nf = [ff[i] for i in idx]
    ndf = df.iloc[idx].copy().reset_index(drop=True)
    for col in ["action", "observation.state"]:
        arr = np.stack(df[col].values).astype(np.float32)
        out = np.zeros((n_new, arr.shape[1]), dtype=np.float32)
        for j in range(arr.shape[1]):
            out[:, j] = interp1d(t_o, arr[:, j])(t_n)
        ndf[col] = list(out)
    return nw, nf, ndf


def aug_frame_drop(fw, ff, df, rng, rate=0.05):
    n = len(fw)
    mask = rng.random(n) > rate
    mask[0] = mask[-1] = True
    idx = np.where(mask)[0]
    return [fw[i] for i in idx], [ff[i] for i in idx], df.iloc[idx].copy()


def aug_temporal_crop(fw, ff, df, rng, ratio=0.7):
    n = len(fw)
    length = max(2, int(n * ratio))
    s = int(rng.integers(0, n - length + 1))
    return fw[s:s + length], ff[s:s + length], df.iloc[s:s + length].copy()


def aug_motor_noise(fw, ff, df, rng, sigma=0.01):
    """Equivalent offline de la technique `sensor` du pipeline online."""
    ndf = df.copy()
    strength = T.Strength(sensor_sigma=sigma)
    for col in ["action", "observation.state"]:
        arr = np.stack(ndf[col].values).astype(np.float32)
        noise = T.sample_sensor_noise(rng, arr.shape, strength)
        ndf[col] = list(T.apply_sensor_noise(arr, noise))
    return fw, ff, ndf


def apply_augmentation(aug_name, fw, ff, df_ep, rng):
    if aug_name == "crop_resize":
        nw, nf = aug_crop_resize(fw, ff, rng)
        return nw, nf, df_ep.copy()
    if aug_name == "gaussian_blur":
        nw, nf = aug_gaussian_blur(fw, ff, rng)
        return nw, nf, df_ep.copy()
    if aug_name == "color_jitter":
        nw, nf = aug_color_jitter(fw, ff, rng)
        return nw, nf, df_ep.copy()
    if aug_name == "speed_slow":
        return aug_speed(fw, ff, df_ep, rng, 0.85)
    if aug_name == "speed_fast":
        return aug_speed(fw, ff, df_ep, rng, 1.15)
    if aug_name == "frame_drop":
        return aug_frame_drop(fw, ff, df_ep, rng)
    if aug_name == "temporal_crop":
        return aug_temporal_crop(fw, ff, df_ep, rng)
    if aug_name == "motor_noise":
        return aug_motor_noise(fw, ff, df_ep, rng)
    raise ValueError(f"Augmentation inconnue : {aug_name}")


def fix_indices(df, ep_idx, global_offset, fps):
    n = len(df)
    df = df.copy().reset_index(drop=True)
    df["episode_index"] = np.int64(ep_idx)
    df["frame_index"] = np.arange(n, dtype=np.int64)
    df["timestamp"] = np.arange(n, dtype=np.float32) / fps
    df["index"] = np.arange(global_offset, global_offset + n, dtype=np.int64)
    df["task_index"] = np.int64(0)
    return df


def make_ep_meta_row(ep_idx, df_ep, orig_row, aug_name=None, source_ep=None):
    n = len(df_ep)
    from_ts = float(df_ep["timestamp"].iloc[0])
    to_ts = float(df_ep["timestamp"].iloc[-1])

    orig_task = orig_row["tasks"][0] if hasattr(orig_row["tasks"], "__len__") else orig_row["tasks"]
    if " | aug:" in str(orig_task):
        orig_task = str(orig_task).split(" | aug:")[0]

    if aug_name is None:
        task_str = str(orig_task)
    else:
        task_str = f"{orig_task} | aug:{aug_name} | source:{source_ep}"

    row = {
        "episode_index":      ep_idx,
        "tasks":              np.array([task_str], dtype=object),
        "length":             n,
        "data/chunk_index":   0,
        "data/file_index":    0,
        "dataset_from_index": int(df_ep["index"].iloc[0]),
        "dataset_to_index":   int(df_ep["index"].iloc[-1]) + 1,
    }
    for cam in CAMERAS:
        row[f"videos/{cam}/chunk_index"] = 0
        row[f"videos/{cam}/file_index"] = 0
        row[f"videos/{cam}/from_timestamp"] = from_ts
        row[f"videos/{cam}/to_timestamp"] = to_ts

    for feat in ["action", "observation.state"]:
        arr = np.stack(df_ep[feat].values).astype(np.float32)
        for stat, val in [
            ("min", arr.min(0)), ("max", arr.max(0)),
            ("mean", arr.mean(0)), ("std", arr.std(0)),
            ("count", np.full(arr.shape[1], float(n))),
        ]:
            row[f"stats/{feat}/{stat}"] = val.tolist()
        for q, p in [("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)]:
            row[f"stats/{feat}/{q}"] = np.percentile(arr, p, axis=0).tolist()

    for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        vals = df_ep[col].values.astype(float)
        for stat, val in [
            ("min", vals.min()), ("max", vals.max()),
            ("mean", vals.mean()), ("std", vals.std()), ("count", float(n)),
        ]:
            row[f"stats/{col}/{stat}"] = val
        for q, p in [("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)]:
            row[f"stats/{col}/{q}"] = float(np.percentile(vals, p))

    for col in orig_row.index:
        if col.startswith("meta/") and col not in row:
            row[col] = orig_row[col]

    return row


def save_parquet_checkpoint(dst, all_dfs, all_ep_metas):
    out_data = dst / "data/chunk-000/file-000.parquet"
    out_ep = dst / "meta/episodes/chunk-000/file-000.parquet"
    out_data.parent.mkdir(parents=True, exist_ok=True)
    out_ep.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_dfs, ignore_index=True).to_parquet(out_data, index=False)
    pd.DataFrame(all_ep_metas).to_parquet(out_ep, index=False)
    print(f"    checkpoint ({len(all_ep_metas)} episodes sauvegardes)")


def generate_augmentation_map(dst, all_ep_metas, augmentations_used):
    episodes = {}
    for row in all_ep_metas:
        ep_idx = int(row["episode_index"])
        task_arr = row["tasks"]
        task_str = str(task_arr[0]) if hasattr(task_arr, "__len__") else str(task_arr)
        if "| aug:" in task_str:
            aug = task_str.split("| aug:")[1].split(" |")[0]
            src = int(task_str.split("| source:")[1])
            episodes[str(ep_idx)] = {"type": "augmented", "source_ep": src, "aug": aug}
        else:
            episodes[str(ep_idx)] = {"type": "original", "source_ep": None, "aug": None}

    result = {
        "total_episodes":     len(episodes),
        "n_original":         sum(1 for v in episodes.values() if v["type"] == "original"),
        "n_augmented":        sum(1 for v in episodes.values() if v["type"] == "augmented"),
        "augmentations_used": augmentations_used,
        "episodes":           episodes,
    }
    with open(dst / "meta/augmentation_map.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  augmentation_map.json -> {len(episodes)} episodes")


def sample_frames_for_stats(frames):
    if not frames:
        return []
    n = min(STATS_FRAMES_PER_EP, len(frames))
    idx = np.unique(np.linspace(0, len(frames) - 1, n).astype(int))
    return [cv2.cvtColor(cv2.resize(frames[i], (STATS_IMG_W, STATS_IMG_H)), cv2.COLOR_BGR2RGB)
            for i in idx]


def _vector_stats(arr):
    arr = arr.astype(np.float64)
    d = {
        "min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
        "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
        "count": [int(arr.shape[0])],
    }
    for name, p in QUANTILES:
        d[name] = np.percentile(arr, p, axis=0).tolist()
    return d


def _image_stats(frames_rgb):
    arr = np.stack(frames_rgb).astype(np.float64) / 255.0
    flat = arr.reshape(-1, 3)

    def shape3(v):
        return v.reshape(3, 1, 1).tolist()

    d = {
        "min": shape3(flat.min(0)), "max": shape3(flat.max(0)),
        "mean": shape3(flat.mean(0)), "std": shape3(flat.std(0)),
        "count": [int(arr.shape[0])],
    }
    for name, p in QUANTILES:
        d[name] = shape3(np.percentile(flat, p, axis=0))
    return d


def build_stats(df, img_samples):
    stats = {}
    for col in ["action", "observation.state"]:
        stats[col] = _vector_stats(np.stack(df[col].values))
    for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        stats[col] = _vector_stats(df[col].values.reshape(-1, 1).astype(np.float64))
    for cam, frames in img_samples.items():
        if frames:
            stats[cam] = _image_stats(frames)
    return stats


def sample_video_frames(video, n_target=2000):
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip())
    target_fps = max(0.05, n_target / dur)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"fps={target_fps},scale={STATS_IMG_W}:{STATS_IMG_H}",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True)
    buf = np.frombuffer(p.stdout, np.uint8)
    frame_bytes = STATS_IMG_W * STATS_IMG_H * 3
    n = buf.size // frame_bytes
    frames = buf[:n * frame_bytes].reshape(n, STATS_IMG_H, STATS_IMG_W, 3)
    return [frames[i] for i in range(n)]


def regen_stats_only(dst):
    print(f"Regeneration de meta/stats.json pour {dst}")
    df = pd.read_parquet(dst / "data/chunk-000/file-000.parquet")
    print(f"Parquet : {len(df)} frames, {df.episode_index.nunique()} episodes")
    img_samples = {}
    for cam in CAMERAS:
        video = dst / f"videos/{cam}/chunk-000/file-000.mp4"
        print(f"Echantillonnage {cam} ...", end=" ", flush=True)
        img_samples[cam] = sample_video_frames(video)
        print(f"{len(img_samples[cam])} frames")
    stats = build_stats(df, img_samples)
    with open(dst / "meta/stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print(f"meta/stats.json regenere sur {len(df)} frames")


def run(src, dst, fps, n_orig, selected, seed):
    print(f"Source      : {src}")
    print(f"Destination : {dst}")
    print(f"Techniques  : {', '.join(selected)}")
    print(f"Seed        : {seed}")

    rng = np.random.default_rng(seed)

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")

    df_all = pd.read_parquet(src / "data/chunk-000/file-000.parquet")
    ep_meta = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    with open(src / "meta/info.json") as f:
        info = json.load(f)

    if n_orig <= 0:
        n_orig = int(ep_meta["episode_index"].max()) + 1

    src_vid = {cam: src / f"videos/{cam}/chunk-000/file-000.mp4" for cam in CAMERAS}

    all_dfs, all_ep_metas = [], []
    global_offset = 0
    new_ep_idx = 0

    total = n_orig * (1 + len(selected))
    print(f"Plan : {n_orig} originaux x (1 + {len(selected)}) = {total} episodes")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        clips = {cam: [] for cam in CAMERAS}
        img_samples = {cam: [] for cam in CAMERAS}

        for orig_ep in range(n_orig):
            meta_row = ep_meta[ep_meta["episode_index"] == orig_ep].iloc[0]
            df_ep = df_all[df_all["episode_index"] == orig_ep].copy()
            from_ts = float(meta_row[f"videos/{CAMERAS[0]}/from_timestamp"])
            to_ts = float(meta_row[f"videos/{CAMERAS[0]}/to_timestamp"])

            print(f"[Ep {orig_ep:02d}/{n_orig - 1}] extraction...", end=" ", flush=True)
            fw = extract_episode_frames(src_vid[CAMERAS[0]], from_ts, to_ts)
            ff = extract_episode_frames(src_vid[CAMERAS[1]], from_ts, to_ts)
            print(f"{len(fw)} frames")

            for cam, frames in zip(CAMERAS, [fw, ff]):
                short = "wrist" if "wrist" in cam else "front"
                clip = tmpdir / f"ep{new_ep_idx:04d}_{short}.mp4"
                write_clip(frames, clip, fps)
                clips[cam].append(clip)
                img_samples[cam] += sample_frames_for_stats(frames)

            ndf = fix_indices(df_ep, new_ep_idx, global_offset, fps)
            all_dfs.append(ndf)
            all_ep_metas.append(make_ep_meta_row(new_ep_idx, ndf, meta_row))
            global_offset += len(ndf)
            new_ep_idx += 1

            for aug in selected:
                nw, nf, ndf2 = apply_augmentation(aug, fw, ff, df_ep, rng)
                print(f"  {aug:15s} ep {new_ep_idx:03d} ({len(ndf2)} frames)")

                for cam, frames in zip(CAMERAS, [nw, nf]):
                    short = "wrist" if "wrist" in cam else "front"
                    clip = tmpdir / f"ep{new_ep_idx:04d}_{short}.mp4"
                    write_clip(frames, clip, fps)
                    clips[cam].append(clip)
                    img_samples[cam] += sample_frames_for_stats(frames)

                ndf2 = fix_indices(ndf2, new_ep_idx, global_offset, fps)
                all_dfs.append(ndf2)
                all_ep_metas.append(make_ep_meta_row(new_ep_idx, ndf2, meta_row,
                                                     aug_name=aug, source_ep=orig_ep))
                global_offset += len(ndf2)
                new_ep_idx += 1

            if orig_ep % 5 == 4:
                save_parquet_checkpoint(dst, all_dfs, all_ep_metas)

        print(f"Concatenation de {new_ep_idx} clips par camera...")
        for cam in CAMERAS:
            concat_clips(clips[cam], dst / f"videos/{cam}/chunk-000/file-000.mp4")

    save_parquet_checkpoint(dst, all_dfs, all_ep_metas)
    generate_augmentation_map(dst, all_ep_metas, selected)

    df_final = pd.concat(all_dfs, ignore_index=True)
    info["total_episodes"] = new_ep_idx
    info["total_frames"] = len(df_final)
    info["splits"]["train"] = f"0:{new_ep_idx}"
    with open(dst / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    stats = build_stats(df_final, img_samples)
    with open(dst / "meta/stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Termine : {new_ep_idx} episodes, {len(df_final)} frames dans {dst}")


def build_parser():
    p = argparse.ArgumentParser(description="Data augmentation pour datasets LeRobot (format v3.0).")
    p.add_argument("--src", default=DEFAULT_SRC, help=f"Dataset source (defaut: {DEFAULT_SRC})")
    p.add_argument("--dst", default=DEFAULT_DST, help=f"Dataset de sortie (defaut: {DEFAULT_DST})")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"FPS (defaut: {DEFAULT_FPS})")
    p.add_argument("--n-orig", type=int, default=0, help="Episodes a traiter (0 = tous)")
    p.add_argument("--seed", type=int, default=42, help="Graine aleatoire (defaut: 42)")
    p.add_argument("--all", action="store_true", help="Applique toutes les techniques")
    p.add_argument("--regen-stats-only", action="store_true",
                   help="Recalcule seulement meta/stats.json du dataset --dst")
    grp = p.add_argument_group("Techniques (cumulables)")
    for name, meta in AUG_REGISTRY.items():
        grp.add_argument(f"--{meta['flag']}", dest=name, action="store_true", help=meta["help"])
    return p


def resolve_selected(args):
    if args.all:
        return list(AUG_REGISTRY.keys())
    return [name for name in AUG_REGISTRY if getattr(args, name)]


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.regen_stats_only:
        regen_stats_only(Path(args.dst))
        return

    selected = resolve_selected(args)
    if not selected:
        parser.error("Aucune technique selectionnee. Utilisez --all ou un flag (ex: --crop --blur).")

    run(Path(args.src), Path(args.dst), args.fps, args.n_orig, selected, args.seed)


if __name__ == "__main__":
    main()
