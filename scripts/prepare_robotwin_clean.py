"""Convert pinned official RoboTwin clean archives to LeRobot datasets."""

import argparse
import fcntl
import hashlib
import io
import json
import multiprocessing
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
import requests
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_ID = "TianxingChen/RoboTwin2.0"
ARCHIVE_NAME = "aloha-agilex_clean_50.zip"
CAMERAS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
MOTORS = [
    f"{arm}_{joint}"
    for arm in ("left", "right")
    for joint in (
        "waist", "shoulder", "elbow", "forearm_roll",
        "wrist_angle", "wrist_rotate", "gripper",
    )
]


def official_sources(revision):
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be a full official commit SHA")
    url = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/{revision}/dataset"
    sources = {}
    with requests.Session() as session:
        params = {"recursive": "true", "limit": 1000}
        while url:
            response = session.get(url, params=params, timeout=(10, 60))
            response.raise_for_status()
            for entry in response.json():
                parts = entry["path"].split("/")
                if len(parts) == 3 and parts[-1] == ARCHIVE_NAME:
                    sources[parts[1]] = {
                        "repo_id": REPO_ID,
                        "revision": revision,
                        "path": entry["path"],
                        "size": entry["size"],
                        "sha256": entry["lfs"]["oid"],
                    }
            url = response.links.get("next", {}).get("url")
            params = None
    if len(sources) != 50:
        raise ValueError(f"Expected 50 official clean tasks, found {len(sources)}")
    return sources


def verify_archive(path, source):
    if path.stat().st_size != source["size"]:
        raise ValueError(f"Archive size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != source["sha256"]:
        raise ValueError(f"Official SHA-256 mismatch: {path}")


def dataset_features():
    features = {
        key: {"dtype": "float32", "shape": (14,), "names": MOTORS}
        for key in ("observation.state", "action")
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    return features


def read_joints(source):
    joints = np.concatenate([
        source["joint_action/left_arm"][:],
        source["joint_action/left_gripper"][:].reshape(-1, 1),
        source["joint_action/right_arm"][:],
        source["joint_action/right_gripper"][:].reshape(-1, 1),
    ], axis=1).astype(np.float32)
    if joints.ndim != 2 or joints.shape[1] != 14 or len(joints) < 2:
        raise ValueError(f"Invalid source joint shape: {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("Non-finite values in official source joints")
    for camera in CAMERAS.values():
        if len(source[f"observation/{camera}/rgb"]) != len(joints):
            raise ValueError(f"Source camera/joint length mismatch: {camera}")
    return joints


def validate_dataset(root, repo_id, episodes, expected_frames=None):
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    if dataset.num_episodes != episodes:
        raise ValueError(f"Unexpected episode count in {root}: {dataset.num_episodes}")
    if expected_frames is not None and len(dataset) != expected_frames:
        raise ValueError(f"Unexpected frame count in {root}: {len(dataset)}")
    for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        item = dataset[index]
        for key in ("observation.state", "action"):
            values = np.asarray(item[key])
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} at frame {index} in {root}")
        for camera in CAMERAS:
            if tuple(item[f"observation.images.{camera}"].shape) != (3, 480, 640):
                raise ValueError(f"Invalid camera shape: {camera} in {root}")
    return len(dataset)


def convert_task(archive_path, output_root, task, source_info, episodes):
    lock_root = output_root / ".prepare_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f"{task}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return convert_locked_task(archive_path, output_root, task, source_info, episodes)


def convert_locked_task(archive_path, output_root, task, source_info, episodes):
    cv2.setNumThreads(1)
    verify_archive(archive_path, source_info)
    root = output_root / task
    repo_id = f"robotwin_clean/{task}"
    marker = root / "meta" / "robotwin_source.json"
    if root.exists():
        if not marker.exists():
            raise FileExistsError(f"Refusing to overwrite incomplete/unmarked dataset: {root}")
        previous = json.loads(marker.read_text())
        if previous["source"] != source_info or previous["episodes"] != episodes:
            raise ValueError(f"Existing dataset provenance differs: {root}")
        validate_dataset(root, repo_id, episodes, previous["frames"])
        print(f"VERIFIED_EXISTING {task}: {episodes} episodes", flush=True)
        return root

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=50,
        robot_type="aloha",
        features=dataset_features(),
        use_videos=True,
        image_writer_threads=4,
        video_backend="pyav",
    )
    total_frames = 0
    chosen_instructions = []
    random = np.random.default_rng(42)
    prefix = ARCHIVE_NAME.removesuffix(".zip")
    with zipfile.ZipFile(archive_path) as archive:
        for episode_id in range(episodes):
            instruction_path = f"{prefix}/instructions/episode{episode_id}.json"
            instructions = json.loads(archive.read(instruction_path))["seen"]
            if not instructions or not all(isinstance(text, str) and text.strip() for text in instructions):
                raise ValueError(f"Missing seen instructions: {task}/{episode_id}")
            instruction = str(random.choice(instructions))
            chosen_instructions.append(instruction)
            hdf5_path = f"{prefix}/data/episode{episode_id}.hdf5"
            with h5py.File(io.BytesIO(archive.read(hdf5_path)), "r") as source:
                joints = read_joints(source)
                for frame_index in range(len(joints) - 1):
                    frame = {
                        "observation.state": joints[frame_index],
                        "action": joints[frame_index + 1],
                        "task": instruction,
                    }
                    for camera, source_camera in CAMERAS.items():
                        encoded = source[f"observation/{source_camera}/rgb"][frame_index]
                        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError(f"Invalid image: {task}/{episode_id}/{source_camera}/{frame_index}")
                        frame[f"observation.images.{camera}"] = cv2.resize(image, (640, 480))
                    dataset.add_frame(frame)
                dataset.save_episode(parallel_encoding=False)
                total_frames += len(joints) - 1
            print(f"CONVERTED {task} episode {episode_id + 1}/{episodes}", flush=True)
    dataset.finalize()
    dataset.stop_image_writer()
    del dataset
    validate_dataset(root, repo_id, episodes, total_frames)
    provenance = {
        "source": source_info,
        "episodes": episodes,
        "frames": total_frames,
        "fps": 50,
        "instruction_split": "seen",
        "instructions": chosen_instructions,
        "source_episode_ids": list(range(episodes)),
        "state_action_alignment": "state[t], action[t+1], images[t]; last observation dropped",
        "image_processing": "RoboTwin pi0: cv2.IMREAD_COLOR, resize to 640x480, separate cameras",
    }
    marker.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"VERIFIED {task}: {episodes} episodes, {total_frames} frames, official SHA-256 matched", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--episodes", type=int, default=50, choices=range(1, 51))
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sources = official_sources(args.revision)
    tasks = sorted(args.tasks or sources)
    if len(set(tasks)) != len(tasks) or set(tasks) - sources.keys():
        raise ValueError("Tasks must be distinct names from the official clean manifest")
    if args.manifest and (set(tasks) != sources.keys() or args.episodes != 50):
        raise ValueError("A competition manifest requires all 50 tasks, 50 episodes each")
    if any(args.input_root.rglob("*randomized*")):
        raise ValueError("Input root must contain clean archives only")
    for task in tasks:
        path = args.input_root / sources[task]["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Download the official clean archive first: {path}")
    if args.workers == 1:
        for task in tasks:
            convert_task(args.input_root / sources[task]["path"], args.output_root, task, sources[task], args.episodes)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            pending = {
                executor.submit(convert_task, args.input_root / sources[task]["path"], args.output_root,
                                task, sources[task], args.episodes): task
                for task in reversed(tasks)
            }
            for future in as_completed(pending):
                future.result()
    roots = [args.output_root / task for task in tasks]
    if args.manifest:
        content = "".join(f"robotwin {root.resolve()}\n" for root in roots)
        if args.manifest.exists() and args.manifest.read_text() != content:
            raise FileExistsError(f"Refusing to replace a different manifest: {args.manifest}")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(content)
        print(f"CLEAN_ONLY_MANIFEST {args.manifest}: {len(roots)} tasks", flush=True)


if __name__ == "__main__":
    main()