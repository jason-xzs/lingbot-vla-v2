"""Build the clean-only object library and background statistics for scene augmentation.

All pixels come from episode first frames of the official clean RoboTwin demonstrations,
decoded with the repository uint8 decoder and resized exactly like training inputs.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingbotvla.data.vla_data.scene_augmentation import (  # noqa: E402
    BODY_DEVIATION,
    HEAD_CAMERA,
    SOFT_DEVIATION,
    WRIST_CAMERAS,
    task_family,
)

MAX_BODY_AREA = 3500
MAX_REGION_AREA = 9000
MIN_BODY_AREA = 40


def decode_first_frames(root):
    from prepare_robotwin_augmentation import open_source, read_source_frames

    dataset = open_source(root)
    frames = {}
    for camera in dataset.meta.camera_keys:
        frames[camera] = np.stack([read_source_frames(dataset, episode, camera, [0])[0].numpy()
                                   for episode in range(dataset.meta.total_episodes)])
    return root.name, frames


def extract_objects(frames, median):
    """Split first-frame foreground into object crops with their own soft shadows."""
    crops, masks = [], []
    m = median.astype(np.float32)
    for episode, frame in enumerate(frames):
        f = frame.astype(np.float32)
        deviation = np.abs(f - m).max(0)
        soft = deviation > SOFT_DEVIATION
        body = ndimage.binary_fill_holes(deviation > BODY_DEVIATION)
        masks.append(ndimage.binary_dilation(soft, iterations=2))
        labels, count = ndimage.label(soft)
        for index, region in enumerate(ndimage.find_objects(labels), 1):
            component = labels[region] == index
            component_body = body[region] & component
            y, x = region[0].start, region[1].start
            h, w = component.shape
            touches_border = y <= 1 or x <= 1 or y + h >= frame.shape[1] - 1 or x + w >= frame.shape[2] - 1
            if (touches_border or component_body.sum() < MIN_BODY_AREA or component_body.sum() > MAX_BODY_AREA
                    or component.sum() > MAX_REGION_AREA):
                continue
            ratio = (f[:, region[0], region[1]] / np.maximum(m[:, region[0], region[1]], 1)).mean(0)
            shade = np.where(component & ~component_body, np.clip(ratio, 0.3, 1.0), 1.0)
            crops.append(dict(episode=episode, y=y, x=x, h=h, w=w,
                              rgb=frame[:, region[0], region[1]].copy(), body=component_body,
                              shade=(shade * 255).round().astype(np.uint8)))
    return crops, np.stack(masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="assets/training_data/robotwin_clean.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=25)
    args = parser.parse_args()
    output = Path(args.output)
    if (output / "library.npz").exists():
        raise FileExistsError(f"Refusing to replace {output / 'library.npz'}; use a new version directory")
    output.mkdir(parents=True, exist_ok=True)
    roots = [Path(line.split()[1]) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with ProcessPoolExecutor(args.workers) as executor:
        decoded = dict(executor.map(decode_first_frames, roots))
    tasks = sorted(decoded)
    medians = {camera: np.median(np.concatenate([decoded[t][camera] for t in tasks]), axis=0).astype(np.uint8)
               for camera in [HEAD_CAMERA, *WRIST_CAMERAS]}

    meta, offsets, rgb, body, shade, fg_masks, report = [], [0], [], [], [], [], {}
    for task_id, task in enumerate(tasks):
        crops, masks = extract_objects(decoded[task][HEAD_CAMERA], medians[HEAD_CAMERA])
        fg_masks.append(np.packbits(masks.reshape(masks.shape[0], -1), axis=1))
        for crop in crops:
            meta.append([task_id, crop["episode"], crop["y"], crop["x"], crop["h"], crop["w"]])
            rgb.append(crop["rgb"].reshape(3, -1).T)
            body.append(crop["body"].reshape(-1))
            shade.append(crop["shade"].reshape(-1))
            offsets.append(offsets[-1] + crop["h"] * crop["w"])
        report[task] = {"crops": len(crops), "family": sorted(task_family(task))}
        print(f"{task}: {len(crops)} crops", flush=True)

    np.savez(output / "library.npz",
             tasks=np.array(tasks), meta=np.array(meta, dtype=np.int32), offsets=np.array(offsets, dtype=np.int64),
             rgb=np.concatenate(rgb).astype(np.uint8), body=np.concatenate(body).astype(bool),
             shade=np.concatenate(shade).astype(np.uint8), episode_foreground=np.stack(fg_masks),
             **{f"median_{camera.split('.')[-1]}": medians[camera] for camera in medians})
    (output / "library_report.json").write_text(json.dumps({
        "source": "episode first frames of official clean demonstrations (cam_high), repository decoder, 256x256",
        "body_deviation": BODY_DEVIATION, "soft_deviation": SOFT_DEVIATION, "max_body_area": MAX_BODY_AREA,
        "total_crops": len(meta), "tasks": report}, indent=2) + "\n")
    for camera, median in medians.items():
        Image.fromarray(median.transpose(1, 2, 0)).save(output / f"median_{camera.split('.')[-1]}.png")
    print(f"Saved {len(meta)} crops to {output / 'library.npz'}", flush=True)


if __name__ == "__main__":
    main()
