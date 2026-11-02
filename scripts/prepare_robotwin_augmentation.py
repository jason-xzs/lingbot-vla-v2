import argparse
import base64
import hashlib
import importlib.util
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torchvision.transforms.v2 import Resize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingbotvla.data.vla_data.augmentation import (
    ArrayCache,
    AugmentationSettings,
    DEFAULT_GUARDS,
    RECIPE_VERSION,
    artifact_path,
    dataset_identity,
    file_sha256,
    fft_augment,
    low_frequency_amplitude,
    pixel_sha256,
    preprocessing_signature,
    read_json,
)
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
from lingbotvla.data.vla_data.transform import apply_visual_augmentation, sample_visual_augmentation_params


ADAIN_COMMIT = "47950d0e6656a95a80a4b105c4c0f58d38ef785c"
ADAIN_REPO = "https://github.com/naoto0804/pytorch-AdaIN"
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z_0-9]*\}|<(?:object\d*|object_name|arm|color|target|left_or_right)>", re.I)


def write_json_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Refusing to replace a different artifact: {path}; use a new recipe directory")
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def save_array_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if existing.dtype != value.dtype or not np.array_equal(existing, value):
            raise FileExistsError(f"Refusing to replace different pixels: {path}")
        return
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def training_roots(task_file):
    roots = []
    for line in Path(task_file).read_text().splitlines():
        if not line.strip():
            continue
        data_name, root = line.split()
        if data_name != "robotwin" or not Path(root).is_dir():
            raise ValueError(f"Expected an existing robotwin training dataset: {line}")
        roots.append(Path(root))
    if not roots or len({root.name for root in roots}) != len(roots):
        raise ValueError("Training list is empty or contains duplicate tasks")
    return roots


def preparation_settings(args, roots, generation=None):
    guards = list(DEFAULT_GUARDS) if generation is None else generation["style_guard_tasks"]
    settings = AugmentationSettings.from_config({"style_guard_tasks": guards})
    settings.validate_tasks([])
    if args.recipe_config:
        with open(args.recipe_config, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)["data"]
        settings = AugmentationSettings.from_config(data.get("mixed_augmentation"),
                                                    data.get("instruction_sampling"), data.get("image_augment", False))
    protected = settings.validate_tasks([root.name for root in roots])
    if generation is not None and set(settings.guards) != set(guards):
        raise ValueError("Recipe style_guard_tasks differ from the cache generation contract; use a new cache version")
    print(f"Style guard training intersection: {protected}", flush=True)
    return settings


def clean_seen(texts):
    if not isinstance(texts, list):
        raise ValueError("seen must be a list")
    cleaned, seen = [], set()
    counts = {"input": len(texts), "empty": 0, "duplicates": 0, "placeholders": 0}
    for text in texts:
        if not isinstance(text, str):
            raise ValueError("Every instruction must be a UTF-8 string")
        text = text.strip()
        if not text:
            counts["empty"] += 1
        elif PLACEHOLDER.search(text):
            counts["placeholders"] += 1
        elif text in seen:
            counts["duplicates"] += 1
        else:
            seen.add(text)
            cleaned.append(text)
    if not cleaned:
        raise ValueError("Instruction pool is empty after filtering")
    return cleaned, counts


def prepare_instructions(args):
    datasets, reports = {}, {}
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    maximum_tokens = 0
    for root in training_roots(args.tasks):
        dataset_uid, provenance = dataset_identity(root)
        source = provenance["source"]
        archive_path = artifact_path(args.raw_root, source["path"])
        print(f"Verifying {root.name}: {archive_path}", flush=True)
        if archive_path.stat().st_size != source["size"] or file_sha256(archive_path) != source["sha256"]:
            raise ValueError(f"Source archive size/SHA-256 mismatch: {archive_path}")
        pools = {}
        report = {"archive_sha256_verified": source["sha256"], "revision": source["revision"],
                  "episodes": len(provenance["source_episode_ids"]), "input": 0, "unique": 0,
                  "empty": 0, "duplicates": 0, "placeholders": 0}
        with zipfile.ZipFile(archive_path) as archive:
            for local_episode, source_id in enumerate(provenance["source_episode_ids"]):
                member = f"aloha-agilex_clean_50/instructions/episode{source_id}.json"
                instructions = json.loads(archive.read(member).decode("utf-8"))
                pool, counts = clean_seen(instructions["seen"])
                if provenance["instructions"][local_episode].strip() not in pool:
                    raise ValueError(f"Converted instruction not found in matching seen pool: {root.name}/{source_id}")
                pools[str(source_id)] = pool
                for key, count in counts.items():
                    report[key] += count
                report["unique"] += len(pool)
                if tokenizer is not None:
                    for text in pool:
                        if args.chat_template == "qwen":
                            prompt = tokenizer.apply_chat_template(
                                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False
                            )
                        else:
                            prompt = (text if text.startswith("<bos>") else f"<bos>{text}")
                            prompt = prompt if prompt.endswith("\n") else prompt + "\n"
                        maximum_tokens = max(maximum_tokens, len(tokenizer(prompt)["input_ids"]))
        datasets[dataset_uid] = {"source_sha256": source["sha256"], "source_episode_ids": provenance["source_episode_ids"],
                                 "source": source, "episodes": pools}
        reports[root.name] = report
        print(f"{root.name}: {report['episodes']} episodes, {report['unique']} unique episode-local instructions", flush=True)
    if tokenizer is not None and maximum_tokens > args.max_tokens:
        raise ValueError(f"Instructions require {maximum_tokens} tokens, exceeding {args.max_tokens}")
    output = Path(args.output)
    write_json_once(output / "seen_pools.json", {"schema_version": 1, "recipe_version": RECIPE_VERSION,
                                                "split": "seen", "datasets": datasets})
    write_json_once(output / "audit/instructions.json", {
        "tasks": reports, "tokenizer": args.tokenizer, "chat_template": args.chat_template,
        "maximum_tokens": maximum_tokens if tokenizer is not None else None,
        "tokenizer_max_length": args.max_tokens, "cleaning": "strip outer whitespace, exact episode-local dedup, confirmed slots",
        "preserved_literal": "<top center>",
    })
    print(f"Instruction pools saved; maximum token count: {maximum_tokens}", flush=True)


def open_source(root, size=256):
    return LeRobotDataset(str(root), image_transforms=Resize((size, size)), export_query_indices=True)


def read_source_frames(dataset, episode, camera, frames):
    metadata = dataset.meta.episodes[episode]
    start = int(metadata["dataset_from_index"])
    length = int(metadata["length"])
    if not frames or any(type(frame) is not int or not 0 <= frame < length for frame in frames):
        raise ValueError(f"Invalid source frame request: episode={episode}, frames={frames}")
    timestamps = [float(value) for value in dataset.hf_dataset[[start + frame for frame in frames]]["timestamp"]]
    images = dataset._query_videos({camera: timestamps}, episode)[camera]
    if images.ndim == 3:
        images = images.unsqueeze(0)
    images = dataset.image_transforms(images)
    if images.dtype != torch.uint8 or images.shape[0] != len(frames):
        raise ValueError("Offline input must use the repository uint8 decoder/resize contract")
    return images.contiguous()


def prepare_styles(args):
    output = Path(args.output)
    pools = read_json(output / "seen_pools.json")
    roots = training_roots(args.tasks)
    settings = preparation_settings(args, roots)
    roots = [root for root in roots if root.name not in settings.guards]
    if not 1 <= args.num_styles <= len(roots):
        raise ValueError("num_styles must be between 1 and the number of unguarded training tasks")
    styles, preprocessing = [], None
    for index, root in enumerate(roots[:args.num_styles]):
        dataset_uid, provenance = dataset_identity(root)
        if dataset_uid not in pools["datasets"]:
            raise ValueError("Style reference is not in the verified training instruction manifest")
        dataset = open_source(root, args.image_size)
        signature = preprocessing_signature(dataset.video_backend, (args.image_size, args.image_size))
        if preprocessing is not None and signature != preprocessing:
            raise ValueError("Inconsistent preprocessing across style datasets")
        preprocessing = signature
        episode = 0
        frame = int(dataset.meta.episodes[episode]["length"]) // 2
        style_id = f"clean_{index:03d}"
        cameras = {}
        for camera in dataset.meta.camera_keys:
            pixels = read_source_frames(dataset, episode, camera, [frame])[0]
            relative = Path("styles") / style_id / camera
            image_path = output / (str(relative) + ".npy")
            amplitude_path = output / (str(relative) + ".amplitude.npy")
            save_array_once(image_path, pixels.numpy())
            save_array_once(amplitude_path, low_frequency_amplitude(pixels, args.beta).numpy())
            preview = output / (str(relative) + ".png")
            if not preview.exists():
                Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(preview)
            cameras[camera] = {"image_path": str(image_path.relative_to(output)), "image_sha256": file_sha256(image_path),
                               "input_sha256": pixel_sha256(pixels.numpy()),
                               "amplitude_path": str(amplitude_path.relative_to(output)),
                               "amplitude_sha256": file_sha256(amplitude_path),
                               "channel_means": pixels.float().mean((1, 2)).tolist()}
        styles.append({"id": style_id, "split": "train", "dataset_uid": dataset_uid,
                       "source_episode_id": provenance["source_episode_ids"][episode], "frame_index": frame,
                       "source_archive_sha256": provenance["source"]["sha256"],
                       "permission": "User-authorized clean training references only; original dataset terms apply",
                       "cameras": cameras})
        print(f"Prepared style {style_id}: {root.name}, source episode 0, frame {frame}", flush=True)
    write_json_once(output / "style_manifest.json", {
        "schema_version": 1, "recipe_version": RECIPE_VERSION, "preprocessing": preprocessing,
        "fft": {"norm": "backward", "shift": True, "beta": args.beta}, "styles": styles,
    })


def download_release_asset(asset, path, headers):
    url = asset["url"] + f"?download=1&request={time.time_ns()}"
    request = urllib.request.Request(url, headers={**headers, "Accept": "application/octet-stream", "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        download_url = response.geturl()
    chunk_size = 256 * 1024
    ranges = [(start, min(start + chunk_size, asset["size"]) - 1) for start in range(0, asset["size"], chunk_size)]

    def read_range(bounds):
        start, end = bounds
        request = urllib.request.Request(download_url, headers={**headers, "Range": f"bytes={start}-{end}"})
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    expected = f"bytes {start}-{end}/{asset['size']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Official asset returned an unexpected range for {asset['name']}: {start}-{end}")
                    payload = response.read(end - start + 2)
                if len(payload) != end - start + 1:
                    raise ValueError(f"Truncated release asset range: {start}-{end}")
                return start, payload
            except (OSError, TimeoutError):
                if attempt == 1:
                    raise

    chunks = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read_range, bounds) for bounds in ranges]
        for future in as_completed(futures):
            start, payload = future.result()
            chunks[start] = payload
            if len(chunks) % 16 == 0 or len(chunks) == len(ranges):
                print(f"{asset['name']}: {len(chunks)}/{len(ranges)} verified ranges", flush=True)
    temporary = path.with_suffix(path.suffix + ".assembled")
    with temporary.open("wb") as target:
        for start, _ in ranges:
            target.write(chunks[start])
    if temporary.stat().st_size != asset["size"]:
        raise ValueError(f"Release asset size mismatch: {temporary}")
    previous = path.with_suffix(path.suffix + ".download")
    if previous.exists():
        prefix = previous.read_bytes()
        with temporary.open("rb") as complete:
            if complete.read(len(prefix)) != prefix:
                raise ValueError("Ranged download does not match the earlier official asset prefix")
    temporary.rename(path)


def download_models(args):
    root = Path(args.output) / "models"
    api = "https://api.github.com/repos/naoto0804/pytorch-AdaIN"
    headers = {"User-Agent": "robotwin-mixaug/1", "X-GitHub-Api-Version": "2022-11-28"}
    with urllib.request.urlopen(urllib.request.Request(f"{api}/releases/tags/v0.0.0", headers=headers), timeout=60) as response:
        release = json.load(response)
    assets = {asset["name"]: asset for asset in release["assets"]}
    root.mkdir(parents=True, exist_ok=True)
    existing = read_json(root / "manifest.json") if (root / "manifest.json").exists() else None
    files = {}
    for name in ("net.py", "function.py", "LICENSE", "decoder.pth", "vgg_normalised.pth"):
        path = root / name
        if name.endswith(".pth"):
            asset = assets[name]
            url = asset["browser_download_url"]
            if not path.exists():
                print(f"Downloading {url} via the official release API", flush=True)
                download_release_asset(asset, path, headers)
            if path.stat().st_size != asset["size"]:
                raise ValueError(f"Existing release asset size mismatch: {path}")
        else:
            url = f"{api}/contents/{name}?ref={ADAIN_COMMIT}"
            print(f"Downloading {url}", flush=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                content = json.load(response)
            payload = base64.b64decode(content["content"], validate=False)
            blob = f"blob {len(payload)}\0".encode() + payload
            if hashlib.sha1(blob).hexdigest() != content["sha"] or len(payload) != content["size"]:
                raise ValueError(f"GitHub content blob mismatch: {name}")
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError(f"Existing upstream source differs from pinned commit: {path}")
            else:
                with path.open("xb") as target:
                    target.write(payload)
        checksum = file_sha256(path)
        if existing is not None and existing["files"][name]["sha256"] != checksum:
            raise ValueError(f"Existing AdaIN model/code checksum mismatch: {path}")
        files[name] = {"url": url, "sha256": checksum, "bytes": path.stat().st_size}
    manifest = {"implementation": ADAIN_REPO, "commit": ADAIN_COMMIT, "release": "v0.0.0",
                "license_file": "LICENSE", "files": files}
    write_json_once(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


class AdaINModel:
    def __init__(self, root, device):
        root = Path(root)
        manifest = read_json(root / "manifest.json")
        if manifest["commit"] != ADAIN_COMMIT:
            raise ValueError("AdaIN implementation must match the pinned commit")
        for name, info in manifest["files"].items():
            if file_sha256(root / name) != info["sha256"]:
                raise ValueError(f"AdaIN code/weight checksum mismatch: {name}")
        function_spec = importlib.util.spec_from_file_location("robotwin_adain_function", root / "function.py")
        function = importlib.util.module_from_spec(function_spec)
        function_spec.loader.exec_module(function)
        net_spec = importlib.util.spec_from_file_location("robotwin_adain_net", root / "net.py")
        net = importlib.util.module_from_spec(net_spec)
        previous = sys.modules.get("function")
        sys.modules["function"] = function
        try:
            net_spec.loader.exec_module(net)
        finally:
            if previous is None:
                sys.modules.pop("function", None)
            else:
                sys.modules["function"] = previous
        net.vgg.load_state_dict(torch.load(root / "vgg_normalised.pth", map_location="cpu", weights_only=True))
        net.decoder.load_state_dict(torch.load(root / "decoder.pth", map_location="cpu", weights_only=True))
        self.encoder = torch.nn.Sequential(*list(net.vgg.children())[:31]).eval().requires_grad_(False).to(device)
        self.decoder = net.decoder.eval().requires_grad_(False).to(device)
        self.adain = function.adaptive_instance_normalization
        self.device = device

    @torch.inference_mode()
    def encode(self, images):
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("AdaIN content must be uint8 NCHW RGB")
        return self.encoder(images.to(device=self.device, dtype=torch.float32) / 255.0)

    @torch.inference_mode()
    def decode(self, content_features, style_features, alpha, expected_shape):
        style_features = style_features.expand(content_features.shape[0], -1, -1, -1)
        transferred = self.adain(content_features, style_features)
        result = self.decoder(alpha * transferred + (1 - alpha) * content_features)
        if tuple(result.shape) != tuple(expected_shape) or not torch.isfinite(result).all():
            raise ValueError("AdaIN returned an invalid shape or nonfinite pixels")
        metrics = {"clip_low_pixels": int((result < 0).sum()), "clip_high_pixels": int((result > 1).sum()),
                   "pixels": result.numel(), "nonfinite": 0}
        pixels = (result * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return pixels, metrics


def cache_worker(gpu, jobs, options):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = Path(options["output"])
    cache_root = output / options["cache_name"]
    styles = read_json(output / "style_manifest.json")
    model = AdaINModel(output / "models", f"cuda:{gpu}")
    style_features = {}
    for style in styles["styles"]:
        for camera, reference in style["cameras"].items():
            path = artifact_path(output, reference["image_path"])
            if file_sha256(path) != reference["image_sha256"]:
                raise ValueError(f"Style reference checksum mismatch: {path}")
            pixels = torch.from_numpy(np.load(path, allow_pickle=False)).unsqueeze(0)
            style_features[style["id"], camera] = model.encode(pixels)
    dataset, current_root, entries = None, None, []
    for root_string, episode in jobs:
        root = Path(root_string)
        if root != current_root:
            dataset = open_source(root, options["image_size"])
            current_root = root
            if preprocessing_signature(dataset.video_backend, (options["image_size"],) * 2) != styles["preprocessing"]:
                raise ValueError("Cache decoder/preprocessing differs from reference styles")
        dataset_uid, provenance = dataset_identity(root)
        source_id = provenance["source_episode_ids"][episode]
        length = int(dataset.meta.episodes[episode]["length"])
        for camera in dataset.meta.camera_keys:
            pending = []
            for style in styles["styles"]:
                style_id = style["id"]
                relative = Path(root.name) / f"episode_{source_id:06d}" / f"{camera}.{style_id}.npy"
                path = cache_root / relative
                sidecar = Path(str(path) + ".json")
                if sidecar.exists():
                    entry = read_json(sidecar)
                    if (entry["generation_sha256"] != options["generation_sha256"]
                            or entry["sha256"] != file_sha256(path)):
                        raise ValueError(f"Cannot resume changed cache group: {path}")
                    entries.append(entry)
                    continue
                if path.exists():
                    raise FileExistsError(f"Unindexed final cache group needs explicit inspection: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(str(path) + ".partial")
                shape = (length, 3, options["image_size"], options["image_size"])
                array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
                pending.append({"style_id": style_id, "path": path, "temporary": temporary, "sidecar": sidecar,
                                "relative": relative, "array": array, "input_sha256": [], "frame_sha256": [],
                                "metrics": {"clip_low_pixels": 0, "clip_high_pixels": 0, "pixels": 0, "nonfinite": 0}})
            if not pending:
                continue
            try:
                for decode_start in range(0, length, options["decode_frames"]):
                    frame_ids = list(range(decode_start, min(length, decode_start + options["decode_frames"])))
                    content = read_source_frames(dataset, episode, camera, frame_ids)
                    content_hashes = [pixel_sha256(frame.numpy()) for frame in content]
                    for group in pending:
                        group["input_sha256"].extend(content_hashes)
                    for batch_start in range(0, len(content), options["batch_size"]):
                        batch = content[batch_start:batch_start + options["batch_size"]]
                        features = model.encode(batch)
                        start = decode_start + batch_start
                        for group in pending:
                            pixels, metrics = model.decode(features, style_features[group["style_id"], camera],
                                                           options["alpha"], batch.shape)
                            group["array"][start:start + len(batch)] = pixels
                            group["frame_sha256"].extend(pixel_sha256(frame) for frame in pixels)
                            for key, value in metrics.items():
                                group["metrics"][key] += value
                for group in pending:
                    group["array"].flush()
                    group["array"]._mmap.close()
                    group["temporary"].rename(group["path"])
                    stat = group["path"].stat()
                    entry = {"dataset_uid": dataset_uid, "task": root.name, "source_episode_id": source_id,
                             "camera": camera, "style_id": group["style_id"], "alpha": options["alpha"],
                             "generation_sha256": options["generation_sha256"], "path": str(group["relative"]),
                             "shape": list(shape), "dtype": "uint8", "frame_indices": list(range(length)),
                             "input_sha256": group["input_sha256"], "frame_sha256": group["frame_sha256"],
                             "sha256": file_sha256(group["path"]), "file_size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                             "metrics": group["metrics"]}
                    write_json_once(group["sidecar"], entry)
                    entries.append(entry)
            finally:
                for group in pending:
                    if not group["array"]._mmap.closed:
                        group["array"]._mmap.close()
        print(f"GPU {gpu}: {root.name} episode {source_id} ({length} frames) complete", flush=True)
    return entries, {"gpu": gpu, "name": torch.cuda.get_device_name(gpu), "episodes": len(jobs),
                     "seconds": time.perf_counter() - started, "gpu_peak_bytes": torch.cuda.max_memory_allocated(gpu)}


def prepare_cache(args):
    if not 0 < args.alpha <= 1 or args.image_size % 8 != 0:
        raise ValueError("Cached AdaIN needs alpha in (0,1] and an image size divisible by 8; alpha=0 needs no cache")
    if args.batch_size < 1 or args.decode_frames < 1 or args.limit_episodes is not None and args.limit_episodes < 1:
        raise ValueError("Batch sizes and optional episode limit must be positive")
    output = Path(args.output)
    cache_root = artifact_path(output, args.cache_name)
    cache_root.mkdir(parents=True, exist_ok=True)
    styles = read_json(output / "style_manifest.json")
    models = read_json(output / "models/manifest.json")
    roots = training_roots(args.tasks)
    settings = preparation_settings(args, roots)
    jobs, sources, pixel_bytes = [], {}, 0
    for root in roots:
        dataset_uid, provenance = dataset_identity(root)
        sources[dataset_uid] = provenance["source"]
        if root.name in settings.guards:
            continue
        episode_count = provenance["episodes"]
        if args.limit_episodes is not None:
            episode_count = min(episode_count, args.limit_episodes)
        jobs.extend((str(root), episode) for episode in range(episode_count))
        pixel_bytes += provenance["frames"] * 3 * 3 * args.image_size ** 2 * len(styles["styles"])
    if args.limit_episodes is not None and args.cache_name == "adain_cache":
        raise ValueError("Smoke generation must use a separate --cache-name, not the training cache")
    if shutil.disk_usage(cache_root).free < pixel_bytes * 1.15:
        raise ValueError(f"Insufficient cache space; allow at least {pixel_bytes * 1.15 / 1e9:.2f} GB")
    gpus = args.gpus
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in gpus):
        raise ValueError("Specify unique available CUDA device IDs")
    for gpu in gpus:
        free, total = torch.cuda.mem_get_info(gpu)
        if total - free > 1024 ** 3:
            raise RuntimeError(f"GPU {gpu} is in use; refusing to allocate an augmentation worker")
    contract = {"schema_version": 1, "recipe_version": RECIPE_VERSION, "preprocessing": styles["preprocessing"],
                "generator_sha256": file_sha256(__file__),
                "alpha": args.alpha, "style_manifest_sha256": file_sha256(output / "style_manifest.json"),
                "model_manifest_sha256": file_sha256(output / "models/manifest.json"), "sources": sources,
                "style_guard_tasks": list(settings.guards), "precision": "float32", "tf32": False,
                "cudnn_benchmark": False, "cudnn_deterministic": True, "seed": 42,
                "batch_size": args.batch_size, "decode_frames": args.decode_frames, "limit_episodes": args.limit_episodes}
    write_json_once(cache_root / "generation.json", contract)
    generation_hash = file_sha256(cache_root / "generation.json")
    if (cache_root / "index.json").exists():
        index = read_json(cache_root / "index.json")
        if index["generation_sha256"] != generation_hash:
            raise ValueError("Existing cache index uses a different generation contract")
        print(f"Cache already finalized: {cache_root}; use the audit command to verify it", flush=True)
        return
    options = {**vars(args), "generation_sha256": generation_hash}
    entries, workers = [], []
    started = time.perf_counter()
    print(f"Preparing {len(jobs)} episodes on GPUs {gpus}; full-range pixels {pixel_bytes / 1e9:.2f} GB", flush=True)
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(cache_worker, gpu, jobs[index::len(gpus)], options)
                   for index, gpu in enumerate(gpus) if jobs[index::len(gpus)]]
        for future in as_completed(futures):
            worker_entries, report = future.result()
            entries.extend(worker_entries)
            workers.append(report)
    entries.sort(key=lambda entry: (entry["dataset_uid"], entry["source_episode_id"], entry["camera"], entry["style_id"]))
    expected = len(jobs) * 3 * len(styles["styles"])
    if len(entries) != expected:
        raise ValueError(f"Incomplete cache: expected {expected} groups, got {len(entries)}")
    write_json_once(cache_root / "index.json", {
        "schema_version": 1, "recipe_version": RECIPE_VERSION, "generation_sha256": generation_hash,
        "complete": args.limit_episodes is None, "preprocessing": styles["preprocessing"], "alpha": args.alpha,
        "style_manifest_sha256": contract["style_manifest_sha256"], "model_manifest_sha256": contract["model_manifest_sha256"],
        "model_weights": models["files"], "entries": entries,
    })
    report = {"groups": len(entries), "file_bytes": sum(entry["file_size"] for entry in entries),
              "seconds": time.perf_counter() - started, "workers": workers, "complete": args.limit_episodes is None}
    write_json_once(output / "audit" / f"{args.cache_name}_generation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def save_comparison(path, columns, labels, title):
    size = columns[0].shape[-1]
    views = columns[0].shape[0]
    canvas = Image.new("RGB", (len(columns) * size, views * (size + 20) + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill="black")
    for column, (images, label) in enumerate(zip(columns, labels)):
        for camera, image in enumerate(images):
            offset = 28 + camera * (size + 20)
            draw.text((column * size + 4, offset + 2), f"{label} / view {camera}", fill="black")
            canvas.paste(Image.fromarray(image.permute(1, 2, 0).numpy()), (column * size, offset + 20))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        canvas.save(path)


def verify_cache_groups(cache_root, entries):
    keys = set()
    for entry in entries:
        key = (entry["dataset_uid"], entry["source_episode_id"], entry["camera"], entry["style_id"])
        if key in keys:
            raise ValueError(f"Duplicate cache group: {key}")
        keys.add(key)
        path = artifact_path(cache_root, entry["path"])
        stat = path.stat()
        if stat.st_size != entry["file_size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"Cache group changed: {path}")
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Cache group checksum mismatch: {path}")
    return len(keys)


def audit_cache(args):
    output = Path(args.output)
    cache_root = artifact_path(output, args.cache_name)
    index_path = cache_root / "index.json"
    index = read_json(index_path)
    generation_path = cache_root / "generation.json"
    if file_sha256(generation_path) != index["generation_sha256"]:
        raise ValueError("Cache generation contract checksum mismatch")
    generation = read_json(generation_path)
    roots = training_roots(args.tasks)
    settings = preparation_settings(args, roots, generation)
    styles = read_json(output / "style_manifest.json")
    recipe = settings if args.recipe_config else None
    recipe_hash = None
    if args.recipe_config:
        recipe_hash = file_sha256(args.recipe_config)
        if (recipe.alpha != index["alpha"] or recipe.beta != args.beta
                or Path(recipe.style_manifest).resolve() != (output / "style_manifest.json").resolve()):
            raise ValueError("Audit recipe differs from cache/style parameters")
    color_enabled = recipe is None or recipe.weights["color"] > 0
    color_label = "color" if color_enabled else "color disabled (original)"
    group_checksums = verify_cache_groups(cache_root, index["entries"]) if args.full_checksums else 0
    if index["style_manifest_sha256"] != file_sha256(output / "style_manifest.json"):
        raise ValueError("Style manifest changed after cache generation")
    entries = {(entry["dataset_uid"], entry["source_episode_id"], entry["camera"], entry["style_id"]): entry
               for entry in index["entries"]}
    cache = ArrayCache(cache_root, max_open=2)
    records, inputs_checked, pixels_checked = [], 0, 0
    audit_root = output / "audit" / args.cache_name
    for root in roots:
        dataset_uid, provenance = dataset_identity(root)
        dataset = open_source(root, args.image_size)
        signature = preprocessing_signature(dataset.video_backend, (args.image_size, args.image_size))
        if signature != index["preprocessing"]:
            raise ValueError("Audit preprocessing differs from the cache")
        guarded = root.name in settings.guards
        available = sorted({entry["source_episode_id"] for entry in index["entries"] if entry["dataset_uid"] == dataset_uid})
        if guarded:
            available = provenance["source_episode_ids"]
        if not available:
            raise ValueError(f"No cached episodes for unguarded training task {root.name}")
        chosen = sorted({available[0], available[len(available) // 2], available[-1]})
        if args.audit_episodes == 1:
            chosen = chosen[:1]
        for source_id in chosen:
            episode = provenance["source_episode_ids"].index(source_id)
            length = int(dataset.meta.episodes[episode]["length"])
            temporal = list(range(max(0, length // 2 - 4), min(length, length // 2 + 4)))
            frames = sorted(set([0, length // 2, max(0, length - 2), length - 1] + temporal))
            raw_images = {camera: read_source_frames(dataset, episode, camera, frames) for camera in dataset.meta.camera_keys}
            reference = next(iter(raw_images.values()))
            params = sample_visual_augmentation_params(reference, torch.Generator().manual_seed(42 + source_id))
            color_images, color_metrics = {}, {}
            for camera, raw in raw_images.items():
                metrics = {}
                color_images[camera] = apply_visual_augmentation(raw, params, metrics=metrics) if color_enabled else raw
                color_metrics[camera] = metrics
            for style in styles["styles"]:
                style_id = style["id"]
                adain_images, fft_images, fft_metrics = {}, {}, {}
                for camera, raw in raw_images.items():
                    style_reference = style["cameras"][camera]
                    amplitude = torch.from_numpy(np.load(artifact_path(output, style_reference["amplitude_path"]), allow_pickle=False))
                    style_pixels = torch.from_numpy(np.load(artifact_path(output, style_reference["image_path"]), allow_pickle=False))
                    torch.testing.assert_close(amplitude, low_frequency_amplitude(style_pixels, args.beta), rtol=0, atol=0)
                    metrics = {}
                    mix = recipe.amplitude_mix if recipe is not None else 0.2
                    fft_images[camera] = raw if guarded else fft_augment(raw, amplitude, args.beta, mix, metrics)
                    fft_metrics[camera] = metrics
                    if guarded:
                        adain_images[camera] = raw
                        continue
                    entry = entries[dataset_uid, source_id, camera, style_id]
                    pixels = cache.read(entry, frames)
                    for position, frame in enumerate(frames):
                        if pixel_sha256(raw[position].numpy()) != entry["input_sha256"][frame]:
                            raise ValueError(f"Online input hash mismatch: {dataset_uid}/{source_id}/{camera}/{frame}")
                        inputs_checked += 1
                        pixels_checked += pixels[position].numel()
                    adain_images[camera] = pixels
                for frame in (0, length // 2, length - 1):
                    position = frames.index(frame)
                    columns = [torch.stack([images[camera][position] for camera in dataset.meta.camera_keys])
                               for images in (raw_images, color_images, fft_images, adain_images)]
                    label = " (style guarded)" if guarded else ""
                    save_comparison(audit_root / f"{root.name}_ep{source_id:03d}_{style_id}_f{frame:04d}.png", columns,
                                    ["original", color_label, "FFT", "AdaIN"], f"{root.name} episode {source_id} frame {frame}{label}")
                for camera in dataset.meta.camera_keys:
                    positions = [frames.index(frame) for frame in temporal]
                    paired = [torch.stack([raw_images[camera][position],
                                          color_images[camera][position] if guarded else adain_images[camera][position]])
                              for position in positions]
                    save_comparison(audit_root / f"{root.name}_ep{source_id:03d}_{style_id}_{camera}_sequence.png", paired,
                                    [f"frame {frame}" for frame in temporal],
                                    f"{root.name}: original above, {color_label if guarded else 'AdaIN'} below; {camera}")
                records.append({"task": root.name, "source_episode_id": source_id, "style_id": style_id,
                                "frames": frames, "guarded": guarded, "color_metrics": color_metrics, "fft_metrics": fft_metrics})
        print(f"Audited {root.name}: source episodes {chosen}, guarded={guarded}", flush=True)
    cache.close()
    report = {"passed": True, "index_sha256": file_sha256(index_path), "generation_sha256": index["generation_sha256"],
              "inputs_checked": inputs_checked, "output_pixels_checked": pixels_checked, "peak_open_arrays": cache.peak_open,
              "full_group_checksums": args.full_checksums, "checks": "sampled input and output hashes, cached FFT equivalence",
              "groups_verified": group_checksums, "recipe_config_sha256": recipe_hash,
              "branch_weights": recipe.weights if recipe is not None else None,
              "visual_review_required": True, "records": records}
    write_json_once(cache_root / "audit.json", report)
    print(f"PASS: {inputs_checked} online input hashes, {pixels_checked} output pixels; comparisons: {audit_root}", flush=True)


def make_parser():
    parser = argparse.ArgumentParser(description="Prepare versioned RoboTwin mixed augmentation without changing source data")
    parser.add_argument("stage", choices=("instructions", "styles", "models", "cache", "audit"))
    parser.add_argument("--tasks", default="assets/training_data/robotwin_clean.txt")
    parser.add_argument("--output", default="assets/augmentation/full50_v1")
    parser.add_argument("--raw-root", default="data/raw_clean")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--chat-template", choices=("qwen", "bos"), default="qwen")
    parser.add_argument("--max-tokens", type=int, default=72)
    parser.add_argument("--num-styles", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-frames", type=int, default=128)
    parser.add_argument("--cache-name", default="adain_cache")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--audit-episodes", type=int, choices=(1, 3), default=3)
    parser.add_argument("--full-checksums", action="store_true")
    parser.add_argument("--recipe-config", help="Training recipe defining style guards for styles/cache/audit and active audit branches")
    return parser


def main():
    args = make_parser().parse_args()
    torch.set_num_threads(2)
    {"instructions": prepare_instructions, "styles": prepare_styles,
        "models": download_models, "cache": prepare_cache, "audit": audit_cache}[args.stage](args)


if __name__ == "__main__":
    main()
