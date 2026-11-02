import hashlib
import importlib.metadata
import json
import math
import os
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from .scene_augmentation import ScenePlan, SceneLibrary
from .transform import apply_visual_augmentation, sample_visual_augmentation_params


RECIPE_VERSION = "mixaug_v1_abs_seen"
LEGACY_BRANCHES = ("original", "color", "fft", "adain")
SCENE_BRANCHES = ("clutter", "texture", "scene")
BRANCHES = LEGACY_BRANCHES + SCENE_BRANCHES
DEFAULT_WEIGHTS = dict(zip(BRANCHES, (0.40, 0.30, 0.20, 0.10, 0.0, 0.0, 0.0)))
DEFAULT_GUARDS = ("blocks_ranking_rgb", "rotate_qrcode", "scan_object")
_SCENE_LIBRARIES = {}


class AugmentationDataError(RuntimeError):
    pass


@contextmanager
def augmentation_errors(context, stage):
    try:
        yield
    except AugmentationDataError:
        raise
    except Exception as error:
        raise AugmentationDataError(
            f"recipe={RECIPE_VERSION} stage={stage} {context}: {type(error).__name__}: {error}"
        ) from error


def _mapping(value, allowed, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown, key=str)}")
    return value


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _number(value, name, low=0.0, high=1.0):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}], not bool or string")
    return float(value)


def _string(value, name, required=False):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _only(value, expected, name):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} only supports {expected!r}")


@dataclass(frozen=True)
class AugmentationSettings:
    image_enabled: bool = False
    instruction_enabled: bool = False
    seed: int | None = None
    weights: dict = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    guards: tuple = DEFAULT_GUARDS
    style_manifest: str | None = None
    pool_manifest: str | None = None
    beta: float = 0.01
    amplitude_mix: float = 0.20
    alpha: float = 0.20
    cache_root: str | None = None
    max_open_arrays: int = 8
    scene_library: str | None = None
    min_objects: int = 3
    max_objects: int = 8
    texture_guards: tuple = ()

    @property
    def enabled(self):
        return self.image_enabled or self.instruction_enabled

    @property
    def scene_enabled(self):
        return self.image_enabled and any(self.weights[name] > 0 for name in SCENE_BRANCHES)

    @classmethod
    def from_config(cls, mixed=None, instructions=None, legacy=False):
        mixed = _mapping(mixed, (
            "enabled", "seed", "weights", "sync_views_and_time", "teacher_view", "style_manifest",
            "style_guard_tasks", "guarded_style_fallback", "fft", "adain", "scene",
        ), "mixed_augmentation")
        instructions = _mapping(instructions, ("enabled", "split", "pool_manifest", "missing_pool"),
                                "instruction_sampling")
        image_enabled = _bool(mixed.get("enabled", False), "mixed_augmentation.enabled")
        instruction_enabled = _bool(instructions.get("enabled", False), "instruction_sampling.enabled")
        seed = mixed.get("seed")
        if seed is not None or image_enabled or instruction_enabled:
            if type(seed) is not int or not 0 <= seed < 2 ** 63:
                raise ValueError("An explicit integer mixed_augmentation.seed in [0, 2**63) is required")
        if image_enabled and legacy:
            raise ValueError("image_augment and mixed_augmentation cannot both be enabled")
        weights = _mapping(mixed.get("weights", DEFAULT_WEIGHTS), BRANCHES, "weights")
        if not set(LEGACY_BRANCHES) <= set(weights):
            raise ValueError(f"weights must specify {LEGACY_BRANCHES}; {SCENE_BRANCHES} default to 0")
        weights = {key: _number(weights.get(key, 0.0), f"weights.{key}") for key in BRANCHES}
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8, rel_tol=0):
            raise ValueError("Branch weights must sum to 1")
        _only(mixed.get("sync_views_and_time", True), True, "sync_views_and_time")
        _only(mixed.get("teacher_view", "augmented"), "augmented", "teacher_view")
        _only(mixed.get("guarded_style_fallback", "original"), "original", "guarded_style_fallback")
        _only(instructions.get("split", "seen"), "seen", "instruction_sampling.split")
        _only(instructions.get("missing_pool", "error"), "error", "instruction_sampling.missing_pool")
        guards = mixed.get("style_guard_tasks", list(DEFAULT_GUARDS))
        if not isinstance(guards, list) or any(not isinstance(task, str) or not task for task in guards):
            raise ValueError("style_guard_tasks must be a list of task names")
        fft = _mapping(mixed.get("fft"), ("low_freq_ratio", "amplitude_mix"), "fft")
        adain = _mapping(mixed.get("adain"), ("alpha", "cache_root", "cache_miss", "max_open_arrays"), "adain")
        beta = _number(fft.get("low_freq_ratio", 0.01), "fft.low_freq_ratio", high=0.499999)
        amplitude_mix = _number(fft.get("amplitude_mix", 0.20), "fft.amplitude_mix")
        alpha = _number(adain.get("alpha", 0.20), "adain.alpha")
        _only(adain.get("cache_miss", "error"), "error", "adain.cache_miss")
        max_open = adain.get("max_open_arrays", 8)
        if type(max_open) is not int or max_open < 1:
            raise ValueError("adain.max_open_arrays must be a positive integer")
        needs_styles = image_enabled and (
            (weights["fft"] > 0 and beta > 0 and amplitude_mix > 0) or (weights["adain"] > 0 and alpha > 0)
        )
        scene = _mapping(mixed.get("scene"), ("library", "min_objects", "max_objects", "texture_guard_tasks"),
                         "scene")
        needs_scene = image_enabled and any(weights[name] > 0 for name in SCENE_BRANCHES)
        min_objects, max_objects = scene.get("min_objects", 3), scene.get("max_objects", 8)
        if type(min_objects) is not int or type(max_objects) is not int or not 1 <= min_objects <= max_objects <= 20:
            raise ValueError("scene.min_objects/max_objects must be integers with 1 <= min <= max <= 20")
        texture_guards = scene.get("texture_guard_tasks", [])
        if not isinstance(texture_guards, list) or any(not isinstance(task, str) or not task for task in texture_guards):
            raise ValueError("scene.texture_guard_tasks must be a list of task names")
        return cls(
            image_enabled=image_enabled, instruction_enabled=instruction_enabled, seed=seed, weights=weights,
            guards=tuple(guards), beta=beta, amplitude_mix=amplitude_mix, alpha=alpha,
            style_manifest=_string(mixed.get("style_manifest"), "style_manifest", needs_styles),
            pool_manifest=_string(instructions.get("pool_manifest"), "pool_manifest", instruction_enabled),
            cache_root=_string(adain.get("cache_root"), "adain.cache_root",
                               image_enabled and weights["adain"] > 0 and alpha > 0),
            max_open_arrays=max_open,
            scene_library=_string(scene.get("library"), "scene.library", needs_scene),
            min_objects=min_objects, max_objects=max_objects, texture_guards=tuple(texture_guards),
        )

    def validate_tasks(self, training_tasks):
        task_file = Path(__file__).resolve().parents[3] / "assets/training_data/robotwin_clean.txt"
        known = {Path(line.split()[1]).name for line in task_file.read_text().splitlines() if line.strip()}
        unknown = (set(self.guards) | set(self.texture_guards)) - known
        if unknown:
            raise ValueError(f"Unknown style_guard_tasks: {sorted(unknown)}")
        return sorted(set(training_tasks) & set(self.guards))


@dataclass(frozen=True)
class AugmentationIndex:
    index: int
    epoch: int = 0
    position: int = 0
    rank: int = 0


class AugmentationSampler(Sampler):
    def __init__(self, sampler):
        self.sampler = sampler
        self.epoch = int(getattr(sampler, "epoch", 0))
        self.position = 0
        self._resume_position = None

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        self.position = self._resume_position if self._resume_position is not None else 0
        self._resume_position = None
        for index in self.sampler:
            item = AugmentationIndex(int(index), self.epoch, self.position, int(getattr(self.sampler, "rank", 0)))
            self.position += 1
            yield item

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self._resume_position = None
            self.position = 0
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def state_dict(self):
        return {"sampler": self.sampler.state_dict(), "epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        self.set_epoch(state["epoch"])
        self.sampler.load_state_dict(state["sampler"])
        self._resume_position = state["position"]
        self.position = state["position"]


def stream_seed(seed, dataset_uid, source_episode_id, frame, access, stream):
    payload = [RECIPE_VERSION, seed, dataset_uid, source_episode_id, frame,
               access.epoch, access.position, access.rank, stream]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 63)


def check_image(image):
    if not isinstance(image, torch.Tensor) or image.dtype != torch.uint8:
        raise ValueError("Expected uint8 RGB tensor")
    if image.ndim not in (3, 4) or image.shape[-3] != 3 or min(image.shape[-2:]) < 1:
        raise ValueError(f"Expected (C,H,W) or (T,C,H,W), got {tuple(image.shape)}")


def low_frequency_amplitude(image, beta):
    check_image(image)
    height, width = image.shape[-2:]
    radius = math.floor(beta * min(height, width))
    amplitude = torch.fft.fftshift(torch.fft.fft2(image.float(), norm="backward"), dim=(-2, -1)).abs()
    return amplitude[..., height // 2 - radius:height // 2 + radius + 1,
                     width // 2 - radius:width // 2 + radius + 1].contiguous()


def fft_augment(image, reference_amplitude, beta, amplitude_mix, metrics=None):
    check_image(image)
    if beta == 0 or amplitude_mix == 0:
        return image
    height, width = image.shape[-2:]
    radius = math.floor(beta * min(height, width))
    expected = (3, 2 * radius + 1, 2 * radius + 1)
    if tuple(reference_amplitude.shape) != expected or not torch.isfinite(reference_amplitude).all():
        raise ValueError(f"Invalid reference amplitude; expected finite {expected}")
    if (reference_amplitude < 0).any() or not torch.allclose(
        reference_amplitude, reference_amplitude.flip((-2, -1)), rtol=1e-5, atol=1e-3
    ):
        raise ValueError("Reference amplitude is not nonnegative and conjugate symmetric")
    spectrum = torch.fft.fftshift(torch.fft.fft2(image.float(), norm="backward"), dim=(-2, -1))
    amplitude = spectrum.abs()
    low = (..., slice(height // 2 - radius, height // 2 + radius + 1),
           slice(width // 2 - radius, width // 2 + radius + 1))
    amplitude[low] = (1 - amplitude_mix) * amplitude[low] + amplitude_mix * reference_amplitude.to(image.device)
    result = torch.fft.ifft2(
        torch.fft.ifftshift(torch.polar(amplitude, spectrum.angle()), dim=(-2, -1)), norm="backward"
    )
    if not torch.isfinite(result).all() or result.imag.abs().max() > 1e-3:
        raise ValueError("FFT produced nonfinite values or a non-negligible imaginary component")
    real = result.real
    if metrics is not None:
        metrics.update(clip_low=float((real < 0).float().mean()), clip_high=float((real > 255).float().mean()),
                       imaginary_max=float(result.imag.abs().max()))
    return real.round().clamp(0, 255).to(torch.uint8)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def artifact_path(root, relative):
    _string(relative, "artifact path", required=True)
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact escapes its root: {relative}")
    return path


def dataset_identity(root):
    root = Path(root)
    provenance = read_json(root / "meta/robotwin_source.json")
    source = provenance["source"]
    revision = _string(source["revision"], "source.revision", required=True)
    task = root.name
    if source["path"] != f"dataset/{task}/aloha-agilex_clean_50.zip":
        raise ValueError(f"Source archive does not belong to task {task}")
    episode_ids = provenance["source_episode_ids"]
    if len(episode_ids) != provenance["episodes"] or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Invalid source_episode_ids mapping")
    if any(type(episode) is not int or episode < 0 for episode in episode_ids):
        raise ValueError("Source episode IDs must be nonnegative integers")
    return f"{task}@{revision}", provenance


def preprocessing_signature(video_backend, image_size):
    package = {"pyav": "av", "torchcodec": "torchcodec", "video_reader": "torchvision"}.get(video_backend)
    if package is None:
        raise ValueError(f"Unsupported decoder for versioned cache: {video_backend}")
    return {
        "size": list(image_size), "dtype": "uint8", "layout": "CHW", "channel_order": "RGB",
        "resize": "torchvision.v2.Resize", "interpolation": "bilinear", "antialias": True,
        "torch": importlib.metadata.version("torch"), "torchvision": importlib.metadata.version("torchvision"),
        "video_backend": video_backend, "decoder_version": importlib.metadata.version(package),
        "decode_sha256": file_sha256(Path(__file__).with_name("video_utils.py")),
    }


class ArrayCache:
    def __init__(self, root, max_open=8):
        self.root = Path(root)
        self.max_open = max_open
        self._arrays = OrderedDict()
        self._pid = os.getpid()
        self.peak_open = 0

    def close(self):
        for array in self._arrays.values():
            array._mmap.close()
        self._arrays.clear()

    def __getstate__(self):
        return {"root": self.root, "max_open": self.max_open, "_arrays": OrderedDict(), "_pid": None, "peak_open": 0}

    def __del__(self):
        if hasattr(self, "_arrays"):
            self.close()

    def read(self, entry, frames):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        path = artifact_path(self.root, entry["path"])
        stat = path.stat()
        if stat.st_size != entry["file_size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"Cache file changed since validation: {path}")
        if path not in self._arrays:
            if len(self._arrays) >= self.max_open:
                _, old = self._arrays.popitem(last=False)
                old._mmap.close()
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != entry["shape"] or array.dtype != np.uint8:
                array._mmap.close()
                raise ValueError(f"Invalid cache shape/dtype: {path}")
            self._arrays[path] = array
            self.peak_open = max(self.peak_open, len(self._arrays))
        self._arrays.move_to_end(path)
        result = []
        for frame in frames:
            pixels = np.array(self._arrays[path][frame], copy=True, order="C")
            if pixel_sha256(pixels) != entry["frame_sha256"][frame]:
                raise ValueError(f"Cache frame checksum mismatch: {path} frame={frame}")
            result.append(torch.from_numpy(pixels))
        return torch.stack(result)


@dataclass
class SampleAugmentation:
    engine: object
    source_episode_id: int
    anchor_frame_index: int
    episode_length: int
    actual_query_frame_indices: dict
    random_key: int
    branch: str
    requested_branch: str
    style_id: str | None = None
    color_params: dict | None = None
    instruction: str | None = None
    guard_reason: str | None = None
    metrics: dict = field(default_factory=dict)
    scene_plan: ScenePlan | None = None

    def __str__(self):
        return (f"dataset={self.engine.dataset_uid} episode={self.source_episode_id} "
                f"frame={self.anchor_frame_index} branch={self.branch} style={self.style_id} key={self.random_key}")

    def apply_batch(self, batch):
        with augmentation_errors(str(self), "image_dispatch"):
            for group in ("image", "future_image"):
                for camera, image in batch.get(group, {}).items():
                    query = self.actual_query_frame_indices[camera]
                    if group == "image":
                        frames = query[:1] if image.ndim == 3 else query[:image.shape[0]]
                    else:
                        frames = query[-1:] if image.ndim == 3 else query[1:]
                    expected_count = 1 if image.ndim == 3 else image.shape[0]
                    if len(frames) != expected_count or any(not 0 <= frame < self.episode_length for frame in frames):
                        raise ValueError(f"Image/query mismatch for {group}/{camera}: {frames}, {tuple(image.shape)}")
                    metrics = {}
                    batch[group][camera] = self.engine.apply_image(self, camera, image, frames, metrics)
                    self.metrics[f"{group}/{camera}"] = metrics


class DatasetAugmenter:
    def __init__(self, settings, root, camera_sources, episode_lengths, preprocessing):
        self.settings = settings
        self.task = Path(root).name
        self.camera_sources = dict(camera_sources)
        self.preprocessing = preprocessing
        self.styles = {}
        self.amplitudes = {}
        self.entries = {}
        self.pools = {}
        self.cache = None
        self.scene_library = None
        self.scene_candidates = np.zeros(0, dtype=np.int64)
        with augmentation_errors(f"task={self.task} episode=init frame=init", "resource_preflight"):
            self.dataset_uid, self.provenance = dataset_identity(root)
            self.source_ids = self.provenance["source_episode_ids"]
            if len(episode_lengths) != len(self.source_ids):
                raise ValueError("Episode metadata does not match provenance")
            self.episode_lengths = dict(zip(self.source_ids, episode_lengths))
            self.guarded = self.task in settings.validate_tasks([self.task])
            if settings.instruction_enabled:
                manifest = read_json(settings.pool_manifest)
                if manifest["schema_version"] != 1 or manifest["split"] != "seen":
                    raise ValueError("Expected version 1 seen instruction pools")
                pool = manifest["datasets"][self.dataset_uid]
                if pool["source_sha256"] != self.provenance["source"]["sha256"]:
                    raise ValueError("Instruction pool source hash mismatch")
                if pool["source_episode_ids"] != self.source_ids:
                    raise ValueError("Instruction pool episode mapping mismatch")
                for source_id in self.source_ids:
                    texts = pool["episodes"][str(source_id)]
                    if not isinstance(texts, list) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
                        raise ValueError(f"Invalid seen pool for source episode {source_id}")
                    if len(set(texts)) != len(texts):
                        raise ValueError(f"Duplicate instructions in source episode {source_id}")
                    self.pools[source_id] = texts
            needs_fft = settings.image_enabled and settings.weights["fft"] > 0 and settings.beta > 0 and settings.amplitude_mix > 0
            needs_adain = settings.image_enabled and settings.weights["adain"] > 0 and settings.alpha > 0
            if not self.guarded and (needs_fft or needs_adain):
                self._load_styles(needs_fft)
                if needs_adain:
                    self._load_cache()
            if settings.scene_enabled:
                path = str(Path(settings.scene_library).resolve())
                if path not in _SCENE_LIBRARIES:
                    _SCENE_LIBRARIES[path] = SceneLibrary(path)
                self.scene_library = _SCENE_LIBRARIES[path]
                if self.task not in self.scene_library.tasks:
                    raise ValueError(f"Scene library has no first-frame masks for {self.task}")
                if set(self.camera_sources.values()) - {"observation.images.cam_high", "observation.images.cam_left_wrist",
                                                         "observation.images.cam_right_wrist"}:
                    raise ValueError(f"Scene augmentation only supports RoboTwin cameras: {self.camera_sources}")
                self.scene_candidates = self.scene_library.candidates(self.task)
                if self.scene_library.size != tuple(self.preprocessing["size"]):
                    raise ValueError("Scene library resolution does not match image preprocessing")

    def _load_styles(self, needs_fft):
        manifest = read_json(self.settings.style_manifest)
        if manifest["schema_version"] != 1 or manifest["preprocessing"] != self.preprocessing:
            raise ValueError("Style preprocessing/version mismatch")
        if manifest["fft"] != {"norm": "backward", "shift": True, "beta": self.settings.beta}:
            raise ValueError("Reference FFT conventions do not match the recipe")
        root = Path(self.settings.style_manifest).parent
        for style in manifest["styles"]:
            style_id = _string(style["id"], "style.id", required=True)
            if style_id in self.styles or style["split"] != "train" or not style["permission"]:
                raise ValueError("Duplicate style ID, non-training style, or undocumented permission")
            self.styles[style_id] = style
            for camera in self.camera_sources.values():
                reference = style["cameras"][camera]
                if needs_fft:
                    path = artifact_path(root, reference["amplitude_path"])
                    if file_sha256(path) != reference["amplitude_sha256"]:
                        raise ValueError(f"Reference amplitude checksum mismatch: {path}")
                    amplitude = torch.from_numpy(np.load(path, allow_pickle=False))
                    self.amplitudes[style_id, camera] = amplitude
                    height, width = self.preprocessing["size"]
                    fft_augment(torch.zeros(3, height, width, dtype=torch.uint8), amplitude,
                                self.settings.beta, self.settings.amplitude_mix)
        if not self.styles:
            raise ValueError("No training styles available")

    def _load_cache(self):
        index_path = Path(self.settings.cache_root) / "index.json"
        manifest = read_json(index_path)
        if (manifest["schema_version"] != 1 or manifest["recipe_version"] != RECIPE_VERSION
                or manifest["preprocessing"] != self.preprocessing or manifest["alpha"] != self.settings.alpha
                or manifest["style_manifest_sha256"] != file_sha256(self.settings.style_manifest)
                or manifest["complete"] is not True):
            raise ValueError("AdaIN cache version, preprocessing, alpha, styles, or completeness mismatch")
        audit = read_json(Path(self.settings.cache_root) / "audit.json")
        if (audit["passed"] is not True or audit["index_sha256"] != file_sha256(index_path)
                or audit["generation_sha256"] != manifest["generation_sha256"] or audit["inputs_checked"] < 1):
            raise ValueError("AdaIN cache requires a matching successful input/output audit")
        for entry in manifest["entries"]:
            if entry["dataset_uid"] != self.dataset_uid:
                continue
            key = (entry["source_episode_id"], entry["camera"], entry["style_id"])
            if key in self.entries:
                raise ValueError(f"Duplicate cache key: {key}")
            self.entries[key] = entry
        for source_id, length in self.episode_lengths.items():
            for camera in self.camera_sources.values():
                for style_id in self.styles:
                    entry = self.entries[source_id, camera, style_id]
                    expected_shape = [length, 3, *self.preprocessing["size"]]
                    if (entry["shape"] != expected_shape or entry["dtype"] != "uint8"
                            or entry["alpha"] != manifest["alpha"]
                            or entry["generation_sha256"] != manifest["generation_sha256"]
                            or entry["frame_indices"] != list(range(length))
                            or len(entry["input_sha256"]) != length or len(entry["frame_sha256"]) != length):
                        raise ValueError(f"Invalid cache index for {source_id}/{camera}/{style_id}")
                    path = artifact_path(self.settings.cache_root, entry["path"])
                    stat = path.stat()
                    if stat.st_size != entry["file_size"] or stat.st_mtime_ns != entry["mtime_ns"]:
                        raise ValueError(f"Changed/truncated cache: {path}")
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    try:
                        if list(array.shape) != expected_shape or array.dtype != np.uint8:
                            raise ValueError(f"Invalid cache array: {path}")
                    finally:
                        array._mmap.close()
        self.cache = ArrayCache(self.settings.cache_root, self.settings.max_open_arrays)

    def sample(self, raw_item, access):
        episode = int(raw_item["episode_index"])
        frame = int(raw_item["frame_index"])
        context = f"dataset={self.dataset_uid} episode={episode} frame={frame} access={access}"
        with augmentation_errors(context, "sample_context"):
            source_id = self.source_ids[episode]
            metadata = raw_item["_augmentation_meta"]
            length = self.episode_lengths[source_id]
            if metadata["episode_length"] != length or not 0 <= frame < length:
                raise ValueError("Sample episode length/frame does not match source metadata")

            def generator(stream):
                return torch.Generator().manual_seed(stream_seed(
                    self.settings.seed, self.dataset_uid, source_id, frame, access, stream
                ))

            branch = "original"
            if self.settings.image_enabled:
                draw = float(torch.rand((), generator=generator("branch")))
                cumulative = 0.0
                for name in BRANCHES:
                    cumulative += self.settings.weights[name]
                    if draw < cumulative:
                        branch = name
                        break
            requested = branch
            guard_reason = None
            if self.guarded and branch in ("fft", "adain"):
                branch = "original"
                guard_reason = "style_guard"
            if self.task in self.settings.texture_guards and branch in ("texture", "scene"):
                branch = "original" if branch == "texture" else "clutter"
                guard_reason = "texture_guard"
            scene_plan = None
            if branch in SCENE_BRANCHES:
                scene_seed = stream_seed(self.settings.seed, self.dataset_uid, source_id, frame, access, "scene")
                rng = np.random.default_rng(scene_seed)
                clutter = branch in ("clutter", "scene") and len(self.scene_candidates) > 0
                crops = []
                if clutter:
                    count = int(rng.integers(self.settings.min_objects, self.settings.max_objects + 1))
                    crops = rng.choice(self.scene_candidates, size=min(count, len(self.scene_candidates)),
                                       replace=False).tolist()
                scene_plan = ScenePlan(scene_seed, clutter, branch in ("texture", "scene"), crops,
                                       self.scene_library.foreground(self.task, episode) if clutter else None)
            style_id = None
            needs_style = ((branch == "fft" and self.settings.beta > 0 and self.settings.amplitude_mix > 0)
                           or (branch == "adain" and self.settings.alpha > 0))
            if needs_style:
                style_ids = sorted(self.styles)
                style_id = style_ids[int(torch.randint(len(style_ids), (), generator=generator("style")))]
            color_params = None
            if branch == "color":
                reference = raw_item[next(iter(self.camera_sources.values()))]
                color_params = sample_visual_augmentation_params(reference, generator("color"))
            instruction = None
            if self.settings.instruction_enabled:
                texts = self.pools[source_id]
                instruction = texts[int(torch.randint(len(texts), (), generator=generator("instruction")))]
            query = {target: metadata["actual_query_frame_indices"][source]
                     for target, source in self.camera_sources.items()}
            return SampleAugmentation(
                self, source_id, frame, length, query,
                stream_seed(self.settings.seed, self.dataset_uid, source_id, frame, access, "context"),
                branch, requested, style_id, color_params, instruction, guard_reason, scene_plan=scene_plan,
            )

    def apply_image(self, context, camera, image, frames, metrics):
        check_image(image)
        if list(image.shape[-2:]) != self.preprocessing["size"]:
            raise ValueError(f"Image preprocessing mismatch for {camera}")
        branch = context.branch
        if branch == "original" or (branch == "adain" and self.settings.alpha == 0):
            return image
        if branch == "color":
            output = apply_visual_augmentation(image, context.color_params, metrics=metrics)
        elif branch == "fft":
            amplitude = self.amplitudes.get((context.style_id, self.camera_sources[camera]))
            output = fft_augment(image, amplitude, self.settings.beta, self.settings.amplitude_mix, metrics)
        elif branch == "adain":
            entry = self.entries[context.source_episode_id, self.camera_sources[camera], context.style_id]
            output = self.cache.read(entry, frames)
            if image.ndim == 3:
                output = output[0]
        elif branch in SCENE_BRANCHES:
            source = self.camera_sources[camera]
            pixels = image.numpy()
            if image.ndim == 3:
                output = torch.from_numpy(context.scene_plan.apply(pixels, source, self.scene_library))
            else:
                output = torch.from_numpy(np.stack([context.scene_plan.apply(frame, source, self.scene_library)
                                                    for frame in pixels]))
        else:
            raise ValueError(f"Unknown image branch: {branch}")
        check_image(output)
        if output.shape != image.shape:
            raise ValueError(f"Augmentation changed image shape: {tuple(image.shape)} -> {tuple(output.shape)}")
        metrics.update(at_zero=float((output == 0).float().mean()), at_255=float((output == 255).float().mean()))
        return output