"""Clean-only scene augmentation: pasted clean-demo objects (clutter) and procedural table/wall textures.

Only pixels are changed; states, actions and masks are untouched. Object crops come from first frames of
the official clean demonstrations, and textures are generated procedurally (no external images).
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

HEAD_CAMERA = "observation.images.cam_high"
WRIST_CAMERAS = ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
BODY_DEVIATION = 40
SOFT_DEVIATION = 12
BACKGROUND_DEVIATION = 60
TABLE_REFERENCE = 247.0
WALL_REFERENCE = 214.0
_GENERIC_TOKENS = {
    "place", "put", "move", "pick", "stack", "open", "handover", "object", "a2b", "left", "right", "ranking",
    "rgb", "size", "empty", "horizontally", "shake", "adjust", "lift", "grab", "beat", "click", "dump",
    "hanging", "press", "rotate", "scan", "turn", "two", "three", "dual", "diverse", "away",
}


def task_family(task):
    """Object tokens of a task name; crops from tasks sharing a token are never pasted as distractors."""
    tokens = set()
    for token in task.split("_"):
        if token in _GENERIC_TOKENS:
            continue
        tokens.add(token[:-1] if token.endswith("s") and len(token) > 3 else token)
    return tokens


class SceneLibrary:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.tasks = [str(task) for task in data["tasks"]]
            self.meta = data["meta"]
            self.offsets = data["offsets"]
            self.rgb = data["rgb"]
            self.body = data["body"]
            self.shade = data["shade"]
            self.episode_foreground = data["episode_foreground"]
            self.median_head = data["median_cam_high"].astype(np.float32)
        if len(self.offsets) != len(self.meta) + 1 or self.offsets[-1] != len(self.rgb):
            raise ValueError(f"Corrupt scene library index: {self.path}")
        height, width = self.median_head.shape[1:]
        self.size = (height, width)
        brightness = self.median_head.mean(0)
        red, green, blue = self.median_head
        self.head_wall = (red - green > 9) & (red - blue > 9)
        self.head_region = brightness > 150
        self._task_index = {task: index for index, task in enumerate(self.tasks)}

    def candidates(self, task):
        family = task_family(task)
        allowed = [index for index, name in enumerate(self.tasks)
                   if name != task and not (task_family(name) & family)]
        allowed = np.isin(self.meta[:, 0], allowed)
        return np.flatnonzero(allowed)

    def foreground(self, task, episode):
        packed = self.episode_foreground[self._task_index[task], episode]
        return np.unpackbits(packed)[: self.size[0] * self.size[1]].reshape(self.size).astype(bool)

    def crop(self, index):
        _, _, y, x, h, w = (int(value) for value in self.meta[index])
        start, end = self.offsets[index], self.offsets[index + 1]
        rgb = self.rgb[start:end].T.reshape(3, h, w).astype(np.float32)
        body = self.body[start:end].reshape(h, w)
        shade = self.shade[start:end].reshape(h, w).astype(np.float32) / 255.0
        return y, x, rgb, body, shade


def _upsample(grid, height, width):
    tensor = torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)[0, 0].numpy()


def _random_color(rng):
    hue, saturation, value = rng.random(), rng.uniform(0.0, 0.85), rng.uniform(0.25, 1.0)
    index = int(hue * 6) % 6
    f = hue * 6 - int(hue * 6)
    p, q, t = value * (1 - saturation), value * (1 - f * saturation), value * (1 - (1 - f) * saturation)
    return np.array([(value, t, p), (q, value, p), (p, value, t), (p, q, value), (t, p, value), (value, p, q)][index],
                    dtype=np.float32)


def generate_texture(rng, height, width):
    """Procedural RGB texture in [0, 1], shape (3, H, W)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    kind = rng.integers(5)
    if kind == 0:
        k = int(rng.integers(2, 33))
        pattern = _upsample(rng.random((k, k)), height, width)
    elif kind == 1:
        theta, period = rng.uniform(0, math.pi), rng.uniform(6, 64)
        pattern = 0.5 + 0.5 * np.sin(2 * math.pi * (xx * math.cos(theta) + yy * math.sin(theta)) / period)
        if rng.random() < 0.5:
            pattern = (pattern > 0.5).astype(np.float32)
    elif kind == 2:
        theta, cell = rng.uniform(0, math.pi / 2), rng.uniform(8, 64)
        u = (xx * math.cos(theta) + yy * math.sin(theta)) / cell
        v = (-xx * math.sin(theta) + yy * math.cos(theta)) / cell
        pattern = ((np.floor(u) + np.floor(v)) % 2).astype(np.float32)
    elif kind == 3:
        pattern = np.zeros((height, width), np.float32)
        for _ in range(int(rng.integers(3, 16))):
            cy, cx, radius = rng.uniform(0, height), rng.uniform(0, width), rng.uniform(8, 64)
            pattern += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius ** 2))
        pattern = pattern / max(float(pattern.max()), 1e-6)
    else:
        k = int(rng.integers(2, 9))
        noise = _upsample(rng.random((k, k)), height, width)
        theta, period = rng.uniform(0, math.pi), rng.uniform(16, 96)
        pattern = 0.5 + 0.5 * np.sin(2 * math.pi * (xx * math.cos(theta) + yy * math.sin(theta)) / period
                                     + rng.uniform(2, 10) * noise)
    pattern = 0.5 + (pattern - 0.5) * rng.uniform(0.3, 1.0)
    low, high = _random_color(rng), _random_color(rng)
    texture = low[:, None, None] * (1 - pattern) + high[:, None, None] * pattern
    texture = texture + rng.uniform(0, 0.12) * (rng.random((1, height, width), dtype=np.float32) - 0.5)
    return np.clip(texture, 0, 1).astype(np.float32)


def _smooth_components(candidates, certain, gray, min_certain_fraction, require_border=False, min_area=0):
    """Keep candidate regions that are smoothly connected to confidently visible table/wall pixels.

    Edges split candidates so that white robot links and white object parts, which are bounded by edges, are
    not mistaken for table even though their colour matches it; soft shadows stay connected to the table.
    """
    gradient = np.zeros_like(gray)
    gradient[:, 1:] = np.abs(np.diff(gray, axis=1))
    gradient[1:, :] = np.maximum(gradient[1:, :], np.abs(np.diff(gray, axis=0)))
    labels, count = ndimage.label(candidates & (gradient < 10))
    if count == 0:
        return np.zeros_like(candidates)
    index = np.arange(1, count + 1)
    area = ndimage.sum(np.ones_like(gray), labels, index)
    certain_count = ndimage.sum(certain, labels, index)
    keep = (certain_count >= min_certain_fraction * area) & (area >= min_area)
    if require_border:
        border = np.zeros(count + 1, dtype=bool)
        for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
            border[edge] = True
        keep &= border[1:]
    selected = np.concatenate([[False], keep])[labels]
    return candidates & ndimage.binary_dilation(selected, iterations=2)


def head_background(frame, library):
    f = frame.astype(np.float32)
    m = library.median_head
    ratio = f / np.maximum(m, 1.0)
    neutral = ratio.max(0) - ratio.min(0) < 0.10
    deviation = np.abs(f - m).max(0)
    candidates = library.head_region & neutral & (deviation < BACKGROUND_DEVIATION)
    background = _smooth_components(candidates, candidates & (deviation < SOFT_DEVIATION), f.mean(0), 0.2)
    reference = np.where(library.head_wall, WALL_REFERENCE, TABLE_REFERENCE)
    luminance = np.clip(f.mean(0) / reference, 0.25, 1.05)
    return background, luminance, library.head_wall


def wrist_background(frame):
    f = frame.astype(np.float32)
    red, green, blue = f
    brightness = f.mean(0)
    wall = (red - green > 9) & (red - blue > 9) & (brightness > 120)
    neutral = (f.max(0) - f.min(0)) / np.maximum(brightness, 1.0) < 0.12
    candidates = (brightness > 120) & (neutral | wall)
    certain = candidates & (brightness > 225)
    min_area = 0.04 * brightness.size
    background = _smooth_components(candidates, certain, brightness, 0.1, require_border=True, min_area=min_area)
    luminance = np.clip(brightness / np.where(wall, WALL_REFERENCE, TABLE_REFERENCE), 0.25, 1.05)
    return background, luminance, wall


class ScenePlan:
    """Per-sample scene augmentation, replayed identically on current and future frames."""

    def __init__(self, seed, clutter, texture, crops, foreground):
        self.seed = int(seed)
        self.clutter = clutter
        self.texture = texture
        self.crops = list(crops)
        self.foreground = foreground
        self.placements = None
        self._textures = {}

    def _texture(self, camera, part, height, width):
        key = (camera, part)
        if key not in self._textures:
            stream = 1 if part == "table" else 2
            rng = np.random.default_rng([self.seed, stream, 0 if camera == HEAD_CAMERA else 1])
            self._textures[key] = generate_texture(rng, height, width)
        return self._textures[key]

    def _place(self, library, background):
        rng = np.random.default_rng([self.seed, 3])
        height, width = library.size
        blocked = ~background | library.head_wall | ~library.head_region
        if self.foreground is not None:
            blocked |= self.foreground
        occupied = np.zeros_like(blocked)
        placements = []
        for index in self.crops:
            y0, _, _, body, _ = library.crop(index)
            h, w = body.shape
            for _ in range(20):
                y = int(np.clip(y0 + rng.integers(-12, 13), 0, height - h))
                x = int(rng.integers(0, width - w + 1))
                window = (slice(y, y + h), slice(x, x + w))
                if not (body & (blocked[window] | occupied[window])).any():
                    occupied[window] |= body
                    placements.append((index, y, x))
                    break
        return placements

    def apply(self, frame, camera, library):
        """frame: uint8 numpy (3, H, W); returns augmented uint8 copy."""
        height, width = frame.shape[1:]
        if camera == HEAD_CAMERA:
            background, luminance, wall = head_background(frame, library)
        else:
            background, luminance, wall = wrist_background(frame)
        output = frame.astype(np.float32)
        if self.texture:
            table = self._texture(camera, "table", height, width)
            wall_texture = self._texture(camera, "wall", height, width)
            textured = np.where(wall[None], wall_texture, table) * 255.0 * luminance[None]
            output = np.where(background[None], textured, output)
        if self.clutter and camera == HEAD_CAMERA:
            if self.placements is None:
                self.placements = self._place(library, background)
            for index, y, x in self.placements:
                _, _, rgb, body, shade = library.crop(index)
                h, w = body.shape
                window = (slice(y, y + h), slice(x, x + w))
                free = background[window]
                region = output[:, window[0], window[1]]
                paste = body & free
                region[:, paste] = rgb[:, paste] * np.minimum(luminance[window][paste], 1.0)
                shadow = ~body & free & (shade < 0.98)
                region[:, shadow] = region[:, shadow] * shade[shadow]
        return np.clip(output.round(), 0, 255).astype(np.uint8)
