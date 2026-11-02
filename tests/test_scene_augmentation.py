import copy
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from lingbotvla.data.dataset import build_vla_dataset
from lingbotvla.data.vla_data.augmentation import AugmentationIndex, AugmentationSettings
from lingbotvla.data.vla_data.scene_augmentation import (
    HEAD_CAMERA,
    WRIST_CAMERAS,
    ScenePlan,
    SceneLibrary,
    generate_texture,
    task_family,
)

RECIPE = "configs/vla/robotwin/robotwin_clean_50task_scene_resume50k_a800_6gpu.yaml"
LIBRARY = "assets/augmentation/clutter_v1/library.npz"


def synthetic_library(directory):
    median = np.full((3, 64, 64), 247, np.uint8)
    median[:, :6] = np.array([224, 208, 208], np.uint8)[:, None, None]
    rgb = np.zeros((3, 4, 4), np.uint8)
    rgb[0] = 200
    body = np.zeros((4, 4), bool)
    body[1:3, 1:3] = True
    shade = np.full((4, 4), 255, np.uint8)
    shade[3] = 150
    foreground = np.zeros((2, 2, 64 * 64 // 8), np.uint8)
    path = Path(directory) / "library.npz"
    np.savez(path, tasks=np.array(["handover_mic", "place_shoe"]), meta=np.array([[1, 0, 30, 30, 4, 4]], np.int32),
             offsets=np.array([0, 16]), rgb=rgb.reshape(3, -1).T, body=body.reshape(-1), shade=shade.reshape(-1),
             episode_foreground=foreground, median_cam_high=median, median_cam_left_wrist=median,
             median_cam_right_wrist=median)
    return path


class SceneSettingsTests(unittest.TestCase):
    def test_legacy_weights_remain_valid_and_scene_requires_library(self):
        legacy = AugmentationSettings.from_config(
            {"enabled": True, "seed": 1, "weights": {"original": 1.0, "color": 0.0, "fft": 0.0, "adain": 0.0}})
        self.assertFalse(legacy.scene_enabled)
        weights = {"original": 0.5, "color": 0.0, "fft": 0.0, "adain": 0.0, "clutter": 0.5}
        with self.assertRaisesRegex(ValueError, "scene.library"):
            AugmentationSettings.from_config({"enabled": True, "seed": 1, "weights": weights})
        settings = AugmentationSettings.from_config(
            {"enabled": True, "seed": 1, "weights": weights, "scene": {"library": "x.npz", "max_objects": 5}})
        self.assertTrue(settings.scene_enabled)
        self.assertEqual((settings.min_objects, settings.max_objects), (3, 5))
        for scene in ({"library": "x", "min_objects": 6, "max_objects": 5}, {"library": "x", "min_objects": 1.0},
                      {"library": "x", "unknown": 1}):
            with self.assertRaises(ValueError):
                AugmentationSettings.from_config({"enabled": True, "seed": 1, "weights": weights, "scene": scene})

    def test_recipe_parses(self):
        with open(RECIPE) as handle:
            data = yaml.safe_load(handle)["data"]
        settings = AugmentationSettings.from_config(data["mixed_augmentation"], data["instruction_sampling"], False)
        self.assertTrue(settings.scene_enabled and settings.instruction_enabled)
        self.assertAlmostEqual(sum(settings.weights.values()), 1.0)

    def test_task_family_excludes_same_object_tasks(self):
        self.assertEqual(task_family("pick_diverse_bottles"), {"bottle"})
        self.assertTrue(task_family("stack_blocks_two") & task_family("blocks_ranking_rgb"))
        self.assertFalse(task_family("handover_mic") & task_family("place_shoe"))


class ScenePlanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.library = SceneLibrary(synthetic_library(self.directory.name))
        self.frame = self.library.median_head.astype(np.uint8).copy()
        self.frame[:, 40:50, 40:50] = 20

    def tearDown(self):
        self.directory.cleanup()

    def test_texture_is_deterministic_bounded_and_keeps_foreground(self):
        texture = generate_texture(np.random.default_rng(3), 64, 64)
        self.assertEqual(texture.shape, (3, 64, 64))
        self.assertTrue(np.isfinite(texture).all() and texture.min() >= 0 and texture.max() <= 1)
        first = ScenePlan(7, False, True, [], None).apply(self.frame, HEAD_CAMERA, self.library)
        second = ScenePlan(7, False, True, [], None).apply(self.frame, HEAD_CAMERA, self.library)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first[:, 40:50, 40:50], self.frame[:, 40:50, 40:50])
        self.assertFalse(np.array_equal(first[:, 10:30, 10:30], self.frame[:, 10:30, 10:30]))

    def test_clutter_placement_is_replayed_and_avoids_foreground(self):
        plan = ScenePlan(11, True, False, [0], np.zeros((64, 64), bool))
        current = plan.apply(self.frame, HEAD_CAMERA, self.library)
        placements = list(plan.placements)
        self.assertEqual(len(placements), 1)
        future_frame = self.frame.copy()
        future_frame[:, 20:24, :] = 30
        future = plan.apply(future_frame, HEAD_CAMERA, self.library)
        self.assertEqual(plan.placements, placements)
        changed = np.any(current != self.frame, axis=0)
        self.assertTrue(changed.any())
        self.assertFalse(changed[40:50, 40:50].any())
        np.testing.assert_array_equal(future[:, 20:24], future_frame[:, 20:24])
        wrist = plan.apply(self.frame, WRIST_CAMERAS[0], self.library)
        np.testing.assert_array_equal(wrist, self.frame)


@unittest.skipUnless(os.environ.get("ROBOTWIN_SCENE_INTEGRATION") == "1", "requires local clean data and library")
class RealSceneTests(unittest.TestCase):
    def test_real_scene_branches_keep_supervision(self):
        from transformers import AutoProcessor

        with open(RECIPE) as handle:
            recipe = yaml.safe_load(handle)
        data = copy.deepcopy(recipe["data"])
        for key in ("joints", "norm_type"):
            data[key] = [str(value) for value in data[key]]
        task_file = Path(tempfile.mkdtemp()) / "tasks.txt"
        lines = Path(data["train_path"]).read_text().splitlines()
        task_file.write_text("\n".join(line for line in lines if line.split("/")[-1] in (
            "handover_mic", "pick_diverse_bottles", "blocks_ranking_rgb")) + "\n")
        data.update(chunk_size=50, prompt_type="global", train_path=str(task_file))
        data = SimpleNamespace(**data)
        model_args = SimpleNamespace(tokenizer_path=recipe["model"]["tokenizer_path"])
        model = SimpleNamespace(max_state_dim=55, max_action_dim=55, tokenizer_max_length=72,
                                return_image_grid_thw=True, use_qwen3_chat_template=True)
        processor = AutoProcessor.from_pretrained(model_args.tokenizer_path, padding_side="right",
                                                  trust_remote_code=True, local_files_only=True)
        dataset = build_vla_dataset(data, model_args, model, processor, use_depth_align=True, allow_augmentation=True)
        for sub in dataset._datasets:
            raw = sub.dataset[200]
            baseline = sub.feature_transform.apply(copy.deepcopy(raw))
            for branch in ("clutter", "texture", "scene"):
                context = sub.augmenter.sample(raw, AugmentationIndex(200, 0, 5, 0))
                context.branch = branch
                rng = np.random.default_rng(context.random_key)
                crops = rng.choice(sub.augmenter.scene_candidates, 6, replace=False).tolist()
                context.scene_plan = ScenePlan(context.random_key, branch != "texture", branch != "clutter", crops,
                                               sub.augmenter.scene_library.foreground(sub.augmenter.task,
                                                                                      int(raw["episode_index"])))
                output = sub.feature_transform.apply(copy.deepcopy(raw), augmentation=context)
                for key in ("state", "actions", "action_is_pad", "joint_mask"):
                    self.assertTrue(torch.equal(output[key], baseline[key]), key)
                self.assertEqual(output["pil_images"].shape, baseline["pil_images"].shape)
                self.assertEqual(output["future_pil_images"].shape, baseline["future_pil_images"].shape)
                self.assertFalse(torch.equal(output["pil_images"], baseline["pil_images"]), branch)
        counts, started = {}, time.time()
        for index in range(0, 600, 3):
            item_dataset = dataset._datasets[index % len(dataset._datasets)]
            access = AugmentationIndex(index, 0, index, 0)
            raw = item_dataset.dataset[index]
            context = item_dataset.augmenter.sample(raw, access)
            counts[context.branch] = counts.get(context.branch, 0) + 1
            item_dataset.feature_transform.apply(raw, augmentation=context)
        print(f"REAL SCENE PASS branches={counts} {200 / (time.time() - started):.1f} samples/s single process")


if __name__ == "__main__":
    unittest.main()
