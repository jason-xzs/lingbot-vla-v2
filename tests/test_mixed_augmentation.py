import copy
import json
import os
import tempfile
import time
import unittest
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import torch
import yaml

from lingbotvla.data.vla_data.augmentation import (
    ArrayCache,
    AugmentationDataError,
    AugmentationIndex,
    AugmentationSampler,
    AugmentationSettings,
    DatasetAugmenter,
    RECIPE_VERSION,
    augmentation_errors,
    fft_augment,
    file_sha256,
    low_frequency_amplitude,
    pixel_sha256,
    stream_seed,
)
from lingbotvla.data.vla_data.transform import apply_visual_augmentation, sample_visual_augmentation_params
from lingbotvla.data.vla_data.utils import FeatureTransform
from lingbotvla.data.vla_data.multi_vla_dataset import MultiVLADataset
from lingbotvla.data.vla_data.base_dataset import VLADataset
from lingbotvla.data.dataset import build_vla_dataset
from lingbotvla.utils.arguments import DataArguments, parse_args
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


@dataclass
class ParserFixture:
    data: DataArguments = field(default_factory=DataArguments)


class LogicalDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, access):
        return torch.tensor([access.index, access.epoch, access.position, access.rank,
                             stream_seed(42, "task@rev", 0, access.index, access, "branch")])


class ErrorDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        with augmentation_errors("task=test episode=3 frame=9", "cache_read"):
            raise FileNotFoundError("cache missing")


def color_config():
    return {"enabled": True, "seed": 42, "weights": {"original": 0.4, "color": 0.6, "fft": 0, "adain": 0}}


class ConfigurationTests(unittest.TestCase):
    def test_matched_six_task_baseline(self):
        root = Path("configs/vla/robotwin")
        baseline = yaml.safe_load((root / "robotwin_clean_6task_baseline_a800_6gpu.yaml").read_text())
        augmented = yaml.safe_load((root / "robotwin_clean_6task_mixaug_nocolor_a800_6gpu.yaml").read_text())
        self.assertEqual(baseline["model"], augmented["model"])
        self.assertNotEqual(baseline["train"].pop("output_dir"), augmented["train"].pop("output_dir"))
        self.assertEqual(baseline["train"], augmented["train"])
        self.assertEqual(baseline["train"]["seed"], 42)
        self.assertEqual(baseline["data"]["train_path"], "assets/training_data/robotwin_clean_6task.txt")
        for name in ("mixed_augmentation", "instruction_sampling"):
            self.assertIs(baseline["data"][name]["enabled"], False)
            baseline["data"].pop(name)
            augmented["data"].pop(name)
        self.assertFalse(baseline["data"]["image_augment"])
        self.assertEqual(baseline["data"], augmented["data"])

    def test_parser_preserves_new_nested_types(self):
        arguments = ["test", "--data.train_path", "unused", "--data.mixed_augmentation", json.dumps(color_config())]
        with patch.object(sys, "argv", arguments):
            parsed = parse_args(ParserFixture)
        self.assertEqual(parsed.data.mixed_augmentation["weights"]["color"], 0.6)
        wrong = color_config()
        wrong["weights"]["color"] = "0.6"
        arguments[-1] = json.dumps(wrong)
        with patch.object(sys, "argv", arguments), self.assertRaises(ValueError):
            parse_args(ParserFixture)

    def test_disabled_defaults(self):
        self.assertFalse(AugmentationSettings.from_config().enabled)

    def test_bad_types_and_unknown_keys(self):
        invalid = [
            {"seed": True}, {"seed": "42"}, {"enabled": 1}, {"unexpected": True},
            {"fft": {"low_freq_ratio": "1e-3"}}, {"fft": {"amplitude_mix": float("nan")}},
            {"adain": {"alpha": float("inf")}}, {"adain": {"alpha": True}},
            {"style_guard_tasks": "scan_object"}, {"style_manifest": 7},
            {"weights": {"original": 1, "color": 1, "fft": 0, "adain": 0}},
        ]
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(ValueError):
                config = color_config()
                config.update(update)
                AugmentationSettings.from_config(config)

    def test_supported_values(self):
        for update in ({"teacher_view": "original"}, {"sync_views_and_time": False},
                       {"guarded_style_fallback": "color"}, {"adain": {"cache_miss": "original"}}):
            config = color_config()
            config.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                AugmentationSettings.from_config(config)
        for update in ({"split": "unseen"}, {"missing_pool": "original"}):
            with self.assertRaises(ValueError):
                AugmentationSettings.from_config(color_config(), update)

    def test_instruction_only_requires_seed(self):
        instructions = {"enabled": True, "pool_manifest": "/missing.json"}
        with self.assertRaises(ValueError):
            AugmentationSettings.from_config(None, instructions)
        self.assertTrue(AugmentationSettings.from_config({"seed": 42}, instructions).instruction_enabled)

    def test_conflicts_and_guards(self):
        with self.assertRaises(ValueError):
            AugmentationSettings.from_config(color_config(), legacy=True)
        config = AugmentationSettings.from_config(color_config())
        self.assertEqual(config.validate_tasks(["handover_mic", "scan_object"]), ["scan_object"])
        wrong = color_config()
        wrong["style_guard_tasks"] = ["scan_objcet"]
        with self.assertRaises(ValueError):
            AugmentationSettings.from_config(wrong).validate_tasks(["handover_mic"])


class TransformTests(unittest.TestCase):
    def test_color_rng_isolated_and_replayable(self):
        image = torch.ones(3, 16, 16, dtype=torch.uint8)
        rng_before = torch.get_rng_state().clone()
        first = sample_visual_augmentation_params(image, torch.Generator().manual_seed(42))
        second = sample_visual_augmentation_params(image, torch.Generator().manual_seed(42))
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        identity = {name: torch.tensor(1.0) for name in first}
        self.assertTrue(torch.equal(image, apply_visual_augmentation(image, identity)))

    def test_stream_independence_and_diversity(self):
        access = AugmentationIndex(3, epoch=2, position=100, rank=0)
        args = (42, "task@revision", 1, 3, access)
        self.assertEqual(stream_seed(*args, "color"), stream_seed(*args, "color"))
        self.assertNotEqual(stream_seed(*args, "color"), stream_seed(*args, "instruction"))
        self.assertNotEqual(stream_seed(*args, "color"), stream_seed(42, "task@revision", 1, 3,
                                                                   AugmentationIndex(3, 3, 100, 0), "color"))

    def test_fft_identity_and_contract(self):
        generator = torch.Generator().manual_seed(7)
        image = torch.randint(30, 210, (3, 256, 256), dtype=torch.uint8, generator=generator)
        reference = torch.randint(30, 210, image.shape, dtype=torch.uint8, generator=generator)
        amplitude = low_frequency_amplitude(reference, 0.01)
        self.assertEqual(tuple(amplitude.shape), (3, 5, 5))
        self.assertIs(fft_augment(image, None, 0, 0.2), image)
        self.assertIs(fft_augment(image, None, 0.01, 0), image)
        output = fft_augment(image, amplitude, 0.01, 0.2)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, torch.uint8)
        self.assertTrue(torch.equal(image, fft_augment(image, low_frequency_amplitude(image, 0.01), 0.01, 0.2)))
        video = image.unsqueeze(0).repeat(2, 1, 1, 1)
        result = fft_augment(video, amplitude, 0.01, 0.2)
        self.assertTrue(torch.equal(result[0], result[1]))

    def test_fft_dc_channel_means(self):
        image = torch.full((3, 256, 256), 100, dtype=torch.uint8)
        style = torch.tensor([50, 150, 200], dtype=torch.uint8)[:, None, None].expand_as(image)
        output = fft_augment(image, low_frequency_amplitude(style, 0.01), 0.01, 0.2)
        torch.testing.assert_close(output.float().mean((1, 2)), torch.tensor([90., 110., 120.]))

    def test_error_boundary_preserves_exit_signals(self):
        for error in (FileNotFoundError("missing"), OSError("broken"), ValueError("shape"), KeyError("index")):
            with self.subTest(error=error), self.assertRaisesRegex(AugmentationDataError, "stage=cache.*episode=3"):
                with augmentation_errors("task=test episode=3 frame=9", "cache"):
                    raise error
        with self.assertRaises(KeyboardInterrupt):
            with augmentation_errors("test", "cache"):
                raise KeyboardInterrupt()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        from scripts.prepare_robotwin_augmentation import make_parser

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.roots = [self.root / task for task in ("handover_mic", "scan_object")]
        for root in self.roots:
            (root / "meta").mkdir(parents=True)
            provenance = {"source": {"revision": "test", "sha256": "source",
                                     "path": f"dataset/{root.name}/aloha-agilex_clean_50.zip"},
                          "episodes": 1, "frames": 2, "source_episode_ids": [0]}
            (root / "meta/robotwin_source.json").write_text(json.dumps(provenance))
        tasks = self.root / "tasks.txt"
        tasks.write_text("".join(f"robotwin {root}\n" for root in self.roots))
        self.recipe = self.root / "recipe.yaml"
        mixed = color_config()
        mixed["style_guard_tasks"] = ["handover_mic", "rotate_qrcode"]
        self.recipe.write_text(yaml.safe_dump({"data": {"mixed_augmentation": mixed}}))
        self.output = self.root / "output"
        self.output.mkdir()
        (self.output / "seen_pools.json").write_text(json.dumps({"datasets": {f"{root.name}@test": {} for root in self.roots}}))
        (self.output / "style_manifest.json").write_text(json.dumps({"styles": [{"id": "style_0"}], "preprocessing": {}}))
        (self.output / "models").mkdir()
        (self.output / "models/manifest.json").write_text("{}")
        self.args = make_parser().parse_args([
            "cache", "--tasks", str(tasks), "--output", str(self.output), "--recipe-config", str(self.recipe),
            "--gpus", "0", "--num-styles", "1",
        ])

    def test_style_selection_uses_recipe_guards(self):
        from scripts.prepare_robotwin_augmentation import prepare_styles

        with patch("scripts.prepare_robotwin_augmentation.open_source", side_effect=RuntimeError("selection captured")) as source:
            with self.assertRaisesRegex(RuntimeError, "selection captured"):
                prepare_styles(self.args)
        self.assertEqual(source.call_args.args[0].name, "scan_object")

    def test_cache_jobs_and_contract_use_recipe_guards(self):
        from scripts.prepare_robotwin_augmentation import prepare_cache

        with patch("scripts.prepare_robotwin_augmentation.torch.cuda.device_count", return_value=1), \
                patch("scripts.prepare_robotwin_augmentation.torch.cuda.mem_get_info", return_value=(2 ** 30, 2 ** 30)), \
                patch("scripts.prepare_robotwin_augmentation.ProcessPoolExecutor") as executor:
            submit = executor.return_value.__enter__.return_value.submit
            submit.side_effect = RuntimeError("jobs captured")
            with self.assertRaisesRegex(RuntimeError, "jobs captured"):
                prepare_cache(self.args)
        jobs = submit.call_args.args[2]
        self.assertEqual(jobs, [(str(self.roots[1]), 0)])
        generation = json.loads((self.output / "adain_cache/generation.json").read_text())
        self.assertEqual(generation["style_guard_tasks"], ["handover_mic", "rotate_qrcode"])

    def test_guards_validate_names_and_inherit_generation_without_recipe(self):
        from scripts.prepare_robotwin_augmentation import preparation_settings

        self.args.recipe_config = None
        generation = {"style_guard_tasks": ["handover_mic"]}
        self.assertEqual(preparation_settings(self.args, self.roots, generation).guards, ("handover_mic",))
        self.assertIn("scan_object", preparation_settings(self.args, self.roots).guards)
        self.args.recipe_config = str(self.recipe)
        self.recipe.write_text(yaml.safe_dump({"data": {"mixed_augmentation": {"style_guard_tasks": ["scan_objcet"]}}}))
        with self.assertRaisesRegex(ValueError, "Unknown style_guard_tasks"):
            preparation_settings(self.args, self.roots)

    def test_audit_rejects_guard_mismatch_before_loading_images(self):
        from scripts.prepare_robotwin_augmentation import audit_cache

        cache = self.output / "adain_cache"
        cache.mkdir()
        generation_path = cache / "generation.json"
        generation_path.write_text(json.dumps({"style_guard_tasks": ["scan_object"]}))
        (cache / "index.json").write_text(json.dumps({"generation_sha256": file_sha256(generation_path)}))
        with patch("scripts.prepare_robotwin_augmentation.open_source", side_effect=AssertionError("must fail before decoding")) as source:
            with self.assertRaisesRegex(ValueError, "style_guard_tasks differ"):
                audit_cache(self.args)
            source.assert_not_called()
            generation_path.write_text(json.dumps({"style_guard_tasks": []}))
            with self.assertRaisesRegex(ValueError, "contract checksum mismatch"):
                audit_cache(self.args)


class ResourceTests(unittest.TestCase):
    def test_full_cache_checksums_include_every_group(self):
        from scripts.prepare_robotwin_augmentation import verify_cache_groups

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for episode in range(2):
                path = root / f"episode{episode}.npy"
                np.save(path, np.full((2, 3, 8, 8), episode, dtype=np.uint8))
                entries.append({"dataset_uid": "task@version", "source_episode_id": episode,
                                "camera": "top", "style_id": "clean_000", "path": path.name,
                                "file_size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                                "sha256": file_sha256(path)})
            self.assertEqual(verify_cache_groups(root, entries), 2)
            entries[-1]["sha256"] = "corrupt"
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_cache_groups(root, entries)

    def test_adain_requires_matching_successful_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            styles = root / "styles.json"
            styles.write_text("{}")
            index = {"schema_version": 1, "recipe_version": RECIPE_VERSION, "preprocessing": {}, "alpha": 0.2,
                     "style_manifest_sha256": file_sha256(styles), "complete": True,
                     "generation_sha256": "generation", "entries": []}
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index))
            engine = DatasetAugmenter.__new__(DatasetAugmenter)
            engine.settings = SimpleNamespace(cache_root=str(root), style_manifest=str(styles), alpha=0.2, max_open_arrays=1)
            engine.preprocessing = {}
            engine.episode_lengths = {}
            with self.assertRaises(FileNotFoundError):
                engine._load_cache()
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({"passed": True, "index_sha256": "wrong",
                                             "generation_sha256": "generation", "inputs_checked": 1}))
            with self.assertRaisesRegex(ValueError, "matching successful"):
                engine._load_cache()
            audit_path.write_text(json.dumps({"passed": True, "index_sha256": file_sha256(index_path),
                                             "generation_sha256": "generation", "inputs_checked": 1}))
            engine._load_cache()
            engine.cache.close()

    def test_instruction_cleaning_preserves_position_semantics(self):
        from scripts.prepare_robotwin_augmentation import clean_seen

        texts, report = clean_seen(["", "tap <top center>", "tap <top center>", "move {object_name}", "use <arm>"])
        self.assertEqual(texts, ["tap <top center>"])
        self.assertEqual(report, {"input": 5, "empty": 1, "duplicates": 1, "placeholders": 2})
        with self.assertRaises(ValueError):
            clean_seen(["", "use <arm>"])

    def test_preparation_never_replaces_different_artifacts(self):
        from scripts.prepare_robotwin_augmentation import save_array_once, write_json_once

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json_once(root / "manifest.json", {"version": 1})
            write_json_once(root / "manifest.json", {"version": 1})
            with self.assertRaises(FileExistsError):
                write_json_once(root / "manifest.json", {"version": 2})
            pixels = np.zeros((3, 8, 8), dtype=np.uint8)
            save_array_once(root / "pixels.npy", pixels)
            save_array_once(root / "pixels.npy", pixels)
            with self.assertRaises(FileExistsError):
                save_array_once(root / "pixels.npy", pixels + 1)

    def test_array_cache_copies_and_bounds_mappings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for index in range(3):
                pixels = np.full((2, 3, 8, 8), index, dtype=np.uint8)
                path = root / f"{index}.npy"
                np.save(path, pixels)
                entries.append({"path": path.name, "shape": list(pixels.shape), "file_size": path.stat().st_size,
                                "mtime_ns": path.stat().st_mtime_ns,
                                "frame_sha256": [pixel_sha256(frame) for frame in pixels]})
            cache = ArrayCache(root, max_open=1)
            first = cache.read(entries[0], [0])
            for _ in range(3):
                for entry in entries:
                    cache.read(entry, [0, 1])
            self.assertEqual(cache.peak_open, 1)
            cache.close()
            self.assertEqual(int(first.sum()), 0)
            self.assertEqual(len(cache._arrays), 0)
            entries[0]["frame_sha256"][0] = "wrong"
            with self.assertRaisesRegex(ValueError, "checksum"):
                cache.read(entries[0], [0])

    def test_episode_instruction_identity_and_synchronized_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "handover_mic"
            (root / "meta").mkdir(parents=True)
            provenance = {"source": {"revision": "test", "sha256": "archive",
                                     "path": "dataset/handover_mic/aloha-agilex_clean_50.zip"},
                          "episodes": 2, "source_episode_ids": [5, 9]}
            (root / "meta/robotwin_source.json").write_text(json.dumps(provenance))
            pool_path = root / "pools.json"
            pool_path.write_text(json.dumps({"schema_version": 1, "split": "seen", "datasets": {
                "handover_mic@test": {"source_sha256": "archive", "source_episode_ids": [5, 9],
                                      "episodes": {"5": ["first", "<top center>"], "9": ["second"]}},
            }}))
            settings = AugmentationSettings.from_config(color_config(), {"enabled": True, "pool_manifest": str(pool_path)})
            cameras = {"top": "cam_top", "left": "cam_left"}
            engine = DatasetAugmenter(settings, root, cameras, [2, 3], {"size": [8, 8]})
            image = torch.full((3, 8, 8), 100, dtype=torch.uint8)
            raw = {"episode_index": torch.tensor(1), "frame_index": torch.tensor(2), "cam_top": image,
                   "_augmentation_meta": {"episode_length": 3, "actual_query_frame_indices": {
                       "cam_top": [2, 2], "cam_left": [2, 2]}}}
            context = engine.sample(raw, AugmentationIndex(4))
            self.assertEqual(context.source_episode_id, 9)
            self.assertEqual(context.instruction, "second")
            context.branch = "color"
            context.color_params = sample_visual_augmentation_params(image, torch.Generator().manual_seed(4))
            batch = {group: {camera: image.clone() for camera in cameras} for group in ("image", "future_image")}
            context.apply_batch(batch)
            for group in batch.values():
                for result in group.values():
                    self.assertTrue(torch.equal(result, batch["image"]["top"]))
            self.assertEqual(len(context.metrics), 4)
            context.branch = "original"
            original = {"image": {"top": image}}
            context.apply_batch(original)
            self.assertIs(original["image"]["top"], image)


class IntegrationTests(unittest.TestCase):
    def test_sampler_workers_epoch_and_resume(self):
        dataset = LogicalDataset()

        def loader():
            sampler = AugmentationSampler(StatefulDistributedSampler(dataset, num_replicas=2, rank=1, seed=7))
            return StatefulDataLoader(dataset, batch_size=2, sampler=sampler, num_workers=2)

        original = loader()
        iterator = iter(original)
        next(iterator)
        next(iterator)
        state = copy.deepcopy(original.state_dict())
        remaining = list(iterator)
        restored = loader()
        restored.load_state_dict(state)
        replayed = list(restored)
        self.assertEqual(len(remaining), len(replayed))
        for expected, actual in zip(remaining, replayed):
            self.assertTrue(torch.equal(expected, actual))
        original.sampler.set_epoch(1)
        next_epoch = torch.cat(list(original))
        self.assertTrue(torch.all(next_epoch[:, 1] == 1))
        self.assertEqual(next_epoch[:, 2].tolist(), list(range(10)))

    def test_fatal_error_survives_worker_wrapper(self):
        loader = StatefulDataLoader(ErrorDataset(), batch_size=1, num_workers=1)
        with self.assertRaisesRegex(AugmentationDataError, "task=test episode=3 frame=9.*FileNotFoundError"):
            next(iter(loader))

    def test_multidataset_does_not_retry_augmentation_errors(self):
        dataset = MultiVLADataset.__new__(MultiVLADataset)
        dataset._datasets = [SimpleNamespace(dataset=SimpleNamespace(num_frames=2))]
        dataset.augmentation_enabled = False
        with patch.object(dataset, "getdata", side_effect=AugmentationDataError("fatal")) as getdata:
            with self.assertRaises(AugmentationDataError):
                dataset[0]
            self.assertEqual(getdata.call_count, 1)

    def test_feature_transform_actions_and_teacher(self):
        class Tokenizer:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"input_ids": torch.ones(1, 8, dtype=torch.long), "attention_mask": torch.ones(1, 8)}

        tokenizer = Tokenizer()
        processor = SimpleNamespace(tokenizer=tokenizer, image_processor=None)
        data = SimpleNamespace(joints=["{'arm.position': 14}", "{'end.position': 14}", "{'effector.position': 2}"],
                               cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
                               norm_type=["{'arm.position': 'bounds_99_woclip'}", "{'effector.position': 'bounds_99_woclip'}"])
        model = SimpleNamespace(max_state_dim=55, max_action_dim=55, tokenizer_max_length=8)
        transform = FeatureTransform("configs/robot_configs/robotwin.yaml", data, model, processor,
                                     chunk_size=2, use_depth_align=True, use_future_image=True, allow_augmentation=True)
        raw = {"observation.state": torch.arange(14).float() / 20, "action": torch.arange(28).reshape(2, 14).float() / 20,
               "action_is_pad": torch.tensor([False, True]), "task": "original instruction"}
        for camera in transform.org_features["images"]:
            raw[camera] = torch.full((2, 3, 8, 8), 100, dtype=torch.uint8)
        baseline = transform.apply(copy.deepcopy(raw))
        calls = []

        def dispatch(batch):
            calls.append(1)
            for group in ("image", "future_image"):
                batch[group] = {name: image + 10 for name, image in batch[group].items()}

        context = SimpleNamespace(instruction="sampled instruction", engine=SimpleNamespace(settings=SimpleNamespace(image_enabled=True)),
                                  apply_batch=dispatch)
        with patch("lingbotvla.data.vla_data.transform.apply_visual_augmentation", side_effect=AssertionError("double augmentation")):
            augmented = transform.apply(copy.deepcopy(raw), augmentation=context)
        self.assertEqual(len(calls), 1)
        for key in ("state", "actions", "action_is_pad", "joint_mask", "state_joint_mask", "action_joint_mask"):
            self.assertTrue(torch.equal(baseline[key], augmented[key]), key)
        self.assertTrue(torch.all(augmented["pil_images"] == 110))
        self.assertTrue(torch.equal(augmented["pil_images"], augmented["future_pil_images"]))
        self.assertNotIn("future_is_pad", augmented)
        self.assertIn("sampled instruction", tokenizer.prompt[0])
        recovered = transform.unapply(augmented)
        torch.testing.assert_close(recovered["action"], raw["action"], atol=1e-5, rtol=0, check_dtype=False)
        transform.allow_augmentation = False
        with patch.object(context, "apply_batch", side_effect=AssertionError("non-training augmentation")):
            unaugmented = transform.apply(copy.deepcopy(raw), augmentation=context)
        self.assertTrue(torch.equal(baseline["images"], unaugmented["images"]))
        self.assertIn("original instruction", tokenizer.prompt[0])


@unittest.skipUnless(os.environ.get("ROBOTWIN_AUGMENTATION_INTEGRATION") == "1", "requires prepared local six-task artifacts")
class RobotwinDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor

        path = "configs/vla/robotwin/robotwin_clean_6task_mixaug_nocolor_a800_6gpu.yaml"
        with open(path) as handle:
            cls.recipe = yaml.safe_load(handle)
        data = copy.deepcopy(cls.recipe["data"])
        for key in ("joints", "norm_type"):
            data[key] = [str(value) for value in data[key]]
        data.update(chunk_size=50, prompt_type="global")
        cls.data = SimpleNamespace(**data)
        cls.model_args = SimpleNamespace(tokenizer_path=cls.recipe["model"]["tokenizer_path"])
        cls.model = SimpleNamespace(max_state_dim=55, max_action_dim=55, tokenizer_max_length=72,
                                   return_image_grid_thw=True, use_qwen3_chat_template=True)
        cls.processor = AutoProcessor.from_pretrained(cls.model_args.tokenizer_path, padding_side="right",
                                                       trust_remote_code=True, local_files_only=True)
        cls.dataset = build_vla_dataset(cls.data, cls.model_args, cls.model, cls.processor,
                                        use_depth_align=True, allow_augmentation=True)

    @classmethod
    def tearDownClass(cls):
        for dataset in cls.dataset._datasets:
            if dataset.augmenter.cache is not None:
                dataset.augmenter.cache.close()

    def assert_same_actions(self, first, second):
        for key in ("state", "actions", "action_is_pad", "joint_mask", "state_joint_mask", "action_joint_mask"):
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_real_four_branches_and_future_clamping(self):
        for dataset in self.dataset._datasets:
            episode = dataset.dataset_meta.episodes[49]
            anchor = int(episode["dataset_to_index"]) - 2
            raw = dataset.dataset[anchor]
            length = int(episode["length"])
            for frames in raw["_augmentation_meta"]["actual_query_frame_indices"].values():
                self.assertEqual(frames, [length - 2, length - 1])
            baseline = dataset.feature_transform.apply(copy.deepcopy(raw))
            context = dataset.augmenter.sample(raw, AugmentationIndex(anchor, 3, 100, 1))
            branches = ("original",) if dataset.augmenter.guarded else ("original", "color", "fft", "adain")
            for branch in branches:
                with self.subTest(task=dataset.augmenter.task, branch=branch):
                    context.branch = branch
                    context.style_id = sorted(dataset.augmenter.styles)[0] if dataset.augmenter.styles else None
                    reference = raw[next(iter(dataset.augmenter.camera_sources.values()))]
                    context.color_params = sample_visual_augmentation_params(reference, torch.Generator().manual_seed(8))
                    output = dataset.feature_transform.apply(copy.deepcopy(raw), augmentation=context)
                    self.assert_same_actions(baseline, output)
                    self.assertEqual(output["images"].shape, baseline["images"].shape)
                    self.assertTrue(torch.equal(output["image_grid_thw"], baseline["image_grid_thw"]))
                    self.assertEqual(output["pil_images"].dtype, torch.uint8)
                    self.assertEqual(output["pil_images"].shape, (3, 3, 256, 256))
                    self.assertEqual(output["future_pil_images"].shape, (3, 3, 256, 256))
                    self.assertNotIn("future_is_pad", output)
                    if branch == "original":
                        self.assertTrue(torch.equal(baseline["images"], output["images"]))
                    recovered = dataset.feature_transform.unapply(output)
                    valid = ~raw["action_is_pad"]
                    torch.testing.assert_close(recovered["action"][valid], raw["action"][valid],
                                               atol=1e-5, rtol=0, check_dtype=False)
            print(f"REAL PASS {dataset.augmenter.task}: actions, views, processor, episode-end future frames", flush=True)

    def test_real_branch_distribution_and_stream_isolation(self):
        for dataset in self.dataset._datasets:
            raw = dataset.dataset[0]
            requested, actual, texts, keys = Counter(), Counter(), set(), set()
            rng_before = torch.get_rng_state().clone()
            for visit in range(3000):
                context = dataset.augmenter.sample(raw, AugmentationIndex(0, visit // 1000, visit, visit % 2))
                requested[context.requested_branch] += 1
                actual[context.branch] += 1
                texts.add(context.instruction)
                keys.add(context.random_key)
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            for branch, weight in dataset.augmentation_settings.weights.items():
                self.assertLess(abs(requested[branch] / 3000 - weight), 0.04)
            self.assertEqual(actual["color"], 0)
            self.assertGreater(len(texts), 1)
            self.assertEqual(len(keys), 3000)
            if dataset.augmenter.guarded:
                self.assertEqual(actual, {"original": 3000})
            print(f"REAL DISTRIBUTION {dataset.augmenter.task}: requested={dict(requested)}, actual={dict(actual)}, unique_texts={len(texts)}", flush=True)

    def test_real_nontraining_resource_isolation(self):
        data = copy.deepcopy(self.data)
        data.mixed_augmentation["style_manifest"] = "/missing/styles.json"
        data.mixed_augmentation["adain"]["cache_root"] = "/missing/cache"
        data.instruction_sampling["pool_manifest"] = "/missing/seen.json"
        with patch("lingbotvla.data.vla_data.base_dataset.DatasetAugmenter", side_effect=AssertionError("augmentation resources accessed")):
            normal = build_vla_dataset(data, self.model_args, self.model, self.processor, use_depth_align=True)
            self.assertFalse(normal.augmentation_enabled)
            for index, dataset in enumerate(normal._datasets):
                result = normal[normal.dataset_start_index[index]]
                raw = dataset.dataset[0]
                expected = dataset.feature_transform.apply(copy.deepcopy(raw))
                self.assertTrue(torch.equal(result["lang_tokens"], expected["lang_tokens"]))
                self.assertTrue(torch.equal(result["images"], expected["images"]))
            direct = VLADataset(normal.repo_ids[0], "robotwin", data, data.robot_config_root,
                                config=self.model, processor=self.processor, image_size=(256, 256),
                                use_future_image=True, use_depth_align=True)
            self.assertFalse(direct.augmentation_enabled)
            direct[0]
            raw = direct.dataset[0]
            policy = direct.feature_transform.apply(copy.deepcopy(raw), policy_eval=True)
            ground_truth = direct.feature_transform.apply(copy.deepcopy(raw))
            self.assertTrue(torch.equal(policy["images"], ground_truth["images"]))
            stats = build_vla_dataset(data, None, None, None, do_nomalize=False,
                                      return_item=True, disabled_image_features=True)
            self.assertFalse(stats.augmentation_enabled)
            for start in stats.dataset_start_index:
                self.assertIn("action.arm.position", stats[int(start)])
        print("REAL PASS nontraining: builder, direct dataset, policy_eval, ground-truth apply, statistics", flush=True)

    def test_real_two_worker_resume(self):
        def loader():
            sampler = AugmentationSampler(StatefulDistributedSampler(self.dataset, num_replicas=2, rank=1, seed=42))
            return StatefulDataLoader(self.dataset, batch_size=1, sampler=sampler, num_workers=2,
                                       multiprocessing_context="spawn", prefetch_factor=2)

        original = loader()
        iterator = iter(original)
        try:
            next(iterator)
            next(iterator)
            state = copy.deepcopy(original.state_dict())
            expected = [next(iterator) for _ in range(3)]
        finally:
            iterator._shutdown_workers()
        restored = loader()
        restored.load_state_dict(state)
        replay = iter(restored)
        try:
            for target in expected:
                actual = next(replay)
                for key, value in target.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(value, actual[key]), key)
                    else:
                        self.assertEqual(value, actual[key])
        finally:
            replay._shutdown_workers()
        print("REAL PASS two spawned workers: exact next-three-batch resume", flush=True)

    def test_real_cache_random_read_budget(self):
        dataset = self.dataset._datasets[0]
        cache = dataset.augmenter.cache
        cache.close()
        candidates = list(dataset.augmenter.entries.values())
        before_fds = len(os.listdir("/proc/self/fd"))
        start = time.perf_counter()
        pixels = 0
        for visit in range(200):
            entry = candidates[(visit * 37) % len(candidates)]
            frame = (visit * 19) % entry["shape"][0]
            result = cache.read(entry, [frame])
            pixels += result.numel()
        elapsed = time.perf_counter() - start
        peak_fds = len(os.listdir("/proc/self/fd")) - before_fds
        self.assertLessEqual(peak_fds, cache.max_open)
        cache.close()
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), before_fds)
        print(f"REAL CACHE READ: {200 / elapsed:.1f} frames/s, {pixels / elapsed / 1e6:.1f} MB/s, max_open={cache.peak_open}, fd_delta={peak_fds}", flush=True)

    @unittest.skipUnless(os.environ.get("ROBOTWIN_AUGMENTATION_TEACHER") == "1", "requires an explicitly available GPU and local teachers")
    def test_real_teacher_targets(self):
        from lingbotvla.models.vla.vision_models.module_utils import (
            build_depth_model, build_video_model, get_depth_target, get_video_target,
        )
        from scripts.prepare_robotwin_augmentation import write_json_once

        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = self.recipe["train"]["align_params"]
        depth_models = build_depth_model(config)
        video_model = build_video_model(config["video"])
        current, future, identities, baseline_positions = [], [], [], {}
        for dataset in self.dataset._datasets:
            episode = dataset.dataset_meta.episodes[25]
            anchor = int(episode["dataset_from_index"]) + int(episode["length"]) // 2
            raw = dataset.dataset[anchor]
            context = dataset.augmenter.sample(raw, AugmentationIndex(anchor, 1, 1, 0))
            variants = [("original", None)]
            if not dataset.augmenter.guarded:
                variants += [(branch, style) for branch in ("fft", "adain") for style in sorted(dataset.augmenter.styles)]
            baseline_positions[dataset.augmenter.task] = len(identities)
            for branch, style in variants:
                context.branch, context.style_id = branch, style
                sample = dataset.feature_transform.apply(copy.deepcopy(raw), augmentation=context)
                current.append(sample["pil_images"])
                future.append(sample["future_pil_images"])
                identities.append({"task": dataset.augmenter.task, "branch": branch, "style": style})
        current = torch.stack(current).cuda()
        future = torch.stack(future).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            current_depth, _ = get_depth_target("MoRGBD", depth_models, current)
            future_depth, _ = get_depth_target("MoRGBD", depth_models, future)
            video = get_video_target(video_model, current, future, config,
                                     effective_fps=torch.full((len(identities),), 50 / 49, device="cuda"))
        targets = {"depth_current": current_depth, "depth_future": future_depth,
                   "video_future": video["patch"], "video_current": video["current_patch"]}
        records = []
        for position, identity in enumerate(identities):
            record = dict(identity)
            baseline = baseline_positions[identity["task"]]
            for name, target in targets.items():
                self.assertTrue(torch.isfinite(target).all(), name)
                reference = target[baseline].float().flatten()
                candidate = target[position].float().flatten()
                self.assertGreater(float(candidate.std()), 1e-8, name)
                record[name] = {
                    "cosine": float(torch.nn.functional.cosine_similarity(reference, candidate, dim=0)),
                    "rms_ratio": float(candidate.square().mean().sqrt() / reference.square().mean().sqrt()),
                }
            records.append(record)
        report = {"passed": True, "samples": len(records), "gpu": torch.cuda.get_device_name(0),
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated(0), "dtype": "bfloat16 autocast",
                  "checks": "finite and nonconstant teacher features; paired cosine and RMS ratio",
                  "not_validated": "VLA optimization, teacher semantic accuracy, closed-loop task success",
                  "records": records}
        print("TEACHER SMOKE " + json.dumps(report), flush=True)
        report_path = os.environ.get("ROBOTWIN_TEACHER_REPORT")
        if report_path:
            write_json_once(report_path, report)


if __name__ == "__main__":
    unittest.main()