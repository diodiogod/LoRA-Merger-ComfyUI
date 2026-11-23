"""Run with ComfyUI's Python; uses temporary tiny LoRAs, never user models."""

import asyncio
import gc
import importlib
from pathlib import Path
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
sys.path.insert(0, str(COMFY))
sys.argv = ["lora-manager-tests", "--cpu"]

import torch
import comfy.lora
import comfy.utils
import execution
import folder_paths
import nodes
from comfy.model_patcher import ModelPatcher
from comfy.weight_adapter.lora import LoRAAdapter
from safetensors.torch import load_file, save_file

package = types.ModuleType("pm_lora_manager_native_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
bridge = importlib.import_module(f"{package.__name__}.src.lora_manager")
Adapter = bridge.PMLoraManagerAdapter
Concatenate = bridge.PMLoraManagerConcatenate
NATIVE_MODEL_KEYS = comfy.lora.model_lora_keys_unet
NATIVE_CLIP_KEYS = comfy.lora.model_lora_keys_clip
KEY = "diffusion_model.blocks.0.attn.weight"
CLIP_KEY = "text_encoders.clip_l.layer.weight"


class LoraFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(gc.collect)
        self.root = Path(self.temp.name)
        self.model = types.SimpleNamespace(model=object())
        self.clip = types.SimpleNamespace(cond_stage_model=object())
        self.map = {"test": KEY}

        def model_keys(model, mapping):
            mapping.update(self.map)
            return mapping

        def clip_keys(clip, mapping):
            mapping["text"] = CLIP_KEY
            mapping[f"text_encoders.{CLIP_KEY[:-7]}"] = CLIP_KEY
            return mapping

        self.enterContext(patch("comfy.lora.model_lora_keys_unet", side_effect=model_keys))
        self.clip_keys = self.enterContext(patch("comfy.lora.model_lora_keys_clip", side_effect=clip_keys))
        self.enterContext(patch.dict(folder_paths.folder_names_and_paths, {
            "loras": ([str(self.root)], folder_paths.supported_pt_extensions),
        }))

    def write_lora(self, name, rank, alpha=None, with_clip=False, conv=False, prefix="test"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(rank + 100)
        up = torch.randn(24, rank, generator=generator) * 0.1
        down = torch.randn(rank, 16 * (9 if conv else 1), generator=generator) * 0.1
        if conv:
            up, down = up.reshape(24, rank, 1, 1), down.reshape(rank, 16, 3, 3)
        state = {f"{prefix}.lora_up.weight": up, f"{prefix}.lora_down.weight": down}
        if alpha is not None:
            state[f"{prefix}.alpha"] = torch.tensor(float(alpha))
        if with_clip:
            state["text.lora_up.weight"] = torch.randn(8, rank, generator=generator) * 0.1
            state["text.lora_down.weight"] = torch.randn(rank, 10, generator=generator) * 0.1
        save_file(state, str(path))
        return str(path)

    def expected_delta(self, path, key=KEY):
        mapping = {**self.map, "text": CLIP_KEY}
        adapter = comfy.lora.load_lora(load_file(path), mapping)[key]
        up, down, alpha, _, _, _ = adapter.weights
        return up.flatten(1) @ down.flatten(1) * (float(alpha) / down.shape[0] if alpha is not None else 1)


class AdapterTests(LoraFixture):
    def test_different_ranks_and_alpha_keep_updates_and_strengths(self):
        a = self.write_lora("A.safetensors", 2, alpha=1)
        b = self.write_lora("B.safetensors", 5)
        components, strengths, info = Adapter.execute([(a, 1.0, 0.2), (b, 0.3, 0.7)], self.model).result
        self.assertEqual(len(strengths), 2)
        self.assertEqual(list(strengths.values()), [
            {"strength_model": 1.0, "strength_clip": 0.2},
            {"strength_model": 0.3, "strength_clip": 0.7},
        ])
        for (up, down, alpha), path in zip(components[KEY].values(), (a, b)):
            self.assertEqual(down.shape[0], 7)
            torch.testing.assert_close(up @ down * (alpha / 7), self.expected_delta(path))
        self.assertIn("PM rank budget: 7-7", info)
        self.clip_keys.assert_not_called()

    def test_clip_strength_and_clip_components_remain_separate(self):
        path = self.write_lora("with_clip.safetensors", 3, with_clip=True)
        components, strengths, _ = Adapter.execute([(path, -0.4, 1.2)], self.model, self.clip).result
        self.assertEqual(set(components), {KEY, CLIP_KEY})
        self.assertEqual(next(iter(strengths.values())), {"strength_model": -0.4, "strength_clip": 1.2})
        up, down, alpha = next(iter(components[CLIP_KEY].values()))
        torch.testing.assert_close(up @ down * (alpha / 3), self.expected_delta(path, CLIP_KEY))

    def test_repeated_files_names_and_relative_paths(self):
        a = self.write_lora("one/style.safetensors", 2)
        b = self.write_lora("two/style.safetensors", 4)
        with patch("comfy.utils.load_torch_file", wraps=comfy.utils.load_torch_file) as load:
            components, strengths, _ = Adapter.execute([
                ("one/style.safetensors", "1", "0"), (b, 0.3, 0), (a, -0.2, 0),
            ], self.model).result
        self.assertEqual(load.call_count, 2)
        self.assertEqual(len(strengths), 3)
        self.assertEqual(len(components[KEY]), 3)
        self.assertEqual(next(iter(components[KEY].values()))[1].shape[0], 8)

    def test_disjoint_targets_do_not_inflate_each_others_rank(self):
        self.map["second"] = "diffusion_model.blocks.1.attn.weight"
        a = self.write_lora("A.safetensors", 2)
        b = self.write_lora("B.safetensors", 5, prefix="second")
        components, _, _ = Adapter.execute([(a, 1, 0), (b, 0.3, 0)], self.model).result
        self.assertEqual(sorted(next(iter(v.values()))[1].shape[0] for v in components.values()), [2, 5])

    def test_plain_convolution_factors_preserve_native_delta(self):
        a = self.write_lora("conv_a.safetensors", 2, alpha=1, conv=True)
        b = self.write_lora("conv_b.safetensors", 5, conv=True)
        components, _, _ = Adapter.execute([(a, 1, 0), (b, 0.3, 0)], self.model).result
        for (up, down, alpha), path in zip(components[KEY].values(), (a, b)):
            torch.testing.assert_close(up @ down * (alpha / down.shape[0]), self.expected_delta(path))

    def test_empty_and_muted_stacks_do_not_read_files(self):
        with patch("comfy.utils.load_torch_file") as load:
            components, strengths, info = Adapter.execute([("missing.safetensors", 0, 0)], self.model).result
        load.assert_not_called()
        self.assertFalse(components)
        self.assertFalse(strengths)
        self.assertIn("No active LoRAs", info)

    def test_invalid_entries_and_missing_files_fail_clearly(self):
        for stack in ([('bad', 1)], [('bad', float('nan'), 0)], [('', 1, 0)]):
            with self.subTest(stack=stack), self.assertRaises(ValueError):
                Adapter.execute(stack, self.model)
        with self.assertRaises(FileNotFoundError):
            Adapter.execute([("missing.safetensors", 1, 0)], self.model)

    def test_unsupported_adapters_are_not_silently_dropped(self):
        path = self.write_lora("A.safetensors", 2)
        dora = LoRAAdapter(set(), (torch.ones(24, 2), torch.ones(2, 16), 2, None, torch.ones(24), None))
        for adapter in (("diff", (torch.ones(24, 16),)), dora):
            with self.subTest(adapter=type(adapter).__name__), patch("comfy.lora.load_lora", return_value={KEY: adapter}):
                with self.assertRaises(ValueError):
                    Adapter.execute([(path, 1, 0)], self.model)

    def test_file_changes_invalidate_cached_components(self):
        path = self.write_lora("A.safetensors", 2)
        stack = [(path, 1, 0)]
        before = Adapter.fingerprint_inputs(stack)
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        self.assertNotEqual(before, Adapter.fingerprint_inputs(stack))

    def test_actual_pm_merge_matches_native_weighted_stack(self):
        pm_root = COMFY / "custom_nodes" / "LoRA-Merger-ComfyUI"
        package = types.ModuleType("pm_bridge_native_test")
        package.__path__ = [str(pm_root)]
        sys.modules[package.__name__] = package
        Merger = importlib.import_module(f"{package.__name__}.src.lora_mergekit_merge").LoraMergerMergekit
        a = self.write_lora("A.safetensors", 2, alpha=1)
        b = self.write_lora("B.safetensors", 5)
        for sa, sb in ((1.0, 0.3), (-0.5, 1.1)):
            with self.subTest(strengths=(sa, sb)):
                components, strengths, _ = Adapter.execute([(a, sa, 0), (b, sb, 0)], self.model).result
                result, _ = Merger().lora_mergekit(
                    method={"name": "task_arithmetic", "settings": {"normalize": False, "rescale_norm": "none"}},
                    components=components, strengths=strengths, output_scale=1, spectral_norm_scale=0,
                    merge_clip=False, device="cpu", dtype="float32", refactor_method="rSVD", offload_models=False,
                )
                actual = comfy.lora.calculate_weight([(1.0, result["lora"][KEY], 1.0, None, None)], torch.zeros(24, 16), KEY)
                source_patches = [
                    (strength, comfy.lora.load_lora(load_file(path), self.map)[KEY], 1.0, None, None)
                    for path, strength in ((a, sa), (b, sb))
                ]
                expected = comfy.lora.calculate_weight(source_patches, torch.zeros(24, 16), KEY)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
                self.assertEqual(result["lora"][KEY].weights[1].shape[0], 7)

    def test_schema_and_registration(self):
        schema = Adapter.define_schema()
        self.assertEqual([value.io_type for value in schema.outputs], ["LoRATensors", "LoRAWeights", "STRING"])
        self.assertTrue(asyncio.run(nodes.load_custom_node(str(ROOT))))
        registered = nodes.NODE_CLASS_MAPPINGS[schema.node_id]
        self.assertEqual(list(registered.RETURN_TYPES), [out.io_type for out in schema.outputs])
        self.assertTrue(registered.__module__.endswith(".src.lora_manager"))


class TinyWeights(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.model_config = types.SimpleNamespace(unet_config={})

    def state_dict(self):
        return dict(self.weights)


class TinyClip:
    def __init__(self, patcher):
        self.patcher = patcher
        self.cond_stage_model = patcher.model

    def clone(self):
        return TinyClip(self.patcher.clone())

    def add_patches(self, patches, strength):
        return self.patcher.add_patches(patches, strength)


class ConcatenationTests(LoraFixture):
    def setUp(self):
        super().setUp()
        self.weights = {KEY: torch.zeros(24, 16)}
        self.model = ModelPatcher(TinyWeights(self.weights), torch.device("cpu"), torch.device("cpu"))
        self.clip = TinyClip(ModelPatcher(TinyWeights({CLIP_KEY: torch.zeros(8, 10)}),
                                          torch.device("cpu"), torch.device("cpu")))
        self.map[KEY[:-7]] = KEY
        pm_root = COMFY / "custom_nodes" / "LoRA-Merger-ComfyUI"
        package = types.ModuleType("pm_concatenate_native_test")
        package.__path__ = [str(pm_root)]
        sys.modules[package.__name__] = package
        self.Apply = importlib.import_module(f"{package.__name__}.src.lora_apply").LoraApply
        self.Save = importlib.import_module(f"{package.__name__}.src.lora_save").LoraSave
        self.save_module = sys.modules[f"{package.__name__}.src.lora_save"]

    def native_stack_result(self, stack):
        model, clip = self.model.clone(), self.clip.clone()
        mapping = {**self.map, "text": CLIP_KEY}
        for path, strength_model, strength_clip in stack:
            patches = comfy.lora.load_lora(load_file(path), mapping)
            model.add_patches(patches, strength_model)
            clip.add_patches(patches, strength_clip)
        return model, clip

    def assert_patched_equal(self, actual, expected):
        self.assertEqual(set(actual.patches), set(expected.patches))
        for key, weight in actual.model.state_dict().items():
            a = comfy.lora.calculate_weight(actual.patches.get(key, []), weight.clone(), key)
            b = comfy.lora.calculate_weight(expected.patches.get(key, []), weight.clone(), key)
            torch.testing.assert_close(a, b, atol=1e-6, rtol=2e-5)

    def save_and_reload(self, bundle):
        with patch.object(self.save_module, "convert_to_regular_lora", side_effect=AssertionError("Must not reconvert")):
            self.Save().lora_save(self.model, bundle, "result", "safetensors")
        saved = comfy.utils.load_torch_file(str(self.root / "result.safetensors"), safe_load=True)
        native_map = NATIVE_MODEL_KEYS(self.model.model, {})
        native_map = NATIVE_CLIP_KEYS(self.clip.cond_stage_model, native_map)
        patches = comfy.lora.load_lora(saved, native_map)
        self.assertEqual(set(patches), set(bundle["lora"]))
        return {"lora": patches, "strength_model": 1.0, "strength_clip": 1.0}

    def test_different_ranks_preview_and_pm_save_match_native_stack(self):
        a = self.write_lora("A.safetensors", 2, alpha=1)
        b = self.write_lora("B.safetensors", 5)
        for sa, sb in ((1.0, 0.3), (-0.5, 1.1)):
            with self.subTest(strengths=(sa, sb)):
                stack = [(a, sa, 0), (b, sb, 0)]
                bundle, info = Concatenate.execute(stack, self.model).result
                self.assertEqual(bundle["lora"][KEY].weights[1].shape[0], 7)
                self.assertEqual(bundle["strength_model"], 1)
                self.assertIn("Concatenated rank: 7-7", info)
                self.assertIn("FP32 tensor payload:", info)
                applied, clip = self.Apply().apply_merged_lora(self.model, self.clip, bundle)
                expected, _ = self.native_stack_result(stack)
                self.assert_patched_equal(applied, expected)
                self.assertFalse(self.model.patches)
                self.assertFalse(self.clip.patcher.patches)
                self.assertFalse(clip.patcher.patches)
                saved = self.save_and_reload(bundle)
                reloaded, _ = self.Apply().apply_merged_lora(self.model, self.clip, saved)
                self.assert_patched_equal(reloaded, applied)

    def test_model_and_clip_strengths_are_baked_independently(self):
        a = self.write_lora("A.safetensors", 2, alpha=1, with_clip=True)
        b = self.write_lora("B.safetensors", 5, with_clip=True)
        stack = [(a, 1, -0.2), (b, 0.3, 0.7)]
        bundle, _ = Concatenate.execute(stack, self.model, self.clip).result
        model, clip = self.Apply().apply_merged_lora(self.model, self.clip, bundle)
        expected_model, expected_clip = self.native_stack_result(stack)
        self.assert_patched_equal(model, expected_model)
        self.assert_patched_equal(clip.patcher, expected_clip.patcher)
        saved = self.save_and_reload(bundle)
        saved_model, saved_clip = self.Apply().apply_merged_lora(self.model, self.clip, saved)
        self.assert_patched_equal(saved_model, model)
        self.assert_patched_equal(saved_clip.patcher, clip.patcher)

    def test_zero_strengths_do_not_inflate_rank_or_add_wrong_component(self):
        a = self.write_lora("A.safetensors", 2, with_clip=True)
        b = self.write_lora("B.safetensors", 5, with_clip=True)
        bundle, _ = Concatenate.execute([(a, 1, 0), (b, 0, 0.3)], self.model, self.clip).result
        self.assertEqual(bundle["lora"][KEY].weights[1].shape[0], 2)
        self.assertEqual(bundle["lora"][CLIP_KEY].weights[1].shape[0], 5)

    def test_half_precision_sources_produce_fp32_factors_without_svd(self):
        path = self.write_lora("half.safetensors", 3, alpha=1.5)
        state = {k: value.half() for k, value in load_file(path).items()}
        save_file(state, path)
        with patch("torch.linalg.svd", side_effect=AssertionError("No SVD")), \
                patch("torch.svd_lowrank", side_effect=AssertionError("No SVD")):
            bundle, _ = Concatenate.execute([(path, 0.3, 0)], self.model).result
        up, down, alpha, _, _, _ = bundle["lora"][KEY].weights
        self.assertEqual(up.dtype, torch.float32)
        self.assertEqual(down.dtype, torch.float32)
        expected, _ = self.native_stack_result([(path, 0.3, 0)])
        actual = up @ down * (alpha / 3)
        reference = comfy.lora.calculate_weight(expected.patches[KEY], self.weights[KEY].clone(), KEY)
        torch.testing.assert_close(actual, reference)

    def test_plain_convolution_round_trips_without_changing_delta(self):
        self.weights[KEY] = torch.zeros(24, 16, 3, 3)
        a = self.write_lora("A.safetensors", 2, alpha=1, conv=True)
        b = self.write_lora("B.safetensors", 5, conv=True)
        stack = [(a, 1, 0), (b, 0.3, 0)]
        bundle, _ = Concatenate.execute(stack, self.model).result
        applied, _ = self.Apply().apply_merged_lora(self.model, self.clip, bundle)
        expected, _ = self.native_stack_result(stack)
        self.assert_patched_equal(applied, expected)
        saved, _ = self.Apply().apply_merged_lora(self.model, self.clip, self.save_and_reload(bundle))
        self.assert_patched_equal(saved, applied)

    def test_split_qkv_and_full_qkv_with_different_down_factors(self):
        fused = "diffusion_model.layers.0.attn.proj.qkv.weight"
        self.weights.clear()
        self.weights[fused] = torch.zeros(72, 16)
        self.map.clear()
        self.map[fused[:-7]] = fused
        for i, proj in enumerate(("q", "k", "v")):
            self.map[proj] = (fused, (0, i * 24, 24))
        a = self.write_lora("q.safetensors", 2, alpha=1, prefix="q")
        b = self.write_lora("k.safetensors", 5, prefix="k")
        c = self.write_lora("v.safetensors", 3, prefix="v")
        full = self.root / "full.safetensors"
        save_file({
            f"{fused[:-7]}.lora_up.weight": torch.randn(72, 4) * 0.1,
            f"{fused[:-7]}.lora_down.weight": torch.randn(4, 16) * 0.1,
        }, str(full))
        stack = [(a, 1, 0), (b, 0.3, 0), (c, -0.5, 0), (str(full), 0.7, 0)]
        bundle, _ = Concatenate.execute(stack, self.model).result
        self.assertEqual(set(bundle["lora"]), {fused})
        self.assertEqual(bundle["lora"][fused].weights[1].shape[0], 14)
        applied, _ = self.Apply().apply_merged_lora(self.model, self.clip, bundle)
        expected, _ = self.native_stack_result(stack)
        self.assert_patched_equal(applied, expected)
        saved, _ = self.Apply().apply_merged_lora(self.model, self.clip, self.save_and_reload(bundle))
        self.assert_patched_equal(saved, applied)

    def test_input_column_slices_and_partial_coverage(self):
        self.weights[KEY] = torch.zeros(24, 32)
        self.map["left"] = (KEY, (1, 0, 16))
        self.map["right"] = (KEY, (1, 16, 16))
        a = self.write_lora("left.safetensors", 2, prefix="left")
        b = self.write_lora("right.safetensors", 5, prefix="right")
        stack = [(a, 1, 0), (b, 0.3, 0)]
        bundle, _ = Concatenate.execute(stack, self.model).result
        applied, _ = self.Apply().apply_merged_lora(self.model, self.clip, bundle)
        expected, _ = self.native_stack_result(stack)
        self.assert_patched_equal(applied, expected)
        self.save_and_reload(bundle)

    def test_repeated_files_and_disjoint_targets_keep_exact_contributions(self):
        second = "diffusion_model.blocks.1.attn.weight"
        self.map["second"] = second
        self.weights[second] = torch.zeros(24, 16)
        a = self.write_lora("A.safetensors", 2)
        b = self.write_lora("B.safetensors", 5, prefix="second")
        stack = [(a, 1, 0), (b, 0.3, 0), (a, -0.2, 0)]
        with patch("comfy.utils.load_torch_file", wraps=comfy.utils.load_torch_file) as load:
            bundle, _ = Concatenate.execute(stack, self.model).result
        self.assertEqual(load.call_count, 2)
        self.assertEqual(sorted(adapter.weights[1].shape[0] for adapter in bundle["lora"].values()), [4, 5])
        applied, _ = self.Apply().apply_merged_lora(self.model, self.clip, bundle)
        expected, _ = self.native_stack_result(stack)
        self.assert_patched_equal(applied, expected)

    def test_bad_shapes_alpha_and_adapters_fail_instead_of_silent_wrong_merge(self):
        path = self.write_lora("A.safetensors", 2)
        self.weights[KEY] = torch.zeros(25, 16)
        with self.assertRaisesRegex(ValueError, "shape"):
            Concatenate.execute([(path, 1, 0)], self.model)
        self.weights[KEY] = torch.zeros(24, 16)
        up, down = torch.ones(24, 2), torch.ones(2, 16)
        adapters = [
            LoRAAdapter(set(), (up, down, float("nan"), None, None, None)),
            LoRAAdapter(set(), (up, down, 2, None, torch.ones(24), None)),
            ("diff", (torch.ones(24, 16),)),
        ]
        for adapter in adapters:
            with self.subTest(adapter=type(adapter).__name__), patch("comfy.lora.load_lora", return_value={KEY: adapter}):
                with self.assertRaises(ValueError):
                    Concatenate.execute([(path, 1, 0)], self.model)

    def test_empty_stack_rejected_and_files_still_invalidate_cache(self):
        with self.assertRaisesRegex(ValueError, "No active LoRA"):
            Concatenate.execute([], self.model)
        path = self.write_lora("A.safetensors", 2)
        stack = [(path, 1, 0)]
        before = Concatenate.fingerprint_inputs(stack)
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        self.assertNotEqual(before, Concatenate.fingerprint_inputs(stack))

    def test_pm_save_regular_bundle_still_uses_original_converter(self):
        raw = {"example": torch.ones(2)}
        bundle = {"lora": {"example": object()}}
        with patch.object(self.save_module, "convert_to_regular_lora", return_value=raw) as convert:
            self.Save().lora_save(self.model, bundle, "regular", "safetensors")
        convert.assert_called_once_with(self.model, bundle["lora"])
        torch.testing.assert_close(load_file(str(self.root / "regular.safetensors"))["example"], raw["example"])

    def test_all_registered_nodes_runtime_metadata_matches_advertised_schema(self):
        self.assertTrue(asyncio.run(nodes.load_custom_node(str(ROOT))))
        for node_id in ("PMLoraManagerAdapter", "PMLoraManagerConcatenate"):
            node = nodes.NODE_CLASS_MAPPINGS[node_id]
            with self.subTest(node=node.__name__):
                info = node.GET_NODE_INFO_V1()
                self.assertEqual(list(node.RETURN_TYPES), info["output"])
                self.assertEqual(list(node.RETURN_NAMES), info["output_name"])
                self.assertEqual(list(node.OUTPUT_IS_LIST), info["output_is_list"])
                self.assertEqual(node.DESCRIPTION, info["description"])

    def test_native_prompt_validation_accepts_concatenate_to_pm_apply(self):
        self.assertTrue(asyncio.run(nodes.load_custom_node(str(ROOT))))
        nodes.NODE_CLASS_MAPPINGS["PMLoraManagerAdapter"].GET_NODE_INFO_V1()
        nodes.NODE_CLASS_MAPPINGS["PMLoraManagerConcatenate"].GET_NODE_INFO_V1()
        source = type("ConcatenateTestSource", (), {
            "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
            "RETURN_TYPES": ("LORA_STACK", "MODEL", "CLIP"),
        })
        prompt = {
            "source": {"class_type": "ConcatenateTestSource", "inputs": {}},
            "concat": {"class_type": "PMLoraManagerConcatenate", "inputs": {
                "lora_stack": ["source", 0], "model": ["source", 1], "clip": ["source", 2],
            }},
            "apply": {"class_type": "ConcatenateTestApply", "inputs": {
                "lora": ["concat", 0], "model": ["source", 1], "clip": ["source", 2],
            }},
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {
            "ConcatenateTestSource": source,
            "ConcatenateTestApply": self.Apply,
        }):
            valid, errors, _ = asyncio.run(execution.validate_inputs("concatenate-test", prompt, "apply", {}))
        self.assertTrue(valid, errors)

    def test_schema_and_registration(self):
        self.assertEqual([out.io_type for out in Concatenate.define_schema().outputs], ["LoRABundle", "STRING"])
        self.assertEqual(Concatenate.define_schema().node_id, "PMLoraManagerConcatenate")


if __name__ == "__main__":
    unittest.main(argv=["lora-manager-tests"], verbosity=2)
