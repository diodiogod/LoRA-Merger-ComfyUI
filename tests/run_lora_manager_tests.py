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
import folder_paths
import nodes
from comfy.weight_adapter.lora import LoRAAdapter
from safetensors.torch import load_file, save_file

package = types.ModuleType("pm_lora_manager_native_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
bridge = importlib.import_module(f"{package.__name__}.src.lora_manager")
Adapter = bridge.PMLoraManagerAdapter
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


if __name__ == "__main__":
    unittest.main(argv=["lora-manager-tests"], verbosity=2)
