"""CPU-only native ComfyUI round trips; no downloaded models or image generation."""

import gc
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
sys.path.insert(0, str(COMFY))
sys.argv = ["moss-export-tests", "--cpu"]

import torch
import comfy.lora
import comfy.model_base
import comfy.sd
import comfy.utils
import folder_paths
from comfy.model_patcher import ModelPatcher
from comfy.weight_adapter import LoRAAdapter

package = types.ModuleType("pm_moss_export_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
exporter = importlib.import_module(f"{package.__name__}.src.architectures.sd_lora")
Save = importlib.import_module(f"{package.__name__}.src.lora_save").LoraSave
Resize = importlib.import_module(f"{package.__name__}.src.lora_resize").LoraResizer
Merger = importlib.import_module(f"{package.__name__}.src.lora_mergekit_merge").LoraMergerMergekit

BASE = "diffusion_model.layers.0.attn.proj"
KEY = f"{BASE}.qkv.weight"


class TinyMOSS(comfy.model_base.MOSSImage):
    def __init__(self):
        # Use the real MOSS key mapper without allocating a production model.
        torch.nn.Module.__init__(self)
        self.model_config = types.SimpleNamespace(unet_config={})
        self.diffusion_model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.attn = torch.nn.Module()
        layer.attn.proj = torch.nn.Module()
        layer.attn.proj.qkv = torch.nn.Linear(16, 24, bias=False)
        layer.ff = torch.nn.Linear(16, 8, bias=False)
        self.diffusion_model.layers = torch.nn.ModuleList([layer])


class MOSSExportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(137)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(gc.collect)
        self.root = Path(self.temp.name)
        self.enterContext(patch.dict(folder_paths.folder_names_and_paths, {
            "loras": ([str(self.root)], folder_paths.supported_pt_extensions),
        }))
        self.model = ModelPatcher(TinyMOSS(), torch.device("cpu"), torch.device("cpu"))
        self.key_map = comfy.lora.model_lora_keys_unet(self.model.model, {})
        for i, projection in enumerate(("q", "k", "v")):
            self.assertEqual(self.key_map[f"{BASE}.{projection}"], (KEY, (0, i * 8, 8)))

    def raw_lora(self, specifications):
        raw = {}
        for projection, rank, alpha in specifications:
            prefix = f"{BASE}.{projection}" if projection != "ff" else "diffusion_model.layers.0.ff"
            rows = 24 if projection == "qkv" else 8
            raw[f"{prefix}.lora_up.weight"] = torch.randn(rows, rank) * 0.1
            raw[f"{prefix}.lora_down.weight"] = torch.randn(rank, 16) * 0.1
            if alpha is not None:
                raw[f"{prefix}.alpha"] = torch.tensor(float(alpha))
        return raw

    def patches(self, raw):
        return comfy.lora.load_lora(raw, self.key_map)

    def patched_weights(self, bundle):
        model = self.model.clone()
        model.add_patches(bundle["lora"], bundle.get("strength_model", 1.0))
        return {
            key: comfy.lora.calculate_weight(model.patches.get(key, []), weight.clone(), key)
            for key, weight in self.model.model.state_dict().items()
        }

    def round_trip(self, bundle, name="result"):
        expected = self.patched_weights(bundle)
        Save().lora_save(self.model, bundle, name, "safetensors")
        raw = comfy.utils.load_torch_file(str(self.root / f"{name}.safetensors"), safe_load=True)
        reloaded, _ = comfy.sd.load_lora_for_models(self.model, None, raw, 1.0, 0.0)
        for key, weight in self.model.model.state_dict().items():
            actual = comfy.lora.calculate_weight(reloaded.patches.get(key, []), weight.clone(), key)
            error = (actual - expected[key]).abs().max().item()
            torch.testing.assert_close(actual, expected[key], atol=1e-6, rtol=2e-5,
                                       msg=f"{name}: saved/reloaded {key} changed; max absolute error {error:g}")
        return raw

    def test_distinct_qkv_down_matrices_preserve_the_preview(self):
        raw = self.raw_lora([(p, 3, 3) for p in ("q", "k", "v")])
        self.assertFalse(torch.equal(raw[f"{BASE}.q.lora_down.weight"], raw[f"{BASE}.k.lora_down.weight"]))
        self.round_trip({"lora": self.patches(raw)})

    def test_shared_down_factors_are_still_serializable(self):
        raw = self.raw_lora([(p, 3, 3) for p in ("q", "k", "v")])
        for projection in ("k", "v"):
            raw[f"{BASE}.{projection}.lora_down.weight"] = raw[f"{BASE}.q.lora_down.weight"]
        self.round_trip({"lora": self.patches(raw)})

    def test_different_ranks_alphas_and_missing_alpha_stay_separate(self):
        raw = self.raw_lora([("q", 2, 0.75), ("k", 5, None), ("v", 3, -1.5)])
        saved = self.round_trip({"lora": self.patches(raw)})
        loaded = self.patches(saved)
        self.assertEqual(sorted(adapter.weights[1].shape[0] for adapter in loaded.values()), [2, 3, 5])

    def test_partial_projection_coverage_keeps_original_offsets(self):
        for projection in ("q", "k", "v"):
            with self.subTest(projection=projection):
                raw = self.raw_lora([(projection, 3, None)])
                saved = self.round_trip({"lora": self.patches(raw)}, name=projection)
                self.assertEqual(set(self.patches(saved)), set(self.patches(raw)))

    def test_fused_qkv_and_regular_weights_are_unchanged(self):
        raw = self.raw_lora([("qkv", 4, None), ("ff", 3, None)])
        self.round_trip({"lora": self.patches(raw)})

    def test_fused_and_split_updates_do_not_overwrite_each_other(self):
        raw = self.raw_lora([("qkv", 4, 1.25), ("q", 2, 0.75), ("v", 3, 3)])
        saved = self.round_trip({"lora": self.patches(raw)})
        self.assertEqual(len(self.patches(saved)), 3)

    def test_merge_then_resize_then_save_matches_resized_preview(self):
        sources = {
            "A": self.patches(self.raw_lora([(p, 2, 1) for p in ("q", "k", "v")])),
            "B": self.patches(self.raw_lora([(p, 5, 3) for p in ("q", "k", "v")])),
        }
        components = {}
        for name, patches in sources.items():
            for key, adapter in patches.items():
                up, down, alpha, _, _, _ = adapter.weights
                rank = down.shape[0]
                padding = 7 - rank
                up = torch.cat([up, up.new_zeros((up.shape[0], padding))], dim=1)
                down = torch.cat([down, down.new_zeros((padding, down.shape[1]))], dim=0)
                components.setdefault(key, {})[name] = (up, down, 7 * alpha / rank)
        merged, _ = Merger().lora_mergekit(
            method={"name": "task_arithmetic", "settings": {"normalize": False, "rescale_norm": "none"}},
            components=components, strengths={
                "A": {"strength_model": 1.0, "strength_clip": 0.0},
                "B": {"strength_model": 0.3, "strength_clip": 0.0},
            }, output_scale=1.0, spectral_norm_scale=0.0, merge_clip=False,
            device="cpu", dtype="float32", refactor_method="rSVD", offload_models=False,
        )
        for method in ("SVD", "rSVD", "energy_rSVD"):
            for rank in (1, 2, 4):
                with self.subTest(method=method, rank=rank):
                    resized, = Resize().lora_resize(merged, method, rank, "cpu", "float32")
                    self.assertTrue(all(a.weights[1].shape[0] == rank for a in resized["lora"].values()))
                    saved = self.round_trip(resized, name=f"{method}_{rank}")
                    self.assertTrue(all(a.weights[1].shape[0] == rank for a in self.patches(saved).values()))

    def test_unknown_slice_mapping_fails_instead_of_writing_wrong_weights(self):
        adapter = LoRAAdapter(set(), (torch.ones(4, 2), torch.ones(2, 16), 2, None, None, None))
        with self.assertRaisesRegex(ValueError, "slice"):
            exporter.convert_to_regular_lora(self.model, {(KEY, (0, 1, 4)): adapter})


if __name__ == "__main__":
    unittest.main(argv=["moss-export-tests"], verbosity=2)
