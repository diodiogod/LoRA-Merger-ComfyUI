import math
import os

import torch
import comfy.lora
import comfy.utils
import folder_paths
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_api.latest import io


def stack_entries(lora_stack):
    for index, entry in enumerate(lora_stack):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ValueError(f"LoRA stack entry {index + 1} must contain a file, model strength, and CLIP strength")
        filename, model_strength, clip_strength = entry
        model_strength, clip_strength = float(model_strength), float(clip_strength)
        if not math.isfinite(model_strength) or not math.isfinite(clip_strength):
            raise ValueError(f"LoRA stack entry {index + 1} has a non-finite strength")
        if model_strength == 0 and clip_strength == 0:
            continue
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"LoRA stack entry {index + 1} has no filename")
        path = filename if os.path.isabs(filename) else folder_paths.get_full_path_or_raise("loras", filename)
        path = os.path.realpath(path)
        if os.path.splitext(path)[1].lower() not in folder_paths.supported_pt_extensions:
            raise ValueError(f"Unsupported LoRA file type: {filename}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LoRA not found: {filename}")
        yield f"{index + 1}: {os.path.basename(path)}", path, model_strength, clip_strength


def stack_fingerprint(lora_stack):
    if lora_stack is None:
        return float("nan")
    result = []
    for _, path, _, _ in stack_entries(lora_stack):
        stat = os.stat(path)
        result.append((path, stat.st_mtime_ns, stat.st_size))
    return tuple(result)


def pad_components(components):
    for key, sources in components.items():
        shapes = {(up.shape[0], down.shape[1]) for up, down, _ in sources.values()}
        if len(shapes) != 1:
            raise ValueError(f"LoRAs target incompatible shapes for {key}; use LoRAs for the same base model")
        combined_rank = sum(down.shape[0] for _, down, _ in sources.values())
        for name, (up, down, alpha) in sources.items():
            rank = down.shape[0]
            scale = float(alpha) / rank if alpha is not None else 1.0
            padding = combined_rank - rank
            if padding:
                up = torch.cat((up, up.new_zeros((up.shape[0], padding))), dim=1)
                down = torch.cat((down, down.new_zeros((padding, down.shape[1]))), dim=0)
            # Preserve alpha/rank while giving PM enough rank for the summed update.
            sources[name] = (up, down, combined_rank * scale)
    return components


def load_components(lora_stack, key_map):
    components, strengths, loaded, info = {}, {}, {}, []
    for name, path, model_strength, clip_strength in stack_entries(lora_stack):
        if path not in loaded:
            loaded[path] = comfy.lora.load_lora(comfy.utils.load_torch_file(path, safe_load=True), key_map)
        patches = loaded[path]
        if not patches:
            raise ValueError(f"No weights in {path} match the connected model/CLIP")
        ranks = []
        for key, adapter in patches.items():
            if not isinstance(adapter, LoRAAdapter):
                raise ValueError(f"{name}: {key} is {type(adapter).__name__}; this PM adapter requires ordinary LoRA factors")
            up, down, alpha, mid, dora_scale, reshape = adapter.weights
            if any(value is not None for value in (mid, dora_scale, reshape)):
                raise ValueError(f"{name}: {key} uses LoCon, DoRA, or reshaped weights, which this adapter does not support")
            up, down = up.flatten(start_dim=1), down.flatten(start_dim=1)
            if up.shape[1] != down.shape[0] or down.shape[0] == 0:
                raise ValueError(f"{name}: incompatible LoRA factor shapes for {key}")
            components.setdefault(key, {})[name] = (up, down, alpha)
            ranks.append(down.shape[0])
        strengths[name] = {"strength_model": model_strength, "strength_clip": clip_strength}
        info.append(f"{name}: rank {min(ranks)}-{max(ranks)}, model {model_strength:g}, CLIP {clip_strength:g}")
    return components, strengths, info


class PMLoraManagerAdapter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PMLoraManagerAdapter",
            display_name="PM LoRA Manager Adapter",
            category="LoRA PowerMerge",
            description="Convert LoRA Manager's stack directly to PM Merger components and strengths. "
                        "Ordinary LoRAs only. Zero-padding preserves input updates and keeps the sum of input ranks; "
                        "it uses more RAM but avoids Decompose rank resizing. Nothing is applied or saved here.",
            inputs=[
                io.Custom("LORA_STACK").Input("lora_stack"),
                io.Model.Input("model"),
                io.Clip.Input("clip", optional=True),
            ],
            outputs=[
                io.Custom("LoRATensors").Output(display_name="components"),
                io.Custom("LoRAWeights").Output(display_name="strengths"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, lora_stack=None, **kwargs):
        return stack_fingerprint(lora_stack)

    @classmethod
    def execute(cls, lora_stack, model, clip=None):
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        components, strengths, info = load_components(lora_stack, key_map)
        pad_components(components)
        if components:
            ranks = [next(iter(sources.values()))[1].shape[0] for sources in components.values()]
            info.append(f"PM rank budget: {min(ranks)}-{max(ranks)} across {len(components)} target weights")
        else:
            info.append("No active LoRAs")
        if clip is None:
            info.append("CLIP not connected: model weights only")
        return io.NodeOutput(components, strengths, "\n".join(info))
