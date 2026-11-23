import math
import os

import torch
import comfy.lora
import comfy.utils
import folder_paths
from comfy.weight_adapter import LoRAAdapter
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


class PMLoraManagerConcatenate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PMLoraManagerConcatenate",
            display_name="PM Direct Concatenate LoRAs",
            category="LoRA PowerMerge",
            description="Concatenate LoRA Manager's stack without SVD or rank reduction. "
                        "Different ranks and negative strengths are supported. Source strengths and alpha/rank "
                        "are baked in; apply the result at 1.0. Rank is the sum of contributing ranks, so files "
                        "can be large. Ordinary LoRAs only, not LoKr/LoHa/DoRA. Connect directly to PM Apply/Save.",
            inputs=[
                io.Custom("LORA_STACK").Input("lora_stack"),
                io.Model.Input("model"),
                io.Clip.Input("clip", optional=True),
            ],
            outputs=[
                io.Custom("LoRABundle").Output(display_name="lora"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, lora_stack=None, **kwargs):
        return stack_fingerprint(lora_stack)

    @classmethod
    def execute(cls, lora_stack, model, clip=None):
        model_map = comfy.lora.model_lora_keys_unet(model.model, {})
        clip_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {}) if clip is not None else {}
        model_keys, clip_keys = set(model_map.values()), set(clip_map.values())
        if model_keys & clip_keys or any(k in model_map and model_map[k] != v for k, v in clip_map.items()):
            raise ValueError("Model and CLIP have ambiguous LoRA targets; cannot bake their strengths separately")
        components, strengths, info = load_components(lora_stack, {**model_map, **clip_map})
        states = {"strength_model": model.model.state_dict()}
        if clip is not None:
            states["strength_clip"] = clip.cond_stage_model.state_dict()
        combined, owners = {}, {}
        for key, sources in components.items():
            owner = "strength_model" if key in model_keys else "strength_clip"
            if isinstance(key, tuple):
                if len(key) != 2 or len(key[1]) != 3:
                    raise ValueError(f"Unsupported LoRA target transform: {key}")
                target, offset = key
            else:
                target, offset = key, None
            shape = tuple(states[owner][target].shape)
            patch_shape = list(shape)
            if offset is not None:
                dim, start, length = offset
                if dim not in (0, 1) or (dim == 1 and len(shape) != 2):
                    raise ValueError(f"Unsupported LoRA slice dimension for {target}: {offset}")
                if start < 0 or length <= 0 or start + length > shape[dim]:
                    raise ValueError(f"Invalid LoRA slice for {target}: {offset}")
                patch_shape[dim] = length
            for name, (up, down, alpha) in sources.items():
                strength = strengths[name][owner]
                if strength == 0:
                    continue
                if up.shape[0] != patch_shape[0] or down.shape[1] != math.prod(patch_shape[1:]):
                    raise ValueError(f"{name}: LoRA factors do not match the connected model's shape for {target}")
                scale = strength * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
                if not math.isfinite(scale):
                    raise ValueError(f"{name}: non-finite LoRA alpha/scale for {target}")
                if scale == 0:
                    continue
                up = up.to(device="cpu", dtype=torch.float32) * scale
                down = down.to(device="cpu", dtype=torch.float32)
                # Expand separate Q/K/V slices before concatenating, keeping distinct down matrices.
                if offset is not None:
                    if dim == 0:
                        full_up = up.new_zeros((shape[0], up.shape[1]))
                        full_up[start:start + length] = up
                        up = full_up
                    else:
                        full_down = down.new_zeros((down.shape[0], shape[1]))
                        full_down[:, start:start + length] = down
                        down = full_down
                combined.setdefault(target, []).append((up, down))
                owners[target] = owner
        if not combined:
            raise ValueError("No active LoRA weights for the connected model/CLIP")
        patches, save_state, ranks = {}, {}, []
        for target, factors in combined.items():
            up = torch.cat([pair[0] for pair in factors], dim=1)
            down = torch.cat([pair[1] for pair in factors], dim=0)
            rank = down.shape[0]
            alpha = torch.tensor(float(rank))
            patches[target] = LoRAAdapter(set(), (up, down, float(rank), None, None, None))
            key_map = model_map if owners[target] == "strength_model" else clip_map
            prefix = target[:-len(".weight")] if target.endswith(".weight") else target
            if owners[target] == "strength_clip":
                prefix = f"text_encoders.{prefix}"
            if key_map.get(prefix) != target:
                prefix = next((alias for alias, value in key_map.items() if value == target), None)
            if prefix is None or f"{prefix}.lora_up.weight" in save_state:
                raise ValueError(f"No unambiguous full-weight export name for {target}")
            save_state[f"{prefix}.lora_up.weight"] = up
            save_state[f"{prefix}.lora_down.weight"] = down
            save_state[f"{prefix}.alpha"] = alpha
            ranks.append(rank)
        size = sum(t.numel() * t.element_size() for t in save_state.values()) / (1024 * 1024)
        info.append(f"Concatenated rank: {min(ranks)}-{max(ranks)} across {len(patches)} target weights")
        info.append(f"FP32 tensor payload: {size:.2f} MiB (file header not included); no SVD or compression")
        info.append("Source strengths are baked in. Apply/load the result at model 1.0, CLIP 1.0.")
        if clip is None:
            info.append("CLIP not connected: model weights only")
        bundle = {"lora": patches, "save_state_dict": save_state,
                  "strength_model": 1.0, "strength_clip": 1.0, "name": "Direct concatenation"}
        return io.NodeOutput(bundle, "\n".join(info))
