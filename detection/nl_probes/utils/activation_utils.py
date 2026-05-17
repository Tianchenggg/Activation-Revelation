import contextlib

import torch
from transformers import AutoModelForCausalLM

from nl_probes.utils.common import get_candidate_layer_stack_paths


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `torch.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = torch.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)  # type: ignore
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD  # type: ignore


def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    if min_offset is not None:
        assert max_offset is not None, "max_offset must be provided if min_offset is provided"
        assert max_offset < min_offset, "max_offset must be less than min_offset"
        assert min_offset < 0, "min_offset must be less than 0"
        assert max_offset < 0, "max_offset must be less than 0"
    else:
        assert max_offset is None, "max_offset must be provided if min_offset is not provided"

    activations_BLD_by_layer = {}

    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}

    max_layer = max(submodules.keys())

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]

        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs

        if min_offset is not None:
            activations_BLD_by_layer[layer] = activations_BLD_by_layer[layer][:, max_offset:min_offset, :]

        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    handles = []

    for layer, submodule in submodules.items():
        handles.append(submodule.register_forward_hook(gather_target_act_hook))

    try:
        # Use the selected context manager
        with torch.no_grad():
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        for handle in handles:
            handle.remove()

    return activations_BLD_by_layer


# LoRA target patterns for text-only training on VLMs (vision-language models).
# When using target_modules='all-linear', LoRA adapters get added to the vision
# tower which won't receive gradients during text-only training, causing DDP errors.
# These patterns target only the language model layers.
VLM_TEXT_ONLY_LORA_TARGETS = {
    "gemma-3": r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    "qwen3-vl": r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
}


def get_text_only_lora_targets(model_name: str) -> str | None:
    """Returns LoRA target pattern for text-only training on VLMs, or None if not a VLM."""
    for pattern, targets in VLM_TEXT_ONLY_LORA_TARGETS.items():
        if pattern in model_name.lower():
            return targets
    return None


def _get_attr_path(module: torch.nn.Module, path: str):
    current = module
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _resolve_named_submodule(model: torch.nn.Module, module_path: str, use_lora: bool) -> torch.nn.Module:
    normalized = module_path.strip().strip(".")
    if not normalized:
        raise ValueError("Layer/module path must be a non-empty string.")

    candidates = [normalized]
    if normalized.startswith("model."):
        candidates.append(normalized[len("model.") :])
    else:
        candidates.append(f"model.{normalized}")

    if use_lora:
        lora_candidates = []
        for candidate in candidates:
            lora_candidates.append(f"base_model.model.{candidate}")
            lora_candidates.append(f"base_model.{candidate}")
        candidates.extend(lora_candidates)

    named_modules = dict(model.named_modules())
    for candidate in dict.fromkeys(candidates):
        if candidate in named_modules:
            return named_modules[candidate]

    raise ValueError(
        f"Module path '{module_path}' was not found. "
        "Tip: generate paths using `for name, _ in model.named_modules(): print(name)`."
    )


def get_hf_submodule(model: AutoModelForCausalLM, layer: int | str, use_lora: bool = False):
    """Gets the target hook submodule for HF transformers."""
    if isinstance(layer, str):
        return _resolve_named_submodule(model, layer, use_lora=use_lora)

    model_name = model.config._name_or_path
    layer_paths = get_candidate_layer_stack_paths(model_name, use_lora=use_lora, config=model.config)

    for layer_path in layer_paths:
        layer_stack = _get_attr_path(model, layer_path)
        if layer_stack is not None:
            return layer_stack[layer]

    raise ValueError(f"Please add submodule path for model {model_name}")
