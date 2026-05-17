import random
import types
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedTokenizerBase, Qwen3VLForConditionalGeneration

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

import transformers

from nl_probes.utils.vlm_compat import (
    FAMILY_QWEN3_VL,
    detect_model_family,
    get_processor_tokenizer,
    patch_mllama_processor_chat_template,
    patch_transformers_remote_code_compat,
    postprocess_loaded_model,
    preferred_attn_implementation,
    prepare_model_config,
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _from_pretrained_with_cache_fallback(factory, model_name: str, *, label: str, **kwargs):
    try:
        return factory.from_pretrained(model_name, **kwargs)
    except Exception as primary_exc:
        if kwargs.get("local_files_only"):
            raise
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        try:
            obj = factory.from_pretrained(model_name, **local_kwargs)
        except Exception as local_exc:
            raise RuntimeError(
                f"{label} primary load failed: {primary_exc}; local cache fallback failed: {local_exc}"
            ) from local_exc
        print(
            f"{label} primary load failed; loaded from local cache instead. "
            f"Original error: {primary_exc}"
        )
        return obj


def _load_model_config(model_name: str):
    try:
        config = _from_pretrained_with_cache_fallback(
            AutoConfig,
            model_name,
            label="AutoConfig",
            trust_remote_code=True,
        )
    except Exception:
        return None
    return prepare_model_config(model_name, config=config)


def _model_identity_strings(model_name: str, config) -> list[str]:
    identities = [model_name.lower()]
    if config is None:
        return identities

    identities.append(str(getattr(config, "model_type", "")).lower())
    identities.append(config.__class__.__name__.lower())
    identities.extend(str(architecture).lower() for architecture in getattr(config, "architectures", []) or [])
    return identities


def _identity_contains(identities: list[str], needle: str) -> bool:
    normalized_needle = needle.lower()
    compact_needle = normalized_needle.replace("-", "").replace("_", "")
    for identity in identities:
        normalized_identity = identity.lower()
        compact_identity = normalized_identity.replace("-", "").replace("_", "")
        if normalized_needle in normalized_identity or compact_needle in compact_identity:
            return True
    return False


def _single_target_device(device_map: object) -> str | None:
    if isinstance(device_map, str):
        if device_map == "auto":
            return None
        return device_map
    if isinstance(device_map, dict) and set(device_map.keys()) == {""}:
        target = device_map.get("")
        return str(target) if target is not None else None
    return None


def _iter_load_kwargs(kwargs: dict[str, object]) -> list[tuple[dict[str, object], str | None]]:
    attempts: list[tuple[dict[str, object], str | None]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    target_device = _single_target_device(kwargs.get("device_map"))

    def _append(candidate: dict[str, object], move_device: str | None) -> None:
        key = tuple(sorted((str(k), repr(v)) for k, v in candidate.items()))
        if key in seen:
            return
        seen.add(key)
        attempts.append((candidate, move_device))

    _append(dict(kwargs), None)

    if "attn_implementation" in kwargs:
        candidate = dict(kwargs)
        candidate.pop("attn_implementation", None)
        _append(candidate, None)

    if target_device is not None and "device_map" in kwargs:
        candidate = dict(kwargs)
        candidate.pop("device_map", None)
        _append(candidate, target_device)
        if "attn_implementation" in candidate:
            candidate_no_attn = dict(candidate)
            candidate_no_attn.pop("attn_implementation", None)
            _append(candidate_no_attn, target_device)

    return attempts


def load_model(
    model_name: str,
    dtype: torch.dtype,
    **model_kwargs,
) -> AutoModelForCausalLM:
    print("Loading model...")
    patch_transformers_remote_code_compat()

    config = _load_model_config(model_name)
    family = detect_model_family(model_name, config=config)
    is_qwen3_vl = family == FAMILY_QWEN3_VL

    attn = preferred_attn_implementation(model_name, config=config)

    kwargs: dict = {
        "device_map": "auto",
        "attn_implementation": attn,
        "torch_dtype": dtype,
        "config": config,
        **model_kwargs,
    }
    print(f"Using attention implementation: {kwargs['attn_implementation']}")

    kwargs["trust_remote_code"] = True

    model_classes = []
    if is_qwen3_vl:
        model_classes.append(Qwen3VLForConditionalGeneration)
    if config is not None:
        for architecture in getattr(config, "architectures", []) or []:
            model_cls = getattr(transformers, str(architecture), None)
            if model_cls is not None and hasattr(model_cls, "from_pretrained"):
                model_classes.append(model_cls)
    for model_cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
        if model_cls is not None:
            model_classes.append(model_cls)

    tried: list[str] = []
    last_error: Exception | None = None
    seen: set[str] = set()
    for model_cls in model_classes:
        cls_name = getattr(model_cls, "__name__", str(model_cls))
        if cls_name in seen:
            continue
        seen.add(cls_name)
        tried.append(cls_name)
        for candidate_kwargs, move_device in _iter_load_kwargs(kwargs):
            try:
                model = model_cls.from_pretrained(model_name, **candidate_kwargs)
                if move_device is not None:
                    model = model.to(move_device)
                model = postprocess_loaded_model(model, model_name=model_name)
                _patch_qwen3_vl_image_feature_splits(model)
                return model
            except Exception as exc:
                last_error = exc
    raise RuntimeError(
        f"Failed to load model '{model_name}'. Tried loaders: {', '.join(tried)}. "
        f"Last error: {last_error}"
    ) from last_error


def _patch_qwen3_vl_image_feature_splits(model) -> None:
    inner_model = getattr(model, "model", None)
    visual = getattr(inner_model, "visual", None)
    if inner_model is None or visual is None or not hasattr(inner_model, "get_image_features"):
        return
    if getattr(inner_model, "_ao_split_sizes_patched", False):
        return

    def patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        kwargs.pop("return_dict", None)
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_embeds = vision_output.pooler_output
        split_sizes = (
            image_grid_thw.detach().cpu().prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

    inner_model.get_image_features = types.MethodType(patched_get_image_features, inner_model)
    inner_model._ao_split_sizes_patched = True


DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% set role = message['role'] %}"
    "{% if role == 'system' %}System: {{ message['content'] }}\\n"
    "{% elif role == 'user' %}User: {{ message['content'] }}\\n"
    "{% elif role == 'assistant' %}Assistant: {{ message['content'] }}\\n"
    "{% else %}{{ role | capitalize }}: {{ message['content'] }}\\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant: {% endif %}"
)


def _tokenizer_looks_broken(tokenizer: AutoTokenizer) -> bool:
    """Heuristic: broken tokenizers often encode regular text to an empty sequence."""
    try:
        probe_ids = tokenizer.encode("hello", add_special_tokens=False)
    except Exception:
        return True
    return not isinstance(probe_ids, list) or len(probe_ids) == 0


def _iter_fallback_tokenizer_dirs(local_model_path: Path) -> list[Path]:
    """Search nearby train checkpoints for reusable tokenizer assets."""
    search_patterns = (
        "train/checkpoint-*",
        "*/train/checkpoint-*",
        "*/*/train/checkpoint-*",
        "*/*/*/train/checkpoint-*",
    )

    candidate_scores: dict[Path, tuple[int, int]] = {}
    for depth, root in enumerate(list(local_model_path.parents[:5])):
        for pattern in search_patterns:
            for candidate in root.glob(pattern):
                if not candidate.is_dir() or not (candidate / "tokenizer.json").is_file():
                    continue

                step = -1
                if candidate.name.startswith("checkpoint-"):
                    step_str = candidate.name.split("checkpoint-", 1)[1]
                    if step_str.isdigit():
                        step = int(step_str)

                # Lower depth (closer path) and larger step are preferred.
                current = candidate_scores.get(candidate)
                new_score = (depth, step)
                if current is None or new_score < current:
                    candidate_scores[candidate] = new_score

    ordered = sorted(candidate_scores.items(), key=lambda item: (item[1][0], -item[1][1], str(item[0])))
    return [path for path, _ in ordered]


def load_tokenizer(
    model_name: str,
) -> AutoTokenizer:
    print("Loading tokenizer...")
    tokenizer = _from_pretrained_with_cache_fallback(
        AutoTokenizer,
        model_name,
        label="AutoTokenizer",
    )

    model_path = Path(model_name).expanduser()
    if _tokenizer_looks_broken(tokenizer) and model_path.is_dir():
        for fallback_dir in _iter_fallback_tokenizer_dirs(model_path.resolve()):
            try:
                fallback_tokenizer = AutoTokenizer.from_pretrained(str(fallback_dir))
            except Exception:
                continue
            if _tokenizer_looks_broken(fallback_tokenizer):
                continue
            print(f"Tokenizer at {model_name} appears incomplete; using tokenizer from {fallback_dir}")
            tokenizer = fallback_tokenizer
            break

    if _tokenizer_looks_broken(tokenizer):
        raise ValueError(
            f"Failed to load a usable tokenizer from '{model_name}'. "
            "The tokenizer encodes regular text into empty token ids. "
            "Please provide a model/tokenizer directory that includes tokenizer files."
        )

    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        print("Tokenizer has no chat_template; using a default text chat template.")

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    return tokenizer


def load_processor(model_name: str):
    print("Loading processor...")
    processor = _from_pretrained_with_cache_fallback(
        AutoProcessor,
        model_name,
        label="AutoProcessor",
        trust_remote_code=True,
    )
    tokenizer = get_processor_tokenizer(processor)
    if tokenizer is None or not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(f"Processor for '{model_name}' does not expose a tokenizer.")

    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        print("Processor tokenizer has no chat_template; using a default text chat template.")
    else:
        patch_mllama_processor_chat_template(processor)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    return processor


def list_decode(x: torch.Tensor, tokenizer: AutoTokenizer) -> list[list[str]]:
    """
    Input: torch.Tensor of shape [batch_size, seq_length]
    Output: list of list of strings of len [batch_size, seq_length] Each inner list corresponds to a single token
    """
    assert len(x.shape) == 1 or len(x.shape) == 2
    # Convert to list of lists, even if x is 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it 2D for consistent handling

    # Convert tensor to list of list of ints
    token_ids = x.tolist()

    # Convert token ids to token strings
    return [tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in token_ids]


def get_bos_eos_pad_mask(tokenizer: AutoTokenizer, token_ids: torch.Tensor) -> torch.Tensor:
    """Create mask for BOS, EOS, and PAD tokens"""
    mask = torch.zeros_like(token_ids, dtype=torch.bool)

    if tokenizer.bos_token_id is not None:
        mask |= token_ids == tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        mask |= token_ids == tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        mask |= token_ids == tokenizer.pad_token_id

    return mask


def assert_no_peft_present(model, check_for_active_adapter_only=False):
    """
    Asserts that no PEFT adapters are present or active on the model.

    Args:
        model: The model to check.
        check_for_active_adapter_only (bool):
            - If False (default), asserts that NO adapters are loaded on the model at all.
            - If True, asserts only that no adapter is currently *active*.
              This allows inactive adapters to still be loaded in memory.
    """
    is_peft_model = isinstance(model, PeftModel)

    if not is_peft_model and not hasattr(model, "peft_config"):
        # If it's not a PeftModel and has no peft_config, we're 100% sure no adapters are loaded.
        return

    # At this point, the model has had PEFT adapters at some point.

    # getattr is used to safely access peft_config, which might be an empty dict.
    loaded_adapters = list(getattr(model, "peft_config", {}).keys())

    if not check_for_active_adapter_only:
        assert not loaded_adapters, (
            f"PEFT check failed! Found loaded adapters: {loaded_adapters}. "
            "Model should have no adapters loaded in memory."
        )

    # PeftModel has an `active_adapters` property which is a list of active adapter names.
    # It's an empty list when the base model is active.
    active_adapters = getattr(model, "active_adapters", [])
    assert not active_adapters, (
        f"PEFT check failed! Found active adapters: {active_adapters}. Model should be running in base mode."
    )


def get_layer_count(model_name: str) -> int:
    """Get the number of layers from a HuggingFace model config."""
    config = _from_pretrained_with_cache_fallback(
        AutoConfig,
        model_name,
        label="AutoConfig",
    )
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "text_config"):
        # Gemma-3 models store config in text_config
        return config.text_config.num_hidden_layers
    raise AttributeError(f"Could not find layer count for {model_name}")


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent to a layer number."""
    max_layers = get_layer_count(model_name)
    return int(max_layers * (layer_percent / 100))
