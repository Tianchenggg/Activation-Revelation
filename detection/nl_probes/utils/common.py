import contextlib
import importlib.util
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from transformers.dynamic_module_utils import custom_object_save, get_class_from_dynamic_module


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


def load_model_config(model_name: str):
    config = _from_pretrained_with_cache_fallback(
        AutoConfig,
        model_name,
        label="AutoConfig",
        trust_remote_code=True,
    )
    if getattr(config, "model_type", "") == "chatglm":
        if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
            config.max_length = config.seq_length
        if not hasattr(config, "use_cache"):
            config.use_cache = True
    return config


@contextlib.contextmanager
def temporarily_disable_generation_parameter_validation(config: Any):
    """
    HF 5.x refuses to save configs that carry generation parameters on the config object.
    Some custom model families (for example ChatGLM) still require fields like `max_length`
    on `config` during model construction, so we suppress the save-time validation without
    stripping those fields from the serialized config.
    """

    if config is None:
        yield
        return

    method_name = "_get_generation_parameters"
    config_cls = config.__class__
    original_method = getattr(config_cls, method_name, None)

    if original_method is None:
        yield
        return

    setattr(config_cls, method_name, lambda self: {})
    try:
        yield
    finally:
        setattr(config_cls, method_name, original_method)


def maybe_save_dynamic_model_code(model: Any, output_dir: str | Path) -> list[Path]:
    """
    Persist custom trust_remote_code modeling files next to a saved checkpoint so the
    checkpoint can be reloaded from the local output directory.
    """

    module_name = str(getattr(model.__class__, "__module__", "") or "")
    if not module_name.startswith("transformers_modules."):
        return []
    return [Path(path) for path in custom_object_save(model, str(output_dir), config=None)]


def infer_model_family(model_name: str, config: Any | None = None) -> str:
    if config is None:
        config = load_model_config(model_name)

    model_name_lower = model_name.lower()
    model_type = str(getattr(config, "model_type", "") or "").lower()
    architectures = [str(arch).lower() for arch in getattr(config, "architectures", []) or []]

    if "qwen3-vl" in model_name_lower or model_type == "qwen3_vl" or any("qwen3vl" in arch for arch in architectures):
        return "qwen3_vl"
    if model_type == "chatglm" or any("chatglm" in arch for arch in architectures):
        return "chatglm"
    if model_type == "mllama" or any("mllama" in arch for arch in architectures):
        return "mllama"
    if "qwen3" in model_name_lower or model_type == "qwen3" or any("qwen3" in arch for arch in architectures):
        return "qwen3"
    if model_type == "llama" or any("llama" in arch for arch in architectures):
        return "llama"
    if "gemma" in model_name_lower or model_type.startswith("gemma"):
        return "gemma"
    return "generic"


def choose_attention_implementation(model_name: str, config: Any | None = None) -> str:
    family = infer_model_family(model_name, config=config)
    if family in {"qwen3_vl", "chatglm", "mllama"}:
        return "sdpa"
    if family == "gemma":
        return "eager"
    if importlib.util.find_spec("flash_attn") is None:
        return "sdpa"
    return "flash_attention_2"


def load_dynamic_auto_class(
    model_name: str,
    config: Any,
    auto_map_key: str,
):
    auto_map = getattr(config, "auto_map", None)
    if not isinstance(auto_map, dict):
        return None
    class_ref = auto_map.get(auto_map_key)
    if not isinstance(class_ref, str) or not class_ref.strip():
        return None
    return get_class_from_dynamic_module(class_ref, model_name)


def _config_layer_count(config: Any) -> int:
    candidates = [
        getattr(config, "num_hidden_layers", None),
        getattr(config, "num_layers", None),
    ]

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        candidates.extend(
            [
                getattr(text_config, "num_hidden_layers", None),
                getattr(text_config, "num_layers", None),
            ]
        )
        if isinstance(text_config, dict):
            candidates.extend(
                [
                    text_config.get("num_hidden_layers"),
                    text_config.get("num_layers"),
                ]
            )

    for value in candidates:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed

    raise AttributeError(f"Could not find layer count in config type {type(config).__name__}")


def get_candidate_layer_stack_paths(
    model_name: str,
    *,
    use_lora: bool = False,
    config: Any | None = None,
) -> list[str]:
    family = infer_model_family(model_name, config=config)

    if family == "qwen3_vl":
        base_paths = [
            "model.language_model.layers",
            "language_model.layers",
            "model.layers",
        ]
    elif family == "mllama":
        base_paths = [
            "model.language_model.layers",
            "language_model.layers",
            "model.layers",
        ]
    elif family == "chatglm":
        base_paths = [
            "transformer.encoder.layers",
            "model.transformer.encoder.layers",
        ]
    else:
        base_paths = [
            "model.layers",
            "model.language_model.layers",
            "language_model.layers",
            "gpt_neox.layers",
        ]

    fallback_paths = [
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "transformer.encoder.layers",
        "model.transformer.encoder.layers",
        "gpt_neox.layers",
    ]

    ordered: list[str] = []
    seen: set[str] = set()
    for path in [*base_paths, *fallback_paths]:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)

    if not use_lora:
        return ordered

    prefixed: list[str] = []
    for path in ordered:
        prefixed.append(f"base_model.model.{path}")
        prefixed.append(f"base_model.{path}")

    lora_ordered: list[str] = []
    seen.clear()
    for path in [*prefixed, *ordered]:
        if path in seen:
            continue
        seen.add(path)
        lora_ordered.append(path)
    return lora_ordered


def load_model(
    model_name: str,
    dtype: torch.dtype,
    **model_kwargs,
) -> AutoModelForCausalLM:
    print("Loading model...")

    config = load_model_config(model_name)
    family = infer_model_family(model_name, config=config)
    attn = choose_attention_implementation(model_name, config=config)

    kwargs: dict = {
        "device_map": "auto",
        "attn_implementation": attn,
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "config": config,
        **model_kwargs,
    }
    print(f"Using attention implementation: {kwargs['attn_implementation']}")

    if family == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    elif family == "chatglm":
        model_cls = load_dynamic_auto_class(model_name, config, "AutoModelForCausalLM")
        if model_cls is None:
            model_cls = AutoModelForCausalLM
        if not hasattr(model_cls, "all_tied_weights_keys"):
            model_cls.all_tied_weights_keys = {}
        if getattr(model_cls, "_tp_plan", None) is None:
            model_cls._tp_plan = {}
        if getattr(model_cls, "_pp_plan", None) is None:
            model_cls._pp_plan = {}
        direct_kwargs = dict(kwargs)
        direct_kwargs.pop("trust_remote_code", None)
        model = model_cls.from_pretrained(model_name, **direct_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model


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
        trust_remote_code=True,
    )

    model_path = Path(model_name).expanduser()
    if _tokenizer_looks_broken(tokenizer) and model_path.is_dir():
        for fallback_dir in _iter_fallback_tokenizer_dirs(model_path.resolve()):
            try:
                fallback_tokenizer = AutoTokenizer.from_pretrained(
                    str(fallback_dir),
                    trust_remote_code=True,
                )
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
    config = load_model_config(model_name)
    return _config_layer_count(config)


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent to a layer number."""
    max_layers = get_layer_count(model_name)
    layer = int(max_layers * (layer_percent / 100))
    return min(max(layer, 0), max_layers - 1)
