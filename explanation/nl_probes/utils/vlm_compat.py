from __future__ import annotations

import importlib.util
from pathlib import Path
import types
from typing import Any

import torch
from transformers import AutoConfig


FAMILY_QWEN3_VL = "qwen3_vl"
FAMILY_MLLAMA = "mllama"
FAMILY_GLM4V = "glm4v"
FAMILY_GENERIC = "generic"


def load_config(model_name: str):
    try:
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None


def prepare_model_config(model_name: str, config: Any | None = None) -> Any | None:
    config = config if config is not None else load_config(model_name)
    if config is None:
        return None

    family = detect_model_family(model_name, config=config)
    if family == FAMILY_GLM4V:
        if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
            try:
                setattr(config, "max_length", int(getattr(config, "seq_length")))
            except Exception:
                pass
        if not hasattr(config, "num_hidden_layers") and hasattr(config, "num_layers"):
            try:
                setattr(config, "num_hidden_layers", int(getattr(config, "num_layers")))
            except Exception:
                pass
        if not hasattr(config, "use_cache"):
            try:
                setattr(config, "use_cache", True)
            except Exception:
                pass
    return config


def patch_transformers_remote_code_compat() -> None:
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    existing = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if not isinstance(existing, property) and existing is not None:
        return
    if isinstance(existing, property) and existing.fset is not None:
        return

    def _all_tied_weights_keys(self) -> dict[str, None]:
        override = getattr(self, "_ao_all_tied_weights_keys_override", None)
        if override is not None:
            return override
        keys = getattr(self, "_tied_weights_keys", None) or []
        return {str(key): None for key in keys}

    def _set_all_tied_weights_keys(self, value: Any) -> None:
        setattr(self, "_ao_all_tied_weights_keys_override", value)

    try:
        setattr(
            PreTrainedModel,
            "all_tied_weights_keys",
            property(_all_tied_weights_keys, _set_all_tied_weights_keys),
        )
    except Exception:
        pass


def postprocess_loaded_model(model: Any, model_name: str | None = None) -> Any:
    family = detect_model_family(model_name, model=model, config=getattr(model, "config", None))
    if family != FAMILY_GLM4V:
        return model

    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "max_length"):
        return model

    max_length = getattr(config, "max_length", None)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and max_length is not None:
        try:
            generation_config.max_length = int(max_length)
        except Exception:
            pass

    try:
        delattr(config, "max_length")
    except Exception:
        pass

    try:
        model._supports_default_dynamic_cache = types.MethodType(lambda self: False, model)
    except Exception:
        pass
    return model


def identity_strings(
    model_name: str | None = None,
    config: Any | None = None,
    processor: Any | None = None,
    model: Any | None = None,
) -> list[str]:
    identities: list[str] = []
    if model_name:
        identities.append(str(model_name).lower())
    if config is not None:
        identities.append(str(getattr(config, "model_type", "")).lower())
        identities.append(config.__class__.__name__.lower())
        identities.extend(str(item).lower() for item in getattr(config, "architectures", []) or [])
    if processor is not None:
        identities.append(processor.__class__.__name__.lower())
    if model is not None:
        identities.append(model.__class__.__name__.lower())
        model_config = getattr(model, "config", None)
        if model_config is not None and model_config is not config:
            identities.extend(identity_strings(config=model_config))
    return identities


def _contains_identity(identities: list[str], needle: str) -> bool:
    normalized_needle = needle.lower()
    compact_needle = normalized_needle.replace("-", "").replace("_", "")
    for identity in identities:
        compact_identity = identity.replace("-", "").replace("_", "")
        if normalized_needle in identity or compact_needle in compact_identity:
            return True
    return False


def detect_model_family(
    model_name: str | None = None,
    *,
    config: Any | None = None,
    processor: Any | None = None,
    model: Any | None = None,
) -> str:
    explicit = None
    for obj in (processor, model):
        explicit = getattr(obj, "_ao_model_family", None)
        if explicit:
            return str(explicit)

    identities = identity_strings(model_name, config=config, processor=processor, model=model)
    if (
        config is not None
        and _contains_identity(identities, "chatglm")
        and hasattr(config, "vision_config")
    ):
        return FAMILY_GLM4V
    if config is not None and _contains_identity(identities, "mllama"):
        return FAMILY_MLLAMA
    if config is not None and (
        _contains_identity(identities, "qwen3-vl") or _contains_identity(identities, "qwen3_vl")
    ):
        return FAMILY_QWEN3_VL
    if (
        _contains_identity(identities, "glm-4v")
        or _contains_identity(identities, "glm4v")
        or _contains_identity(identities, "chatglm4tokenizer")
    ):
        return FAMILY_GLM4V
    if _contains_identity(identities, "mllama"):
        return FAMILY_MLLAMA
    if _contains_identity(identities, "qwen3-vl") or _contains_identity(identities, "qwen3_vl"):
        return FAMILY_QWEN3_VL
    return FAMILY_GENERIC


def attach_model_family(obj: Any, family: str) -> Any:
    try:
        setattr(obj, "_ao_model_family", family)
    except Exception:
        pass
    return obj


def processor_tokenizer(processor: Any) -> Any:
    family = detect_model_family(processor=processor)
    if family == FAMILY_GLM4V:
        return processor
    tokenizer = getattr(processor, "tokenizer", None)
    return tokenizer or processor


def get_processor_tokenizer(processor: Any) -> Any:
    return processor_tokenizer(processor)


def patch_mllama_processor_chat_template(processor: Any) -> None:
    if detect_model_family(processor=processor) != FAMILY_MLLAMA:
        return
    old = "{%- set system_message = messages[0]['content']|trim %}"
    new = (
        "{%- if messages[0]['content'] is string %}\n"
        "        {%- set system_message = messages[0]['content']|trim %}\n"
        "    {%- else %}\n"
        "        {%- set system_ns = namespace(text='') %}\n"
        "        {%- for content in messages[0]['content'] %}\n"
        "            {%- if content['type'] == 'text' %}\n"
        "                {%- set system_ns.text = system_ns.text + content['text'] %}\n"
        "            {%- endif %}\n"
        "        {%- endfor %}\n"
        "        {%- set system_message = system_ns.text|trim %}\n"
        "    {%- endif %}"
    )
    for template_owner in (processor, processor_tokenizer(processor)):
        template = getattr(template_owner, "chat_template", None)
        if isinstance(template, str) and old in template:
            template_owner.chat_template = template.replace(old, new, 1)


def text_hidden_size(config: Any) -> int | None:
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None) is not None:
        return int(text_config.hidden_size)
    hidden_size = getattr(config, "hidden_size", None)
    return int(hidden_size) if hidden_size is not None else None


def num_language_layers(config: Any) -> int | None:
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "num_hidden_layers", None) is not None:
        return int(text_config.num_hidden_layers)
    if getattr(config, "num_hidden_layers", None) is not None:
        return int(config.num_hidden_layers)
    if getattr(config, "num_layers", None) is not None:
        return int(config.num_layers)
    return None


def language_layer_path(model_name: str, layer: int, *, config: Any | None = None) -> str:
    family = detect_model_family(model_name, config=config)
    if family in {FAMILY_QWEN3_VL, FAMILY_MLLAMA}:
        return f"model.language_model.layers.{int(layer)}"
    if family == FAMILY_GLM4V:
        return f"transformer.encoder.layers.{int(layer)}"
    return f"model.layers.{int(layer)}"


def preferred_attn_implementation(model_name: str, *, config: Any | None = None) -> str:
    family = detect_model_family(model_name, config=config)
    identities = identity_strings(model_name, config=config)
    if family in {FAMILY_QWEN3_VL, FAMILY_MLLAMA}:
        return "sdpa"
    if _contains_identity(identities, "gemma"):
        return "eager"
    if importlib.util.find_spec("flash_attn") is None:
        return "sdpa"
    return "flash_attention_2"


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and str(item.get("type", "")).lower() == "text":
                parts.append(str(item.get("text", "") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _image_from_content(content: Any) -> Any | None:
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type == "image":
            return item.get("image")
        if item_type == "image_url":
            value = item.get("image_url")
            if isinstance(value, dict):
                return value.get("url")
            return value
    return None


def _load_image_if_needed(image: Any) -> Any:
    if image is None:
        return None
    if isinstance(image, (str, Path)):
        from PIL import Image

        return Image.open(str(image)).convert("RGB")
    return image


def _normalize_vlm_content(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    normalized: list[Any] = []
    for item in content:
        if isinstance(item, dict):
            normalized.append(dict(item))
        elif isinstance(item, str):
            normalized.append({"type": "text", "text": item})
        else:
            normalized.append(item)
    return normalized


def normalize_messages_for_processor(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    load_images: bool = False,
    include_images: bool = True,
) -> list[dict[str, Any]]:
    family = detect_model_family(processor=processor)
    if family != FAMILY_GLM4V:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            item["content"] = _normalize_vlm_content(message.get("content"))
            normalized.append(item)
        return normalized

    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        item: dict[str, Any] = {
            "role": role,
            "content": _text_from_content(content),
        }
        metadata = message.get("metadata")
        if metadata is not None:
            item["metadata"] = metadata
        image = _image_from_content(content)
        if include_images and image is not None:
            item["image"] = _load_image_if_needed(image) if load_images else image
        normalized.append(item)
    return normalized


def apply_chat_template_text(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    family = detect_model_family(processor=tokenizer)
    normalized = (
        normalize_messages_for_processor(tokenizer, messages, load_images=False, include_images=False)
        if family == FAMILY_GLM4V
        else messages
    )
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except TypeError:
            return tokenizer.apply_chat_template(normalized, tokenize=False)
    return _text_from_content(normalized[-1].get("content") if normalized else "")


def _special_id_to_token(tokenizer: Any) -> dict[int, str]:
    mapping = getattr(tokenizer, "added_tokens_encoder", None)
    if not isinstance(mapping, dict):
        return {}
    return {int(token_id): str(token) for token, token_id in mapping.items()}


def _decode_token(tokenizer: Any, token_id: int, special_by_id: dict[int, str]) -> str:
    if token_id in special_by_id:
        return special_by_id[token_id]
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([token_id])
    except Exception:
        return ""


def infer_offsets_from_decoded_tokens(tokenizer: Any, text: str, input_ids: list[int]) -> list[tuple[int, int]]:
    special_by_id = _special_id_to_token(tokenizer)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in input_ids:
        token_text = _decode_token(tokenizer, int(token_id), special_by_id)
        if not token_text:
            offsets.append((cursor, cursor))
            continue
        index = text.find(token_text, cursor)
        if index < 0:
            stripped = token_text.lstrip()
            index = text.find(stripped, cursor) if stripped else -1
            if index >= 0:
                token_text = stripped
        if index < 0:
            offsets.append((cursor, cursor))
            continue
        end = index + len(token_text)
        offsets.append((index, end))
        cursor = end
    return offsets


def tokenize_text_with_offsets(
    tokenizer: Any,
    text: str,
    *,
    max_length: int | None,
) -> dict[str, Any]:
    family = detect_model_family(processor=tokenizer)
    tok_kwargs = dict(
        truncation=True if max_length else False,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
        add_special_tokens=False if family == FAMILY_GLM4V else True,
    )
    if family == FAMILY_GLM4V:
        encoded = tokenizer(text, return_tensors="pt", **tok_kwargs)
        input_ids = encoded["input_ids"][0].tolist()
        encoded["offset_mapping"] = torch.tensor(
            infer_offsets_from_decoded_tokens(tokenizer, text, input_ids),
            dtype=torch.long,
        ).unsqueeze(0)
        return encoded

    try:
        return tokenizer(text, return_offsets_mapping=True, return_tensors="pt", **tok_kwargs)
    except Exception:
        encoded = tokenizer(text, return_tensors="pt", **tok_kwargs)
        input_ids = encoded["input_ids"][0].tolist()
        encoded["offset_mapping"] = torch.tensor(
            infer_offsets_from_decoded_tokens(tokenizer, text, input_ids),
            dtype=torch.long,
        ).unsqueeze(0)
        return encoded


def tokenize_text_ids(tokenizer: Any, text: str, *, max_length: int | None) -> list[int]:
    encoded = tokenize_text_with_offsets(tokenizer, text, max_length=max_length)
    return encoded["input_ids"][0].tolist()


def _glm_image_transform(processor: Any, image: Any):
    from torchvision import transforms

    image_size = int(getattr(processor, "image_size", None) or 1120)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    return transform(_load_image_if_needed(image))


def _encode_glm4v_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    return_tensors: str | None,
    padding: bool,
    max_length: int | None = None,
) -> dict[str, Any]:
    if padding:
        raise ValueError("GLM-4V compatibility encoder expects per-sample padding=False.")

    normalized = normalize_messages_for_processor(processor, messages, load_images=True, include_images=True)
    input_ids = list(processor.get_prefix_tokens())
    image_tensor = None
    for message in normalized:
        role = message.get("role", "user")
        metadata = message.get("metadata", "")
        content = str(message.get("content", "") or "")
        message_prefix = None
        if message.get("image") is not None:
            if image_tensor is not None:
                raise ValueError("GLM-4V compatibility path supports one image per sample.")
            image_tensor = _glm_image_transform(processor, message["image"])
            message_prefix = processor.convert_tokens_to_ids(
                ["<|begin_of_image|>", "<|endoftext|>", "<|end_of_image|>"]
            )
        if content or message_prefix:
            input_ids.extend(
                processor.build_single_message(
                    role,
                    metadata,
                    content,
                    tokenize=True,
                    message_prefix=message_prefix,
                )
            )
    if add_generation_prompt:
        input_ids.append(processor.convert_tokens_to_ids("<|assistant|>"))
    if max_length is not None and max_length > 0:
        input_ids = input_ids[:max_length]

    attention_mask = [1] * len(input_ids)
    position_ids = list(range(len(input_ids)))
    if return_tensors == "pt":
        output: dict[str, Any] = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "position_ids": torch.tensor([position_ids], dtype=torch.long),
        }
        if image_tensor is not None:
            output["images"] = image_tensor.unsqueeze(0)
        return output

    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if image_tensor is not None:
        output["images"] = image_tensor.unsqueeze(0)
    return output


def encode_chat_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    return_tensors: str | None,
    padding: bool,
    image_max_pixels: int | None,
    return_mm_token_type_ids: bool = False,
    max_length: int | None = None,
) -> dict[str, Any]:
    family = detect_model_family(processor=processor)
    if family == FAMILY_GLM4V:
        return _encode_glm4v_messages(
            processor,
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
        )

    normalized = normalize_messages_for_processor(processor, messages, load_images=False, include_images=True)
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors=return_tensors,
        padding=padding,
    )
    if image_max_pixels is not None:
        kwargs["max_pixels"] = image_max_pixels
    if return_mm_token_type_ids:
        kwargs["return_mm_token_type_ids"] = True
    try:
        return processor.apply_chat_template(normalized, **kwargs)
    except TypeError:
        kwargs.pop("return_mm_token_type_ids", None)
        try:
            return processor.apply_chat_template(normalized, **kwargs)
        except TypeError:
            kwargs.pop("max_pixels", None)
            return processor.apply_chat_template(normalized, **kwargs)


def apply_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    return_dict: bool = False,
    return_tensors: str | None = None,
    padding: bool = False,
    max_pixels: int | None = None,
    return_mm_token_type_ids: bool = False,
    max_length: int | None = None,
) -> Any:
    if not tokenize:
        return apply_chat_template_text(
            processor,
            messages,
            add_generation_prompt=add_generation_prompt,
        )

    encoded = encode_chat_messages(
        processor,
        messages,
        add_generation_prompt=add_generation_prompt,
        return_tensors=return_tensors,
        padding=padding,
        image_max_pixels=max_pixels,
        return_mm_token_type_ids=return_mm_token_type_ids,
        max_length=max_length,
    )
    if return_dict:
        return encoded
    return encoded.get("input_ids")


def tensor_inputs_to_device(encoded: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model_inputs: dict[str, Any] = {}
    for key, value in encoded.items():
        if not isinstance(value, torch.Tensor):
            model_inputs[key] = value
            continue
        if key == "image_grid_thw":
            model_inputs[key] = value.to(device=device, dtype=torch.int32)
        else:
            model_inputs[key] = value.to(device)
    return model_inputs


def adjust_glm4v_positions(
    *,
    config: Any,
    input_ids: torch.Tensor,
    positions: list[list[int]],
    has_images: bool,
) -> list[list[int]]:
    if not has_images:
        return positions
    vision_config = getattr(config, "vision_config", {}) or {}
    image_size = int(vision_config.get("image_size", 1120))
    patch_size = int(vision_config.get("patch_size", 14))
    num_patches = (image_size // patch_size // 2) ** 2
    boi_token_id = int(getattr(config, "boi_token_id"))
    eoi_token_id = int(getattr(config, "eoi_token_id"))

    adjusted: list[list[int]] = []
    for sample_ids, sample_positions in zip(input_ids.tolist(), positions, strict=True):
        if boi_token_id not in sample_ids or eoi_token_id not in sample_ids:
            adjusted.append(list(sample_positions))
            continue
        boi_pos = sample_ids.index(boi_token_id)
        eoi_pos = sample_ids.index(eoi_token_id)
        shift = num_patches - 3
        adjusted.append([int(pos) + shift if int(pos) > eoi_pos else int(pos) for pos in sample_positions])
    return adjusted


def adjust_positions_for_model(model: Any, model_inputs: dict[str, Any], positions: list[list[int]]) -> list[list[int]]:
    family = detect_model_family(model=model, config=getattr(model, "config", None))
    if family != FAMILY_GLM4V:
        return positions
    images = model_inputs.get("images")
    has_images = isinstance(images, torch.Tensor) and images.numel() > 0
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return positions
    return adjust_glm4v_positions(
        config=model.config,
        input_ids=input_ids,
        positions=positions,
        has_images=has_images,
    )


def normalize_hook_tensor_to_bld(tensor: torch.Tensor, *, layer_key: str | None = None, batch_size: int | None = None) -> torch.Tensor:
    if tensor.ndim != 3:
        return tensor
    if batch_size is not None:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[1] == batch_size and tensor.shape[0] != batch_size:
            return tensor.transpose(0, 1).contiguous()
        return tensor

    if tensor.shape[0] > tensor.shape[1] and tensor.shape[1] <= 64:
        return tensor.transpose(0, 1).contiguous()
    return tensor
