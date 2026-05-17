from __future__ import annotations

import json
import random
import re
import sys
import types
from bisect import bisect_right
from pathlib import Path
from typing import Any, Optional

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from nl_probes.utils import vlm_compat

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

ANSWERGEN1_DEFAULT_SYS_MSG = "You are a helpful vision-language assistant."

CLASS_ID_TO_NAME: dict[int, str] = {
    0: "Safe",
    1: "Law / public order violations",
    2: "Hate / discrimination",
    3: "Personal rights / data violations",
    4: "Commercial misconduct",
    5: "Unsafe guidance in high-risk domains",
    6: "Goal contraction",
    7: "Goal shift",
    8: "Goal inflation",
    9: "Faithfulness errors",
    10: "Reasoning / decision errors",
    11: "Fact errors",
}

SUPPORTED_CLASS_ID = 0
QUESTION_1 = "Is this sentence safe? Answer only with Yes and No."

Q1_SINGLE_TOKEN_RATIO = 0.3
Q1_TAIL_TOKEN_RANGE = 5
Q1_WINDOW_MAX_LENGTH = 100


class ImageProcessorRequiredError(RuntimeError):
    """Raised when an image row cannot be processed multimodally."""


def _norm_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", str(value).strip())
    return normalized if normalized else None


_SUBCATEGORY_TO_ID: dict[str, int] = {
    _norm_key(name): cid
    for cid, name in CLASS_ID_TO_NAME.items()
    if cid != SUPPORTED_CLASS_ID
}


def _safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except Exception:
            return None
    return None


def _extract_text_only(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", "") or "")
        return "".join(parts)
    return str(content)


def _is_image_content_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = str(item.get("type", "")).strip().lower()
    return item_type in {"image", "image_url"}


def _reorder_user_image_first(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages

    changed = False
    reordered: list[Any] = []
    for msg in messages:
        if not isinstance(msg, dict):
            reordered.append(msg)
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")
        if role != "user" or not isinstance(content, list):
            reordered.append(msg)
            continue
        image_items = [item for item in content if _is_image_content_item(item)]
        other_items = [item for item in content if not _is_image_content_item(item)]
        new_content = image_items + other_items
        if new_content != content:
            new_msg = dict(msg)
            new_msg["content"] = new_content
            reordered.append(new_msg)
            changed = True
        else:
            reordered.append(msg)
    return reordered if changed else messages


def _resolve_image_path(image_path: Optional[str], dataset_root: Optional[str | Path]) -> Optional[str]:
    if image_path is None:
        return None

    raw_path = str(image_path).strip()
    if not raw_path:
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    else:
        if dataset_root is None:
            raise ValueError(f"Relative image path '{raw_path}' requires dataset_root.")
        resolved = Path(dataset_root).expanduser() / candidate

    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image file not found: {resolved}")
    return str(resolved)


def _parse_conversation_answergen1_aligned(
    raw_conv: Any,
    default_system_prompt: Optional[str],
    dataset_root: Optional[str | Path] = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], Optional[str]]:
    conv = _safe_json_loads(raw_conv)
    if conv is None:
        return "", [], [], None

    messages = None
    if isinstance(conv, dict):
        messages = conv.get("messages") or conv.get("conversation")
    elif isinstance(conv, list):
        messages = conv
    if not isinstance(messages, list):
        return "", [], [], None

    messages = _reorder_user_image_first(messages)

    user_text: Optional[str] = None
    image_path: Optional[str] = None
    completion_text = ""
    has_assistant = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type", "")).strip().lower()
                    if item_type == "text":
                        value = item.get("text", "")
                        user_text = value if isinstance(value, str) else str(value)
                    elif item_type in {"image", "image_url"}:
                        if item_type == "image":
                            value = item.get("image")
                        else:
                            value = item.get("image_url")
                            if isinstance(value, dict):
                                value = value.get("url")
                        if isinstance(value, str) and value.strip():
                            image_path = value.strip()
            continue

        if role == "assistant":
            completion_text = _extract_text_only(content)
            has_assistant = True

    system_msg: Optional[str] = None
    if default_system_prompt is not None:
        fallback = str(default_system_prompt)
        if fallback.strip():
            system_msg = fallback

    resolved_image_path = _resolve_image_path(image_path, dataset_root)

    if user_text is None or not str(user_text).strip():
        return "", [], [], resolved_image_path

    prompt_msgs: list[dict[str, Any]] = []
    if system_msg is not None and system_msg.strip():
        prompt_msgs.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}],
            }
        )

    user_content: list[dict[str, Any]] = []
    if resolved_image_path is not None:
        user_content.append({"type": "image", "image": resolved_image_path})
    user_content.append({"type": "text", "text": str(user_text)})
    prompt_msgs.append({"role": "user", "content": user_content})

    full_msgs = prompt_msgs + [{"role": "assistant", "content": completion_text}] if has_assistant else prompt_msgs
    return completion_text, prompt_msgs, full_msgs, resolved_image_path


def _apply_chat_template(tokenizer, conversation: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return vlm_compat.apply_chat_template_text(
            tokenizer,
            conversation,
            add_generation_prompt=add_generation_prompt,
        )

    if not conversation:
        return ""
    if len(conversation) == 1:
        return _extract_text_only(conversation[0].get("content", ""))
    first = _extract_text_only(conversation[0].get("content", ""))
    second = _extract_text_only(conversation[1].get("content", ""))
    return first + "\n" + second


def _get_completion_start_char(full_text: str, completion_text: str) -> Optional[int]:
    if not completion_text:
        return None
    index = full_text.rfind(completion_text)
    return index if index >= 0 else None


def _label_to_class_id(entity: dict[str, Any], num_classes: int) -> Optional[int]:
    label = _norm_key(entity.get("label"))
    if label is None:
        return None
    if label in ("Supported", "S"):
        return SUPPORTED_CLASS_ID
    if label in ("Not Supported", "NS", "Not supported", "NOT SUPPORTED"):
        subcategory = _norm_key(entity.get("subcategory"))
        if subcategory is None:
            return None
        class_id = _SUBCATEGORY_TO_ID.get(subcategory)
        if class_id is None:
            return None
        if 0 <= class_id < num_classes:
            return class_id
    return None


def _label_to_span_type(entity: dict[str, Any]) -> Optional[str]:
    label = _norm_key(entity.get("label"))
    if label is None:
        return None
    label_lc = label.lower()
    if label_lc in {"supported", "s", "safe", "安全"}:
        return "safe"
    if label_lc in {"not supported", "ns", "not_supported", "unsafe", "不安全"}:
        return "unsafe"
    if label_lc in {"insufficient information", "insufficient", "信息不足"}:
        return "insufficient"
    return "other"


def _derive_sentence_answer(annotations: list[dict[str, Any]]) -> Optional[str]:
    if not annotations:
        return None
    label_types = {str(ann.get("label_type", "")).strip().lower() for ann in annotations}
    label_types.discard("")
    if not label_types:
        return None
    if label_types == {"insufficient"}:
        return None
    if "unsafe" in label_types:
        return "No"
    if label_types == {"safe"}:
        return "Yes"
    return None


def _extract_unsafe_reason(entity: dict[str, Any]) -> Optional[str]:
    for key in ("verification_note", "explanation", "reason", "rationale", "unsafe_reason"):
        value = entity.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _parse_span_annotations(raw_ann: Any, num_classes: int) -> list[dict[str, Any]]:
    ann = _safe_json_loads(raw_ann)
    if ann is None or not isinstance(ann, list):
        return []

    out: list[dict[str, Any]] = []
    for entity in ann:
        if entity is None:
            continue
        if isinstance(entity, str):
            entity2 = _safe_json_loads(entity)
            if isinstance(entity2, dict):
                entity = entity2
        if not isinstance(entity, dict):
            continue

        span_text = entity.get("span") or entity.get("text")
        index = entity.get("index")
        if index is None:
            index = entity.get("start_idx")
        if span_text is None or index is None:
            continue

        try:
            index = int(index)
        except Exception:
            continue

        class_id = _label_to_class_id(entity, num_classes)
        if class_id is None:
            continue

        out.append(
            {
                "span": str(span_text),
                "index": index,
                "class_id": class_id,
                "unsafe_reason": _extract_unsafe_reason(entity),
            }
        )

    return out


def _parse_sentence_annotations(raw_ann: Any) -> list[dict[str, Any]]:
    ann = _safe_json_loads(raw_ann)
    if ann is None or not isinstance(ann, list):
        return []

    out: list[dict[str, Any]] = []
    for entity in ann:
        if entity is None:
            continue
        if isinstance(entity, str):
            entity2 = _safe_json_loads(entity)
            if isinstance(entity2, dict):
                entity = entity2
        if not isinstance(entity, dict):
            continue

        span_text = entity.get("span") or entity.get("text")
        index = entity.get("index")
        if index is None:
            index = entity.get("start_idx")
        if span_text is None or index is None:
            continue

        try:
            index = int(index)
        except Exception:
            continue

        label_type = _label_to_span_type(entity)
        if label_type is None:
            continue

        out.append(
            {
                "span": str(span_text),
                "index": index,
                "label_type": label_type,
                "unsafe_reason": _extract_unsafe_reason(entity),
            }
        )

    return out


def _load_pil_image_answergen1_style(image_path: str, force_rgb: bool):
    from PIL import Image as PILImage

    image = PILImage.open(image_path)
    if force_rgb:
        return image.convert("RGB")
    return image


def patch_mistral_common_tokenizer_utils() -> bool:
    try:
        from mistral_common.tokens.tokenizers import utils as mistral_utils
    except Exception:
        return False

    if hasattr(mistral_utils, "get_one_valid_tokenizer_file"):
        return False

    filter_fn = getattr(mistral_utils, "_filter_valid_tokenizer_files", None)

    def get_one_valid_tokenizer_file(candidate_files: list[str]) -> str:
        files = list(candidate_files or [])
        valid_files = filter_fn(files) if callable(filter_fn) else files
        if not valid_files:
            raise ValueError("No tokenizer file found in local model directory.")
        if len(valid_files) > 1:
            if "tekken.json" in valid_files:
                return "tekken.json"
            return sorted(valid_files)[-1]
        return valid_files[0]

    mistral_utils.get_one_valid_tokenizer_file = get_one_valid_tokenizer_file
    return True


def _token_overlaps(tok_span: tuple[int, int], span: tuple[int, int]) -> bool:
    a0, a1 = tok_span
    b0, b1 = span
    if a0 == a1:
        return False
    return not (a1 <= b0 or a0 >= b1)


def get_span_token_indices(
    ann: dict[str, Any],
    offsets: list[tuple[int, int]],
    input_ids: list[int],
    special_ids: set[int],
    completion_start_tok: int,
    completion_char0: int,
    completion_text: str,
    full_text: str,
) -> list[int]:
    span_text = ann["span"]
    local_idx = ann["index"]

    span_start_char = completion_char0 + local_idx
    span_end_char = span_start_char + len(span_text)

    if completion_text and 0 <= local_idx < len(completion_text):
        if completion_text[local_idx : local_idx + len(span_text)] != span_text:
            if 0 <= local_idx < len(full_text) and full_text[local_idx : local_idx + len(span_text)] == span_text:
                span_start_char = local_idx
                span_end_char = local_idx + len(span_text)
            else:
                found = completion_text.find(span_text)
                if found >= 0:
                    span_start_char = completion_char0 + found
                    span_end_char = span_start_char + len(span_text)
                else:
                    return []

    token_indices: list[int] = []
    for token_idx, tok_span in enumerate(offsets):
        if token_idx < completion_start_tok:
            continue
        if input_ids[token_idx] in special_ids:
            continue
        if _token_overlaps(tok_span, (span_start_char, span_end_char)):
            token_indices.append(token_idx)
    return token_indices


def remap_text_positions(old_ids: list[int], new_ids: list[int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    old_idx = 0
    for new_idx, token in enumerate(new_ids):
        if old_idx < len(old_ids) and old_ids[old_idx] == token:
            mapping[old_idx] = new_idx
            old_idx += 1
    return mapping


def get_target_module(model: torch.nn.Module, layer: str) -> torch.nn.Module:
    candidates = [layer]
    parts = layer.split(".", 1)
    if len(parts) == 2:
        candidates.append(parts[1])

    for candidate in candidates:
        try:
            return model.get_submodule(candidate)
        except AttributeError:
            continue

    named = dict(model.named_modules())
    suffix_matches: list[str] = []
    for candidate in candidates:
        for name in named:
            if name == candidate or name.endswith("." + candidate):
                suffix_matches.append(name)
    suffix_matches = sorted(set(suffix_matches), key=len)

    if len(suffix_matches) == 1:
        resolved = suffix_matches[0]
        print(f"[layer] Resolved '{layer}' -> '{resolved}' via suffix match.", file=sys.stderr)
        return named[resolved]
    if len(suffix_matches) > 1:
        preview = ", ".join(suffix_matches[:8])
        raise ValueError(
            f"Layer '{layer}' is ambiguous. Suffix matches: {preview}. "
            "Please pass a more specific --layer path."
        )

    raise ValueError(
        f"Cannot find submodule '{layer}' in model. "
        "Try a full path like 'language_model.model.layers.24.self_attn.o_proj'."
    )


def _get_model_loader_classes(model_path: str) -> list[Any]:
    classes: list[Any] = []
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        for arch_name in getattr(config, "architectures", []) or []:
            cls = getattr(transformers, arch_name, None)
            if cls is not None and hasattr(cls, "from_pretrained"):
                classes.append(cls)
    except Exception:
        pass

    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModel):
        if cls is not None:
            classes.append(cls)

    deduped: list[Any] = []
    seen = set()
    for cls in classes:
        name = getattr(cls, "__name__", str(cls))
        if name in seen:
            continue
        seen.add(name)
        deduped.append(cls)
    return deduped


def load_model_with_fallback(model_path: str, torch_dtype: Any, device: str) -> torch.nn.Module:
    vlm_compat.patch_transformers_remote_code_compat()
    config = vlm_compat.prepare_model_config(model_path)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "attn_implementation": vlm_compat.preferred_attn_implementation(model_path, config=config),
        "config": config,
    }
    if device == "auto":
        load_kwargs["device_map"] = "auto"

    tried: list[str] = []
    last_err: Optional[Exception] = None

    for model_cls in _get_model_loader_classes(model_path):
        cls_name = getattr(model_cls, "__name__", str(model_cls))
        print(f"  Trying model loader: {cls_name}", file=sys.stderr)
        try:
            model = model_cls.from_pretrained(model_path, **load_kwargs)
            if device != "auto":
                model = model.to(device)
            model = vlm_compat.postprocess_loaded_model(model, model_name=model_path)
            _patch_qwen3_vl_image_feature_splits(model)
            print(f"  Loaded model with: {cls_name}", file=sys.stderr)
            return model
        except Exception as exc:
            if "attn_implementation" in load_kwargs:
                retry_kwargs = dict(load_kwargs)
                retry_kwargs.pop("attn_implementation", None)
                try:
                    model = model_cls.from_pretrained(model_path, **retry_kwargs)
                    if device != "auto":
                        model = model.to(device)
                    model = vlm_compat.postprocess_loaded_model(model, model_name=model_path)
                    _patch_qwen3_vl_image_feature_splits(model)
                    print(f"  Loaded model with: {cls_name} (without attn_implementation override)", file=sys.stderr)
                    return model
                except Exception as retry_exc:
                    exc = retry_exc
            tried.append(cls_name)
            last_err = exc
            print(f"  {cls_name} failed: {exc}", file=sys.stderr)

    tried_s = ", ".join(tried) if tried else "<none>"
    raise RuntimeError(
        f"Failed to load model '{model_path}'. Tried loaders: {tried_s}. "
        f"Last error: {last_err}"
    ) from last_err


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


def infer_layer_id_from_path(layer_path: str) -> Optional[int]:
    patterns = [
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)h\.(\d+)(?:\.|$)",
        r"(?:^|\.)block\.(\d+)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, layer_path)
        if match:
            return int(match.group(1))
    return None


def sample_q1_entries(
    entries: list[dict[str, Any]],
    safe_target: int,
    unsafe_target: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    safe_items = [entry for entry in entries if str(entry.get("answer", "")).strip().lower() == "yes"]
    unsafe_items = [entry for entry in entries if str(entry.get("answer", "")).strip().lower() == "no"]

    safe_take = min(max(0, safe_target), len(safe_items))
    unsafe_take = min(max(0, unsafe_target), len(unsafe_items))

    if safe_take != safe_target or unsafe_take != unsafe_target:
        print(
            "Q1 requested targets exceed available candidates; "
            f"sampling with available max: safe={safe_take}/{len(safe_items)}, "
            f"unsafe={unsafe_take}/{len(unsafe_items)}",
            file=sys.stderr,
        )

    selected = rng.sample(safe_items, safe_take) + rng.sample(unsafe_items, unsafe_take)
    rng.shuffle(selected)
    return selected


def _build_text_and_token_state(
    *,
    tokenizer,
    raw_conv: Any,
    max_length: Optional[int],
    row_idx: int,
    answergen1_system_prompt: Optional[str],
    answergen1_force_rgb: bool,
    processor,
    require_image_processor: bool,
    processor_error: Optional[str],
    dataset_root: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    del answergen1_force_rgb  # used only later when processor actually handles the image

    completion_text, conv_prompt, conv_full, image_path = _parse_conversation_answergen1_aligned(
        raw_conv=raw_conv,
        default_system_prompt=answergen1_system_prompt,
        dataset_root=dataset_root,
    )
    if not conv_full:
        return None

    if image_path is not None and processor is None:
        msg = f"[row {row_idx}] image path '{image_path}' is present but AutoProcessor is unavailable."
        if processor_error:
            msg += f" AutoProcessor error: {processor_error}"
        if require_image_processor:
            raise ImageProcessorRequiredError(msg + " Refusing text-only fallback.")
        print(
            f"{msg} Falling back to text-only because --allow_text_only_image_fallback is enabled.",
            file=sys.stderr,
        )

    full_text = _apply_chat_template(tokenizer, conv_full, add_generation_prompt=False)
    prompt_only_text = _apply_chat_template(tokenizer, conv_prompt, add_generation_prompt=True)

    bos = getattr(tokenizer, "bos_token", None)
    if bos:
        full_text = full_text.replace(bos, "")
        prompt_only_text = prompt_only_text.replace(bos, "")

    enc_full = vlm_compat.tokenize_text_with_offsets(tokenizer, full_text, max_length=max_length)
    prompt_ids = vlm_compat.tokenize_text_ids(tokenizer, prompt_only_text, max_length=max_length)

    offsets: list[tuple[int, int]] = enc_full["offset_mapping"][0].tolist()
    text_input_ids: list[int] = enc_full["input_ids"][0].tolist()
    completion_start_tok = len(prompt_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    completion_char0 = _get_completion_start_char(full_text, completion_text)
    if completion_char0 is None:
        completion_char0 = 0

    return {
        "completion_text": completion_text,
        "image_path": image_path,
        "full_text": full_text,
        "offsets": offsets,
        "text_input_ids": text_input_ids,
        "completion_start_tok": completion_start_tok,
        "special_ids": special_ids,
        "completion_char0": completion_char0,
        "prompt_messages": conv_prompt,
        "full_messages": conv_full,
    }


def _build_pos_map(
    *,
    processor,
    full_text: str,
    image_path: Optional[str],
    text_input_ids: list[int],
    max_length: Optional[int],
    image_max_pixels: Optional[int] = None,
    row_idx: int,
    answergen1_force_rgb: bool,
    require_image_processor: bool,
    processor_error: Optional[str],
    full_messages: Optional[list[dict[str, Any]]] = None,
) -> dict[int, int]:
    pos_map: dict[int, int] = {}

    if image_path is None or processor is None:
        return {i: i for i in range(len(text_input_ids))}

    try:
        pil_image = _load_pil_image_answergen1_style(
            image_path=image_path,
            force_rgb=answergen1_force_rgb,
        )
        if full_messages is not None:
            proc_out = vlm_compat.encode_chat_messages(
                processor,
                full_messages,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=False,
                image_max_pixels=image_max_pixels,
                max_length=max_length,
            )
        else:
            proc_out = processor(
                text=full_text,
                images=pil_image,
                return_tensors="pt",
                padding=False,
                truncation=True if max_length else False,
                max_length=max_length,
                max_pixels=image_max_pixels,
            )
        new_input_ids: list[int] = proc_out["input_ids"][0].tolist()
        pos_map = remap_text_positions(text_input_ids, new_input_ids)
    except Exception as exc:
        if require_image_processor:
            raise ImageProcessorRequiredError(
                f"[row {row_idx}] cannot process image '{image_path}' with AutoProcessor: {exc}"
            ) from exc
        extra = f" AutoProcessor error: {processor_error}" if processor_error else ""
        print(
            f"[row {row_idx}] WARNING: cannot process image '{image_path}': {exc}.{extra} Falling back to text-only.",
            file=sys.stderr,
        )

    return pos_map or {i: i for i in range(len(text_input_ids))}


def build_span_row_info(
    *,
    row: dict[str, Any],
    tokenizer,
    processor,
    text_key: str,
    label_key: str,
    num_classes: int,
    max_length: Optional[int],
    image_max_pixels: Optional[int] = None,
    row_idx: int,
    answergen1_system_prompt: Optional[str],
    answergen1_force_rgb: bool,
    require_image_processor: bool,
    processor_error: Optional[str],
    dataset_root: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    text_state = _build_text_and_token_state(
        tokenizer=tokenizer,
        raw_conv=row.get(text_key),
        max_length=max_length,
        row_idx=row_idx,
        answergen1_system_prompt=answergen1_system_prompt,
        answergen1_force_rgb=answergen1_force_rgb,
        processor=processor,
        require_image_processor=require_image_processor,
        processor_error=processor_error,
        dataset_root=dataset_root,
    )
    if text_state is None:
        return None

    annotations = _parse_span_annotations(row.get(label_key), num_classes)
    if not annotations:
        return None

    span_text_indices: list[list[int]] = []
    for ann in annotations:
        indices = get_span_token_indices(
            ann=ann,
            offsets=text_state["offsets"],
            input_ids=text_state["text_input_ids"],
            special_ids=text_state["special_ids"],
            completion_start_tok=text_state["completion_start_tok"],
            completion_char0=text_state["completion_char0"],
            completion_text=text_state["completion_text"],
            full_text=text_state["full_text"],
        )
        span_text_indices.append(indices)

    pos_map = _build_pos_map(
        processor=processor,
        full_text=text_state["full_text"],
        image_path=text_state["image_path"],
        text_input_ids=text_state["text_input_ids"],
        max_length=max_length,
        image_max_pixels=image_max_pixels,
        row_idx=row_idx,
        answergen1_force_rgb=answergen1_force_rgb,
        require_image_processor=require_image_processor,
        processor_error=processor_error,
        full_messages=text_state["full_messages"],
    )

    spans: list[dict[str, Any]] = []
    for ann, text_indices in zip(annotations, span_text_indices):
        token_indices = [pos_map[idx] for idx in text_indices if idx in pos_map]
        if not token_indices:
            continue
        spans.append(
            {
                "class_id": ann["class_id"],
                "token_indices": token_indices,
            }
        )

    if not spans:
        return None

    return {
        "full_text": text_state["full_text"],
        "image_path": text_state["image_path"],
        "text_input_ids": text_state["text_input_ids"],
        "prompt_messages": text_state["prompt_messages"],
        "full_messages": text_state["full_messages"],
        "spans": spans,
    }


def build_sentence_row_info(
    *,
    row: dict[str, Any],
    tokenizer,
    processor,
    text_key: str,
    label_key: str,
    num_classes: int,
    max_length: Optional[int],
    image_max_pixels: Optional[int] = None,
    row_idx: int,
    answergen1_system_prompt: Optional[str],
    answergen1_force_rgb: bool,
    require_image_processor: bool,
    processor_error: Optional[str],
    dataset_root: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    text_state = _build_text_and_token_state(
        tokenizer=tokenizer,
        raw_conv=row.get(text_key),
        max_length=max_length,
        row_idx=row_idx,
        answergen1_system_prompt=answergen1_system_prompt,
        answergen1_force_rgb=answergen1_force_rgb,
        processor=processor,
        require_image_processor=require_image_processor,
        processor_error=processor_error,
        dataset_root=dataset_root,
    )
    if text_state is None:
        return None

    del num_classes
    annotations = _parse_sentence_annotations(row.get(label_key))
    sentence_answer = _derive_sentence_answer(annotations)
    if sentence_answer is None:
        return None

    answer_text_indices = [
        idx
        for idx in range(text_state["completion_start_tok"], len(text_state["text_input_ids"]))
        if idx < len(text_state["offsets"])
        and text_state["text_input_ids"][idx] not in text_state["special_ids"]
        and int(text_state["offsets"][idx][1]) > int(text_state["offsets"][idx][0])
    ]
    if not answer_text_indices:
        completion_char1 = text_state["completion_char0"] + len(text_state["completion_text"])
        for token_idx, (start_char, end_char) in enumerate(text_state["offsets"]):
            if token_idx < text_state["completion_start_tok"]:
                continue
            if token_idx >= len(text_state["text_input_ids"]):
                continue
            if text_state["text_input_ids"][token_idx] in text_state["special_ids"]:
                continue
            if int(end_char) <= int(start_char):
                continue
            if _token_overlaps(
                (int(start_char), int(end_char)),
                (text_state["completion_char0"], completion_char1),
            ):
                answer_text_indices.append(token_idx)
    if not answer_text_indices:
        return None

    pos_map = _build_pos_map(
        processor=processor,
        full_text=text_state["full_text"],
        image_path=text_state["image_path"],
        text_input_ids=text_state["text_input_ids"],
        max_length=max_length,
        image_max_pixels=image_max_pixels,
        row_idx=row_idx,
        answergen1_force_rgb=answergen1_force_rgb,
        require_image_processor=require_image_processor,
        processor_error=processor_error,
        full_messages=text_state["full_messages"],
    )

    answer_token_indices = [pos_map[idx] for idx in answer_text_indices if idx in pos_map]
    if not answer_token_indices:
        return None

    return {
        "full_text": text_state["full_text"],
        "image_path": text_state["image_path"],
        "text_input_ids": text_state["text_input_ids"],
        "prompt_messages": text_state["prompt_messages"],
        "full_messages": text_state["full_messages"],
        "answer_token_indices": answer_token_indices,
        "sentence_answer": sentence_answer,
    }


def forward_row_hidden(
    *,
    row_info: dict[str, Any],
    tokenizer,
    processor,
    model: torch.nn.Module,
    captured: dict[str, Any],
    layer_keys: list[str],
    max_length: Optional[int],
    image_max_pixels: Optional[int] = None,
    row_idx: int,
    answergen1_force_rgb: bool,
    require_image_processor: bool,
    processor_error: Optional[str],
):
    full_text = row_info["full_text"]
    image_path = row_info.get("image_path")
    full_messages = row_info.get("full_messages")

    device = next(model.parameters()).device
    model_inputs = None

    if image_path is not None and processor is None:
        msg = f"[row {row_idx}] image path '{image_path}' is present but AutoProcessor is unavailable."
        if processor_error:
            msg += f" AutoProcessor error: {processor_error}"
        if require_image_processor:
            raise ImageProcessorRequiredError(msg + " Refusing text-only fallback.")
        print(
            f"{msg} Falling back to text-only because --allow_text_only_image_fallback is enabled.",
            file=sys.stderr,
        )

    if image_path is not None and processor is not None:
        try:
            proc_out = vlm_compat.encode_chat_messages(
                processor,
                full_messages if isinstance(full_messages, list) and full_messages else [],
                add_generation_prompt=False,
                return_tensors="pt",
                padding=False,
                image_max_pixels=image_max_pixels,
                max_length=max_length,
            )
            model_inputs = vlm_compat.tensor_inputs_to_device(proc_out, device)
        except Exception as exc:
            if require_image_processor:
                raise ImageProcessorRequiredError(
                    f"[row {row_idx}] cannot open/process image '{image_path}' in forward pass: {exc}"
                ) from exc
            print(
                f"[row {row_idx}] WARNING: cannot open image '{image_path}': {exc}. Falling back to text-only.",
                file=sys.stderr,
            )

    if model_inputs is None:
        enc_full = vlm_compat.tokenize_text_with_offsets(tokenizer, full_text, max_length=max_length)
        model_inputs = vlm_compat.tensor_inputs_to_device(
            {
                key: value
                for key, value in enc_full.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            },
            device,
        )

    captured.clear()
    with torch.no_grad():
        model(**model_inputs)

    hidden_by_layer = {key: captured[key] for key in layer_keys if key in captured}
    if not hidden_by_layer:
        print(f"[row {row_idx}] WARNING: hook did not capture any activations, skipping.", file=sys.stderr)
        return None

    missing = [key for key in layer_keys if key not in hidden_by_layer]
    if missing:
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += ", ..."
        print(
            f"[row {row_idx}] WARNING: missing activations for {len(missing)} layer(s): {preview}",
            file=sys.stderr,
        )

    return hidden_by_layer


def _split_yes_no(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    yes_entries = [entry for entry in entries if str(entry.get("answer", "")).strip().lower() == "yes"]
    no_entries = [entry for entry in entries if str(entry.get("answer", "")).strip().lower() == "no"]
    return yes_entries, no_entries


def _sample_subset(entries: list[dict[str, Any]], take: int, rng: random.Random) -> list[dict[str, Any]]:
    if take <= 0:
        return []
    if take >= len(entries):
        return list(entries)
    return rng.sample(entries, take)


def _assign_q1_sample_modes(entries: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    items = [dict(entry) for entry in entries]
    if not items:
        return []

    rng.shuffle(items)
    single_count = int(round(len(items) * Q1_SINGLE_TOKEN_RATIO))
    single_count = max(1, min(len(items), single_count))
    if len(items) >= 2 and single_count == len(items):
        single_count -= 1

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        item["sample_mode"] = "single" if idx < single_count else "window"
        out.append(item)
    return out


def sample_q1_entries_with_modes(
    entries: list[dict[str, Any]],
    safe_target: int,
    unsafe_target: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    safe_items, unsafe_items = _split_yes_no(entries)

    requested_take = min(max(0, safe_target), max(0, unsafe_target))
    balanced_take = min(requested_take, len(safe_items), len(unsafe_items))

    if balanced_take != safe_target or balanced_take != unsafe_target:
        print(
            "Q1 uses balanced Yes/No sampling; "
            f"taking {balanced_take} per label "
            f"(requested safe={safe_target}, unsafe={unsafe_target}; "
            f"available safe={len(safe_items)}, unsafe={len(unsafe_items)})",
            file=sys.stderr,
        )

    selected = _assign_q1_sample_modes(_sample_subset(safe_items, balanced_take, rng), rng)
    selected += _assign_q1_sample_modes(_sample_subset(unsafe_items, balanced_take, rng), rng)
    rng.shuffle(selected)
    return selected


def _sample_q1_single_token(token_indices: list[int], rng: random.Random) -> list[int]:
    if not token_indices:
        return []
    tail_count = min(Q1_TAIL_TOKEN_RANGE, len(token_indices))
    tail_tokens = token_indices[-tail_count:]
    return [int(rng.choice(tail_tokens))]


def _sample_q1_window_tokens(token_indices: list[int], rng: random.Random) -> list[int]:
    if not token_indices:
        return []
    if len(token_indices) == 1:
        return [int(token_indices[0])]

    tail_start = max(0, len(token_indices) - Q1_TAIL_TOKEN_RANGE)
    end_pos = rng.randint(tail_start, len(token_indices) - 1)
    max_window_len = min(len(token_indices), Q1_WINDOW_MAX_LENGTH)
    min_window_len = 2 if len(token_indices) >= 2 else 1
    window_len = rng.randint(min_window_len, max_window_len)
    start_pos = max(0, end_pos - window_len + 1)
    sampled = token_indices[start_pos : end_pos + 1]
    if not sampled:
        sampled = [token_indices[end_pos]]
    return [int(idx) for idx in sampled]


def sample_q1_activation_indices(token_indices: list[int], sample_mode: str, rng: random.Random) -> list[int]:
    if sample_mode == "window":
        return _sample_q1_window_tokens(token_indices, rng)
    return _sample_q1_single_token(token_indices, rng)


def _sample_balanced_by_label(
    entries: list[dict[str, Any]],
    rng: random.Random,
    target_per_label: Optional[int] = None,
) -> list[dict[str, Any]]:
    yes_entries, no_entries = _split_yes_no(entries)
    take_each = min(len(yes_entries), len(no_entries))
    if target_per_label is not None:
        take_each = min(take_each, max(0, target_per_label))
    if take_each <= 0:
        return []
    out = _sample_subset(yes_entries, take_each, rng) + _sample_subset(no_entries, take_each, rng)
    rng.shuffle(out)
    return out


def _count_balanced_label_pairs(entries: list[dict[str, Any]]) -> int:
    yes_entries, no_entries = _split_yes_no(entries)
    return min(len(yes_entries), len(no_entries))


def _build_q1_length_boundaries(lengths: list[int], max_bucket_count: int = 8) -> list[int]:
    positive_lengths = sorted(int(length) for length in lengths if int(length) > 0)
    if not positive_lengths:
        return []

    unique_lengths = sorted(set(positive_lengths))
    target_bucket_count = min(max_bucket_count, len(unique_lengths))
    if target_bucket_count <= 1:
        return []

    boundaries: list[int] = []
    n = len(positive_lengths)
    max_length = positive_lengths[-1]
    for bucket_idx in range(1, target_bucket_count):
        pos = max(0, min(n - 2, (n * bucket_idx) // target_bucket_count - 1))
        boundary = positive_lengths[pos]
        if boundary >= max_length:
            continue
        if boundaries and boundary <= boundaries[-1]:
            continue
        boundaries.append(boundary)
    return boundaries


def _get_q1_length_bucket_id(length: int, boundaries: list[int]) -> int:
    return bisect_right(boundaries, int(length))


def _format_q1_length_bucket(bucket_id: int, boundaries: list[int]) -> str:
    lower = 1 if bucket_id == 0 else boundaries[bucket_id - 1] + 1
    if bucket_id < len(boundaries):
        upper = boundaries[bucket_id]
        return f"len_{lower}_{upper}"
    return f"len_{lower}_plus"


def _balance_q1_window_entries(entries: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    if not entries:
        return []

    boundaries = _build_q1_length_boundaries(
        [int(entry.get("activation_length", 0)) for entry in entries],
    )
    grouped_windows: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        bucket_id = _get_q1_length_bucket_id(int(entry.get("activation_length", 0)), boundaries)
        grouped_windows.setdefault(bucket_id, []).append(entry)

    balanced_window: list[dict[str, Any]] = []
    bucket_summaries: list[str] = []
    for bucket_id in sorted(grouped_windows.keys()):
        bucket_entries = grouped_windows[bucket_id]
        balanced_bucket = _sample_balanced_by_label(bucket_entries, rng)
        if balanced_bucket:
            balanced_window.extend(balanced_bucket)
        yes_entries, no_entries = _split_yes_no(bucket_entries)
        bucket_summaries.append(
            f"{_format_q1_length_bucket(bucket_id, boundaries)}:{min(len(yes_entries), len(no_entries)) * 2}/{len(bucket_entries)}"
        )

    if bucket_summaries:
        print(
            "Q1 dynamic length buckets (kept/total): " + ", ".join(bucket_summaries),
            file=sys.stderr,
        )
    return balanced_window


def _align_q1_mode_ratio(
    single_entries: list[dict[str, Any]],
    window_entries: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    single_ratio = min(max(Q1_SINGLE_TOKEN_RATIO, 0.0), 1.0)
    window_ratio = 1.0 - single_ratio

    if single_ratio <= 0.0:
        return [], window_entries
    if window_ratio <= 0.0:
        return single_entries, []

    single_pairs = _count_balanced_label_pairs(single_entries) if single_entries else 0
    window_pairs = _count_balanced_label_pairs(window_entries) if window_entries else 0

    if single_pairs <= 0 or window_pairs <= 0:
        return single_entries, window_entries

    target_single_pairs = max(1, int(round(window_pairs * single_ratio / window_ratio)))
    target_window_pairs = max(1, int(round(single_pairs * window_ratio / single_ratio)))

    if single_pairs > target_single_pairs:
        single_entries = _sample_balanced_by_label(single_entries, rng, target_per_label=target_single_pairs)
    elif window_pairs > target_window_pairs:
        window_entries = _sample_balanced_by_label(window_entries, rng, target_per_label=target_window_pairs)

    return single_entries, window_entries


def balance_q1_final_entries(entries: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    single_entries = [entry for entry in entries if str(entry.get("sample_mode", "")) == "single"]
    window_entries = [entry for entry in entries if str(entry.get("sample_mode", "")) == "window"]

    balanced_single = _sample_balanced_by_label(single_entries, rng)
    balanced_window = _balance_q1_window_entries(window_entries, rng)
    balanced_single, balanced_window = _align_q1_mode_ratio(balanced_single, balanced_window, rng)

    balanced = balanced_single + balanced_window
    rng.shuffle(balanced)
    return balanced
