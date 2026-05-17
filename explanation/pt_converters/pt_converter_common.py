from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoProcessor, AutoTokenizer

def _iter_ao_project_candidates() -> list[Path]:
    script_path = Path(__file__).resolve()
    relative_candidates = [
        (),
        ("src", "explanation", "activation_oracles-main"),
        ("src", "ours", "AO", "activation_oracles-main"),
        ("activation_oracles-main",),
        ("activation_oracles-main", "activation_oracles-main"),
        ("AO-safe-vlm", "activation_oracles-main", "activation_oracles-main"),
        ("AO-final", "activation_oracles-main", "activation_oracles-main"),
    ]

    ordered_candidates: list[Path] = []
    seen: set[Path] = set()
    for parent in script_path.parents:
        for rel_parts in relative_candidates:
            candidate = parent.joinpath(*rel_parts).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered_candidates.append(candidate)
    return ordered_candidates


AO_PROJECT_CANDIDATES = _iter_ao_project_candidates()
for _candidate in AO_PROJECT_CANDIDATES:
    if (_candidate / "nl_probes").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

try:
    from nl_probes.dataset_classes.custom_pt_serialization import (  # noqa: E402
        CUSTOM_PT_MANIFEST_FORMAT,
        pack_training_datapoint_records,
    )
    from nl_probes.utils.dataset_utils import (  # noqa: E402
        build_vlm_prompt_messages,
        create_vlm_training_datapoint,
    )
    from nl_probes.utils.vlm_compat import (  # noqa: E402
        attach_model_family,
        detect_model_family,
        load_config,
        normalize_hook_tensor_to_bld,
        processor_tokenizer,
    )
except ModuleNotFoundError as exc:
    if exc.name != "nl_probes":
        raise

    searched = "\n".join(f"  - {candidate}" for candidate in AO_PROJECT_CANDIDATES)
    raise ModuleNotFoundError(
        "Could not import 'nl_probes'. Checked these activation_oracles candidates:\n"
        f"{searched}"
    ) from exc

DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

ACTIVATION_STORAGE_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class PtShardWriter:
    def __init__(
        self,
        output_pt: str,
        shard_size: int,
        activation_dtype: torch.dtype,
        *,
        single_file_output: bool = False,
    ) -> None:
        self.output_path = Path(output_pt).resolve()
        self.shard_size = max(1, int(shard_size))
        self.activation_dtype = activation_dtype
        self.single_file_output = single_file_output
        self.shard_dir = self.output_path.parent / f"{self.output_path.stem}.shards"
        self.buffer: list[Any] = []
        self.shards: list[dict[str, Any]] = []
        self.total_entries = 0

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.single_file_output:
            if self.shard_dir.exists():
                shutil.rmtree(self.shard_dir)
            self.shard_dir.mkdir(parents=True, exist_ok=True)

    def append(self, datapoint: Any) -> None:
        self.buffer.append(datapoint)
        self.total_entries += 1
        if not self.single_file_output and len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self.single_file_output:
            return
        if not self.buffer:
            return

        shard_idx = len(self.shards)
        shard_name = f"part-{shard_idx:06d}.pt"
        shard_path = self.shard_dir / shard_name
        shard_data = self.buffer
        self.buffer = []

        packed_shard = pack_training_datapoint_records(
            shard_data,
            activation_dtype=self.activation_dtype,
        )
        torch.save(packed_shard, shard_path)
        self.shards.append(
            {
                "path": f"{self.shard_dir.name}/{shard_name}",
                "num_entries": int(packed_shard["num_entries"]),
            }
        )

    def close(self, metadata: dict[str, Any]) -> None:
        if self.single_file_output:
            if not self.buffer:
                raise ValueError("Cannot write a single PT file with zero datapoints.")
            packed_shard = pack_training_datapoint_records(
                self.buffer,
                activation_dtype=self.activation_dtype,
            )
            torch.save(packed_shard, self.output_path)
            self.shards = [
                {
                    "path": self.output_path.name,
                    "num_entries": int(packed_shard["num_entries"]),
                }
            ]
            self.buffer = []
            return

        self.flush()
        manifest = {
            "format": CUSTOM_PT_MANIFEST_FORMAT,
            "num_entries": self.total_entries,
            "num_shards": len(self.shards),
            "shards": [s["path"] for s in self.shards],
            "shard_stats": self.shards,
            **metadata,
        }
        torch.save(manifest, self.output_path)


def resolve_layer_paths_and_ids(args, infer_layer_id_fn):
    single_layer = str(args.layer).strip() if args.layer is not None else ""
    multi_layers = [str(x).strip() for x in (args.layers or []) if str(x).strip()]

    if single_layer and multi_layers:
        raise ValueError("Please pass either --layer or --layers, not both.")
    if single_layer:
        layer_paths = [single_layer]
    elif multi_layers:
        layer_paths = multi_layers
    else:
        raise ValueError("You must pass --layer or --layers.")

    if args.layer_id is not None and args.layer_ids is not None:
        raise ValueError("Please pass either --layer_id or --layer_ids, not both.")

    if args.layer_ids is not None:
        if len(args.layer_ids) != len(layer_paths):
            raise ValueError(
                f"Length of --layer_ids ({len(args.layer_ids)}) must match number of layers ({len(layer_paths)})."
            )
        layer_ids = [int(x) for x in args.layer_ids]
    elif args.layer_id is not None:
        if len(layer_paths) != 1:
            raise ValueError("--layer_id can only be used with a single layer. Use --layer_ids for multiple layers.")
        layer_ids = [int(args.layer_id)]
    else:
        layer_ids = []
        for layer_path in layer_paths:
            inferred = infer_layer_id_fn(layer_path)
            if inferred is None:
                raise ValueError(
                    f"Cannot infer integer layer_id from layer path '{layer_path}'. "
                    "Please pass --layer_ids explicitly."
                )
            layer_ids.append(inferred)

    if len(set(layer_paths)) != len(layer_paths):
        raise ValueError("Duplicate layer paths found in --layers/--layer; please provide unique layer paths.")

    layer_id_by_path = {layer_path: layer_id for layer_path, layer_id in zip(layer_paths, layer_ids)}
    return layer_paths, layer_ids, layer_id_by_path


def load_processor_and_tokenizer(
    *,
    model_path: str,
    allow_text_only_image_fallback: bool,
    patch_mistral_fn,
):
    processor = None
    tokenizer = None
    processor_load_error: Optional[Exception] = None
    print(f"Loading processor/tokenizer from: {model_path}", file=sys.stderr)
    try:
        if patch_mistral_fn():
            print("  Applied mistral_common compatibility patch for tokenizer utils.", file=sys.stderr)
        processor = _from_pretrained_with_cache_fallback(
            AutoProcessor,
            model_path,
            label="AutoProcessor",
            trust_remote_code=True,
        )
        config = load_config(model_path)
        family = detect_model_family(model_path, config=config, processor=processor)
        attach_model_family(processor, family)
        tokenizer = processor_tokenizer(processor)
        attach_model_family(tokenizer, family)
        print("  AutoProcessor loaded (multimodal / VLM mode).", file=sys.stderr)
    except Exception as exc:
        processor_load_error = exc
        print(f"  AutoProcessor failed: {exc}", file=sys.stderr)

    if tokenizer is None:
        tokenizer = _from_pretrained_with_cache_fallback(
            AutoTokenizer,
            model_path,
            label="AutoTokenizer",
            trust_remote_code=True,
        )
        print("  AutoTokenizer loaded (text-only mode).", file=sys.stderr)
    if processor is None and not allow_text_only_image_fallback:
        print(
            "  AutoProcessor unavailable: rows containing images will now raise an error "
            "to prevent silently dropping image inputs.",
            file=sys.stderr,
        )

    return processor, tokenizer, processor_load_error


def _from_pretrained_with_cache_fallback(factory, model_path: str, *, label: str, **kwargs):
    try:
        return factory.from_pretrained(model_path, **kwargs)
    except Exception as primary_exc:
        if kwargs.get("local_files_only"):
            raise
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        try:
            obj = factory.from_pretrained(model_path, **local_kwargs)
        except Exception as local_exc:
            raise RuntimeError(
                f"{label} primary load failed: {primary_exc}; local cache fallback failed: {local_exc}"
            ) from local_exc
        print(
            f"  {label} primary load failed; loaded from local cache instead. "
            f"Original error: {primary_exc}",
            file=sys.stderr,
        )
        return obj


def register_forward_hooks(model, layer_paths: list[str], layer_id_by_path: dict[str, int], get_target_module_fn):
    captured: dict[str, Any] = {}
    handles = []

    def _make_hook(layer_key: str):
        def hook_fn(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            tensor = normalize_hook_tensor_to_bld(tensor, layer_key=layer_key)
            captured[layer_key] = tensor.detach().float().cpu()

        return hook_fn

    for layer_path in layer_paths:
        target_module = get_target_module_fn(model, layer_path)
        handle = target_module.register_forward_hook(_make_hook(layer_path))
        handles.append(handle)
        print(f"Hook registered on: {layer_path} (layer_id={layer_id_by_path[layer_path]})", file=sys.stderr)

    return captured, handles


def remove_forward_handles(handles) -> None:
    for handle in handles:
        handle.remove()
