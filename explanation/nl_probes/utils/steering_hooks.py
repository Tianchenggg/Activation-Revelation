import contextlib
import os
from typing import Callable

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


_NONFINITE_HOOK_INPUT_WARNING_COUNT = 0


def _warn_nonfinite_hook_input(*, bad_count: int, debug_info: str) -> None:
    global _NONFINITE_HOOK_INPUT_WARNING_COUNT
    limit = _env_int("AO_NONFINITE_HOOK_INPUT_WARNING_LIMIT", 4)
    if limit <= 0:
        return
    if _NONFINITE_HOOK_INPUT_WARNING_COUNT < limit:
        print(
            "WARNING: Non-finite activation detected at steering hook input; "
            "sanitizing with torch.nan_to_num because "
            "AO_SANITIZE_NONFINITE_HOOK_INPUT=1. "
            f"nonfinite_values={bad_count}, {debug_info}"
        )
    elif _NONFINITE_HOOK_INPUT_WARNING_COUNT == limit:
        print(
            "WARNING: Further non-finite hook-input sanitization warnings suppressed. "
            "Adjust AO_NONFINITE_HOOK_INPUT_WARNING_LIMIT to show more."
        )
    _NONFINITE_HOOK_INPUT_WARNING_COUNT += 1

def get_vllm_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[int],
    prompt_lengths: list[int],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Debug version of your steering hook with detailed logging
    """
    vec_BD = torch.stack(vectors)  # (B, d_model)
    pos_B = torch.tensor(positions, dtype=torch.long)  # (B,)
    B, d_model = vec_BD.shape
    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    def hook_fn(module, _input, output):
        # passed prompt lengths should line up hopefully
        tokens_L = _input[0]

        if tokens_L.shape[0] == B:
            # means we are in decoding, not prefill. So no need to steer.
            return output

        # if there aren't any 0s in tokens_L, then we are NOT in prefill. So skip
        if not torch.any(tokens_L == 0):
            return output

        number_of_zeroes = torch.sum(tokens_L == 0).item()
        # should be equal to number of prompts
        if number_of_zeroes != len(prompt_lengths):
            breakpoint()
            raise ValueError(
                f"Number of zeroes {number_of_zeroes} is not equal to number of prompt lengths {len(prompt_lengths)}"
            )

        count = 0
        for prompt_length in prompt_lengths:
            expected_position_indices_L = torch.arange(prompt_length, device=device)
            try:
                assert tokens_L[count : count + prompt_length].equal(expected_position_indices_L), (
                    f"Position indices mismatch at index {count}, expected {expected_position_indices_L}, got {tokens_L[count : count + prompt_length]}"
                )
            except AssertionError as e:
                raise e

            count += prompt_length

        before_resid_flat, resid_flat, *rest = output

        assert count == tokens_L.shape[0]
        assert resid_flat.shape[0] == tokens_L.shape[0]
        assert resid_flat.shape[1] == d_model

        intervention_indices_L = []
        idx = 0

        for i in range(len(prompt_lengths)):
            intervention_idx = torch.tensor(idx + positions[i], device=device)
            intervention_indices_L.append(intervention_idx)
            idx += prompt_lengths[i]

        assert idx >= tokens_L.shape[0]

        intervention_indices_L = torch.stack(intervention_indices_L)

        assert intervention_indices_L.shape[0] == B

        orig_BD = resid_flat[intervention_indices_L]

        assert orig_BD.shape == (B, d_model)

        # Compute norms and steering
        norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()
        normalized_features = torch.nn.functional.normalize(vec_BD, dim=-1)
        steered_BD = normalized_features * norms_B1 * steering_coefficient

        # print(f"  Normalized feature norms: {normalized_features.norm(dim=-1).tolist()}")
        # print(f"  Original norms: {norms_B1.squeeze().tolist()}")
        # print(f"  Steered activation norms: {steered_BD.norm(dim=-1).tolist()}")

        # Calculate the change magnitude BEFORE applying
        change_magnitude = (steered_BD - orig_BD).norm(dim=-1)
        print(f"  Change magnitudes: {change_magnitude.tolist()}")

        if change_magnitude.max() < 1e-4:
            print("  ⚠️  WARNING: Very small change magnitude!")

        # Apply the steering
        # print(f"  Applying steering at positions: {pos_B.tolist()}")
        resid_flat[intervention_indices_L] = steered_BD

        return (before_resid_flat, resid_flat, *rest)

    return hook_fn


@contextlib.contextmanager
def add_hook(
    module: torch.nn.Module,
    hook: Callable,
):
    """Temporarily adds a forward hook to a model module.

    Args:
        module: The PyTorch module to hook
        hook: The hook function to apply

    Yields:
        None: Used as a context manager

    Example:
        with add_hook(model.layer, hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],  # len B, each tensor is (K_b, d_model)
    positions: list[list[int]],  # len B, each list has length K_b
    steering_coefficient: float,
    steering_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    debug_infos: list[str] | None = None,
) -> Callable:
    """
    HF hook with debug prints to compare against vLLM.
    Supports a variable number of target positions per batch element.

    Semantics:
      For each batch item b and slot k, build:
      steer = normalize(vectors[b][k]) * ||resid[b, positions[b][k], :]|| * steering_coefficient.

      - steering_mode == "replace": resid <- steer
      - steering_mode == "add":     resid <- resid + steer

    We use a for loop instead of vectorized operations as it's simpler and we are just doing indexing in
    a single layer, so the simplicity won out for now.
    """

    # ---- move inputs to device and prepare ragged tensors ----
    assert len(vectors) == len(positions), "vectors and positions must have same batch length"
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    if steering_mode not in {"replace", "add"}:
        raise ValueError(f"Unknown steering_mode={steering_mode!r}. Expected 'replace' or 'add'.")

    # Pre-normalize once in fp32; we never backprop through these activations.
    normed_list = []
    for b, v_b in enumerate(vectors):
        v_b_f32 = v_b.to(device=device, dtype=torch.float32)
        if not torch.isfinite(v_b_f32).all():
            debug_info = debug_infos[b] if debug_infos and b < len(debug_infos) else f"batch_index={b}"
            raise FloatingPointError(f"Non-finite steering vector detected before hook. {debug_info}")
        normed_list.append(torch.nn.functional.normalize(v_b_f32, dim=-1).detach())
    warned_outlier = False
    sanitize_nonfinite_hook_input = _env_flag("AO_SANITIZE_NONFINITE_HOOK_INPUT")

    def hook_fn(module, _input, output):
        nonlocal warned_outlier
        # Normalize output API across model families
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        output_is_LBD = False
        if resid_BLD.ndim != 3:
            raise ValueError(f"Expected 3-D hook activation, got shape={list(resid_BLD.shape)}")
        if resid_BLD.shape[0] != B and resid_BLD.shape[1] == B:
            resid_BLD = resid_BLD.transpose(0, 1).contiguous()
            output_is_LBD = True

        B_actual, L, d_model_actual = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={B}")
        for b, normed_b in enumerate(normed_list):
            if int(normed_b.shape[-1]) != int(d_model_actual):
                debug_info = debug_infos[b] if debug_infos and b < len(debug_infos) else f"batch_index={b}"
                raise ValueError(
                    "Steering vector hidden size does not match hook layer hidden size. "
                    f"vector_dim={int(normed_b.shape[-1])}, hook_dim={int(d_model_actual)}. "
                    "Extraction layer and injection layer may differ only when their hidden sizes match. "
                    f"{debug_info}"
                )

        # Only touch the prompt forward pass
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        nonfinite_mask = ~torch.isfinite(resid_BLD)
        if nonfinite_mask.any():
            if not sanitize_nonfinite_hook_input:
                # Keep the stricter historical behavior unless a caller explicitly opts in.
                for b in range(B):
                    pos_b = torch.tensor(positions[b], dtype=torch.long, device=device)
                    orig_KD = resid_BLD[b, pos_b, :]
                    if not torch.isfinite(orig_KD).all():
                        bad_slots = torch.nonzero(~torch.isfinite(orig_KD), as_tuple=False)
                        first_bad = bad_slots[0].tolist() if bad_slots.numel() else []
                        debug_info = debug_infos[b] if debug_infos and b < len(debug_infos) else f"batch_index={b}"
                        raise FloatingPointError(
                            "Non-finite activation detected at steering hook input. "
                            f"first_bad_index={first_bad}, positions={positions[b]}, {debug_info}"
                        )
            else:
                bad_count = int(nonfinite_mask.sum().item())
                debug_info = debug_infos[0] if debug_infos else "no sample debug info"
                _warn_nonfinite_hook_input(bad_count=bad_count, debug_info=debug_info)
                resid_BLD = torch.nan_to_num(resid_BLD, nan=0.0, posinf=0.0, neginf=0.0)

        steered_resid_BLD = resid_BLD.clone()

        # Per-batch element work. Vectorized over K_b where safe.
        for b in range(B):
            pos_b = positions[b]
            pos_b = torch.tensor(pos_b, dtype=torch.long, device=device)
            assert pos_b.min() >= 0
            assert pos_b.max() < L
            # Gather original activations at requested slots and compute norms
            orig_KD = resid_BLD[b, pos_b, :]  # (K_b, d)
            norms_K1 = orig_KD.float().norm(dim=-1, keepdim=True)  # (K_b, 1)

            if b == 0:
                norms_flat = norms_K1.view(-1)
                max_norm = float(norms_flat.max().item())
                median_norm = float(norms_flat.median().item())
                # Warn on clear outliers; stable-but-large norms are model/layer dependent.
                if median_norm > 0.0 and max_norm > 300.0 and (max_norm / median_norm) > 3.0:
                    if not warned_outlier:
                        print(
                            "WARNING: Activation norm outlier detected in steering hook "
                            f"(max={max_norm:.2f}, median={median_norm:.2f})."
                        )
                        warned_outlier = True

            # Build steered vectors for this b
            steered_KD = (normed_list[b] * norms_K1.float() * steering_coefficient).to(dtype)  # (K_b, d)

            if steering_mode == "replace":
                steered_resid_BLD[b, pos_b, :] = steered_KD.detach()
            else:  # steering_mode == "add"
                steered_resid_BLD[b, pos_b, :] = orig_KD + steered_KD.detach()

        if output_is_LBD:
            steered_resid_BLD = steered_resid_BLD.transpose(0, 1).contiguous()

        return (steered_resid_BLD, *rest) if output_is_tuple else steered_resid_BLD

    return hook_fn
