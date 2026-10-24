"""Exact, non-cumulative, parameter-space task-vector arithmetic."""
from __future__ import annotations
from collections.abc import Mapping
import math
from pathlib import Path
import torch


def unique_parameters(model) -> dict[str, torch.nn.Parameter]:
    """named_parameters removes duplicate references to tied embedding weights."""
    return {k: p for k, p in model.named_parameters() if p.is_floating_point()}


def _check_keys(reference: Mapping, other: Mapping, label: str) -> None:
    if set(reference) != set(other):
        missing = sorted(set(reference) - set(other))[:5]
        extra = sorted(set(other) - set(reference))[:5]
        raise ValueError(f"{label}: parameter mismatch; missing={missing}; extra={extra}")
    for name in reference:
        if reference[name].shape != other[name].shape:
            raise ValueError(f"{label}: shape mismatch for {name}")


def extract_vector(base_model, tuned_model) -> dict[str, torch.Tensor]:
    """Compute tau_t=theta_t-theta_0 in FP32 with tied tensors counted once."""
    base = unique_parameters(base_model)
    tuned = unique_parameters(tuned_model)
    _check_keys(base, tuned, "extract_vector")
    return {
        k: tuned[k].detach().to(device="cpu", dtype=torch.float32)
        - base[k].detach().to(device="cpu", dtype=torch.float32)
        for k in base
    }


def vector_norm(vector: Mapping[str, torch.Tensor]) -> float:
    sq = sum(float(torch.sum(v.double() ** 2)) for v in vector.values())
    return math.sqrt(sq)


def vector_cosine(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    _check_keys(a, b, "vector_cosine")
    dot = sum(float(torch.sum(a[k].double() * b[k].double())) for k in a)
    denom = vector_norm(a) * vector_norm(b)
    return dot / denom if denom else float("nan")


def save_vector(vector: Mapping[str, torch.Tensor], path: str | Path, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "metadata": metadata,
               "delta": {k: v.detach().cpu().float().contiguous() for k, v in vector.items()}}
    torch.save(payload, path)


def load_vector(path: str | Path, *, expected_base: str | None = None):
    # weights_only=True prevents arbitrary object unpickling for tensor artifacts.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or not isinstance(payload.get("delta"), dict):
        raise ValueError(f"Unrecognized vector format: {path}")
    metadata = payload.get("metadata", {})
    if expected_base and metadata.get("base_model") != expected_base:
        raise ValueError(f"Vector {path} expects base {metadata.get('base_model')}, not {expected_base}")
    return payload["delta"], metadata


class ArithmeticModel:
    """Apply independent vector combinations ALWAYS from immutable base weights.

    All model weights and vectors should derive from the exact same base checkpoint.
    Only floating parameters are modified, preserving tied weights via unique names.
    """
    def __init__(self, model):
        self.model = model
        self.params = unique_parameters(model)
        self.base = {k: p.detach().cpu().float().clone() for k, p in self.params.items()}
        self.vectors: dict[str, dict[str, torch.Tensor]] = {}

    def register(self, label: str, vector: Mapping[str, torch.Tensor]):
        _check_keys(self.base, vector, label)
        self.vectors[label] = {k: v.detach().cpu().float() for k, v in vector.items()}

    @torch.no_grad()
    def apply(self, coefficients: Mapping[str, float]):
        for name in coefficients:
            if name not in self.vectors:
                raise KeyError(f"Unregistered task vector: {name}")
        for key, param in self.params.items():
            # Compute in float32; reset to base on every call to avoid cumulative drift.
            edited = self.base[key].clone()
            for label, coefficient in coefficients.items():
                if not math.isfinite(float(coefficient)):
                    raise ValueError("Coefficient must be finite")
                if coefficient:
                    edited.add_(self.vectors[label][key], alpha=float(coefficient))
            param.copy_(edited.to(param.device, dtype=param.dtype))
        return self.model

    def restore(self):
        return self.apply({})

    @torch.no_grad()
    def max_abs_error_vs(self, other_model) -> float:
        params = unique_parameters(other_model)
        _check_keys(self.params, params, "reconstruction")
        return max(float((p.detach().float().cpu() - params[k].detach().float().cpu()).abs().max())
                   for k, p in self.params.items())
