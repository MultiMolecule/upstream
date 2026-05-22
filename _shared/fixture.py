# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

"""Shared fixture writer and validator utilities.

The helpers in this module intentionally stay close to the artifact contract:
one fixture case directory contains `inputs.safetensors`, `expected.safetensors`,
and `meta.json`. They do not import MultiMolecule model code.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from safetensors import safe_open

from _shared.checksum import ChecksumError
from _shared.checksum import remember_sha256 as _remember_sha256
from _shared.checksum import sha256_of_file as _cached_sha256_of_file

if any(name == "multimolecule" or name.startswith("multimolecule.") for name in sys.modules):
    raise RuntimeError("fixture generators must not import the downstream multimolecule package")

INPUTS_FILENAME = "inputs.safetensors"
EXPECTED_FILENAME = "expected.safetensors"
META_FILENAME = "meta.json"
ARTIFACT_FILENAMES = (INPUTS_FILENAME, EXPECTED_FILENAME, META_FILENAME)
REQUIRED_META_FIELDS = frozenset(
    {
        "version",
        "model",
        "case",
        "outputs",
        "tolerance",
        "inputs_source",
        "upstream",
    }
)
OPTIONAL_META_FIELDS = frozenset({"auto_model"})
UPSTREAM_META_FIELDS = frozenset(
    {
        "repository",
        "commit",
        "checkpoint_source",
        "checkpoint_sha256",
        "target_slice",
    }
)
INPUTS_SOURCE_FIELDS = {
    "corpus_crop": frozenset({"type", "id", "record_id", "crop", "sha256"}),
    "corpus_variant_pair": frozenset({"type", "id", "anchor_record_id", "crop", "ref_sha256", "alt_sha256", "variant"}),
    "synthetic_sequence_probe": frozenset({"type", "id"}),
    "synthetic_token_probe": frozenset({"type", "id"}),
}
__all__ = [
    "ARTIFACT_FILENAMES",
    "FixtureError",
    "fixture_out_dir",
    "inputs_source_from_anchor_crop",
    "inputs_source_from_variant_pair",
    "sha256_of_file",
    "validate_fixture",
    "write_fixture_artifacts",
]

ALLOWED_INPUTS_KEYS = frozenset(
    {
        frozenset({"input_ids"}),
        frozenset({"input_ids", "attention_mask"}),
        frozenset({"input_ids", "attention_mask", "features"}),
        frozenset({"input_ids", "alternative_input_ids"}),
        frozenset({"one_hot"}),
    }
)


class FixtureError(RuntimeError):
    """Raised when a fixture cannot be written or validated."""


class FixtureContractError(FixtureError):
    """Raised when fixture metadata or tensor key sets violate the contract."""


class FixtureChecksumError(FixtureError):
    """Raised when a fixture checksum does not match the declared value."""


def fixture_out_dir(repo_root: Path, model: str, case: str) -> Path:
    """Return the standard `out/<model>/<case>` fixture directory."""
    return repo_root / "out" / model / case


def sha256_of_file(path: Path) -> str:
    """Return the sha256 hex digest for `path`, using the shared checksum cache."""
    try:
        return _cached_sha256_of_file(path)
    except ChecksumError as error:
        raise FixtureChecksumError(str(error)) from error


def _coerce_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(meta, Mapping):
        return copy.deepcopy(dict(meta))
    raise FixtureContractError(f"meta must be a mapping, got {type(meta).__name__}")


def _normalize_meta(payload: dict[str, Any]) -> dict[str, Any]:
    keep_top_level = REQUIRED_META_FIELDS | OPTIONAL_META_FIELDS
    payload = {key: copy.deepcopy(payload[key]) for key in keep_top_level if key in payload}

    inputs_source = payload.get("inputs_source")
    if isinstance(inputs_source, dict):
        source_type = inputs_source.get("type")
        keep = INPUTS_SOURCE_FIELDS.get(source_type)
        if keep is not None:
            payload["inputs_source"] = {key: copy.deepcopy(inputs_source[key]) for key in keep if key in inputs_source}

    upstream = payload.get("upstream")
    if not isinstance(upstream, dict):
        return payload

    payload["upstream"] = {key: copy.deepcopy(upstream[key]) for key in UPSTREAM_META_FIELDS if key in upstream}
    return payload


def inputs_source_from_anchor_crop(
    crop: Mapping[str, Any],
    *,
    crop_name: str | None = None,
) -> dict[str, Any]:
    """Build the canonical inputs_source payload for a single corpus crop."""
    crop_meta = copy.deepcopy(dict(_require_mapping(crop.get("crop"), label="crop.crop")))
    crop_meta.pop("sha256", None)
    if crop_name is not None:
        crop_meta = {"name": crop_name, **crop_meta}
    payload: dict[str, Any] = {
        "type": "corpus_crop",
        "id": crop["id"],
        "record_id": crop["record_id"],
        "crop": crop_meta,
        "sha256": crop["sha256"],
    }
    return payload


def inputs_source_from_variant_pair(
    crop: Mapping[str, Any],
    *,
    crop_name: str | None = None,
) -> dict[str, Any]:
    """Build the canonical inputs_source payload for a corpus variant-pair crop."""
    crop_meta = dict(_require_mapping(crop.get("crop"), label="crop.crop"))
    crop_meta.pop("sha256", None)
    name = crop_name if crop_name is not None else crop_meta.pop("name", None)
    variant_index = int(crop["position_in_anchor"]) - int(crop_meta["requested_start"])
    payload: dict[str, Any] = {
        "type": "corpus_variant_pair",
        "id": crop["id"],
        "anchor_record_id": crop["anchor_record_id"],
        "crop": {"name": name, **crop_meta},
        "ref_sha256": crop["ref_sha256"],
        "alt_sha256": crop["alt_sha256"],
        "variant": {
            "index": variant_index,
            "position_in_anchor": crop["position_in_anchor"],
            "reference": crop["ref_allele"],
            "alternative": crop["alt_allele"],
        },
    }
    return payload


def _validate_fixture_inputs(inputs_keys: set[str]) -> None:
    """Validate fixture input tensor key sets."""
    errors = []
    input_key_set = frozenset(inputs_keys)
    if input_key_set not in ALLOWED_INPUTS_KEYS:
        errors.append(f"unknown inputs key set: {sorted(inputs_keys)}")
    if errors:
        raise FixtureContractError(_format_errors(errors))


def _format_errors(errors: list[str]) -> str:
    return "\n".join(f"- {error}" for error in errors)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _tensor_keys(tensors: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    _require_mapping(tensors, label=label)
    keys = tuple(tensors.keys())
    errors = []
    if not keys:
        errors.append(f"{label} must contain at least one tensor")
    for key in keys:
        if not isinstance(key, str) or not key:
            errors.append(f"{label} tensor keys must be non-empty strings; got {key!r}")
    if len(set(keys)) != len(keys):
        errors.append(f"{label} tensor keys must be unique")
    if errors:
        raise FixtureError(_format_errors(errors))
    return keys


def _infer_identity(case_dir: Path) -> tuple[str | None, str | None]:
    if not case_dir.name:
        return None, None
    model = case_dir.parent.name or None
    return model, case_dir.name


def _validate_meta(
    meta: Mapping[str, Any],
    *,
    expected_outputs: set[str] | tuple[str, ...] | list[str] | None = None,
    expected_model: str | None = None,
    expected_case: str | None = None,
    meta_path: Path | None = None,
) -> None:
    """Validate the top-level shape of fixture metadata."""
    label = str(meta_path) if meta_path is not None else "meta"
    errors: list[str] = []
    if not isinstance(meta, Mapping):
        raise FixtureError(f"{label}: meta must be an object")

    missing = REQUIRED_META_FIELDS - meta.keys()
    if missing:
        errors.append(f"{label}: missing fields {sorted(missing)}")

    if meta.get("version") != 1:
        errors.append(f"{label}: version must be 1")

    model = meta.get("model")
    if not isinstance(model, str) or not model:
        errors.append(f"{label}: model must be a non-empty string")
    elif expected_model is not None and model != expected_model:
        errors.append(f"{label}: model={model!r} != {expected_model!r}")

    case = meta.get("case")
    if not isinstance(case, str) or not case:
        errors.append(f"{label}: case must be a non-empty string")
    elif expected_case is not None and case != expected_case:
        errors.append(f"{label}: case={case!r} != {expected_case!r}")

    outputs = meta.get("outputs")
    if not isinstance(outputs, list) or not outputs or not all(isinstance(key, str) and key for key in outputs):
        errors.append(f"{label}: outputs must be a non-empty list of strings")
    elif len(set(outputs)) != len(outputs):
        errors.append(f"{label}: outputs must not contain duplicate names")
    elif expected_outputs is not None:
        observed = set(outputs)
        expected = set(expected_outputs)
        if observed != expected:
            errors.append(f"{label}: outputs {sorted(observed)} != expected tensor keys {sorted(expected)}")

    tolerance = meta.get("tolerance")
    if not isinstance(tolerance, Mapping):
        errors.append(f"{label}: tolerance must be an object")
    else:
        for key in ("atol", "rtol"):
            value = tolerance.get(key)
            if not isinstance(value, (int, float)):
                errors.append(f"{label}: tolerance.{key} must be numeric")

    for key in ("inputs_source", "upstream"):
        if not isinstance(meta.get(key), Mapping):
            errors.append(f"{label}: {key} must be an object")

    auto_model = meta.get("auto_model")
    if auto_model is not None and not isinstance(auto_model, str):
        errors.append(f"{label}: auto_model must be a string when present")

    if errors:
        raise FixtureError(_format_errors(errors))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise FixtureError(f"{path}: invalid JSON: {error}") from error
    except OSError as error:
        raise FixtureError(f"{path}: unable to read JSON: {error}") from error
    if not isinstance(payload, dict):
        raise FixtureError(f"{path}: JSON payload must be an object")
    return payload


def _read_safetensor_keys(path: Path) -> tuple[str, ...]:
    try:
        with safe_open(path, framework="pt") as tensors:
            keys = tuple(tensors.keys())
    except Exception as error:
        raise FixtureError(f"{path}: unable to open safetensors: {error}") from error
    if not keys:
        raise FixtureError(f"{path}: contains no tensors")
    return keys


def _atomic_replace(source: Path, destination: Path, *, digest: str) -> None:
    try:
        os.replace(source, destination)
        _remember_sha256(destination, digest)
    except OSError as error:
        raise FixtureError(f"{destination}: unable to replace file atomically: {error}") from error


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _as_torch_tensor(value: Any, *, label: str, key: str) -> Any:
    try:
        import torch
    except ImportError as error:
        raise FixtureError(f"{label}.{key}: torch is required to write fixture safetensors") from error

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(value, np.ndarray):
        if value.dtype == np.dtype("O"):
            raise FixtureError(f"{label}.{key}: object arrays cannot be written as safetensors")
        return torch.from_numpy(np.ascontiguousarray(value))

    if hasattr(value, "numpy") and callable(value.numpy):
        try:
            return _as_torch_tensor(value.numpy(), label=label, key=key)
        except Exception as error:
            if isinstance(error, FixtureError):
                raise
            raise FixtureError(
                f"{label}.{key}: unable to convert tensor-like value through .numpy(): {error}"
            ) from error

    try:
        return torch.as_tensor(value).detach().cpu().contiguous()
    except Exception as error:
        raise FixtureError(f"{label}.{key}: unable to convert value to torch.Tensor: {error}") from error


def _torch_tensor_mapping(tensors: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    return {key: _as_torch_tensor(value, label=label, key=key) for key, value in tensors.items()}


def _save_safetensors(path: Path, tensors: Mapping[str, Any], *, label: str) -> str:
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise FixtureError(f"{label}: safetensors.torch is required to write tensors") from error

    temporary = _temporary_path(path)
    try:
        save_file(_torch_tensor_mapping(tensors, label=label), str(temporary))
        digest = sha256_of_file(temporary)
        _read_safetensor_keys(temporary)
        _atomic_replace(temporary, path, digest=digest)
        return digest
    except Exception as error:
        temporary.unlink(missing_ok=True)
        if isinstance(error, FixtureError):
            raise
        raise FixtureError(f"{path}: unable to write {label}: {error}") from error


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    temporary = _temporary_path(path)
    try:
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        temporary.write_text(encoded)
        digest = sha256_of_file(temporary)
        _load_json(temporary)
        _atomic_replace(temporary, path, digest=digest)
        return digest
    except Exception as error:
        temporary.unlink(missing_ok=True)
        if isinstance(error, FixtureError):
            raise
        raise FixtureError(f"{path}: unable to write JSON: {error}") from error


def _artifact_summary(path: Path, tensor_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": path, "sha256": sha256_of_file(path)}
    if tensor_keys:
        payload["tensor_keys"] = tensor_keys
    return payload


def _relocate_summary(summary: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    relocated = copy.deepcopy(summary)
    relocated["path"] = out_dir
    relocated["inputs"]["path"] = out_dir / INPUTS_FILENAME
    relocated["expected"]["path"] = out_dir / EXPECTED_FILENAME
    relocated["meta"]["path"] = out_dir / META_FILENAME
    return relocated


def write_fixture_artifacts(
    out_dir: Path,
    *,
    inputs: Mapping[str, Any],
    expected: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a standard `out/<model>/<case>` fixture directory atomically."""
    out_dir = out_dir.expanduser().resolve()
    if len(out_dir.parents) < 3 or out_dir.parent.parent.name != "out":
        raise FixtureError(f"{out_dir}: expected an out/<model>/<case> fixture path")
    model = out_dir.parent.name
    case = out_dir.name
    input_keys = _tensor_keys(inputs, label="inputs")
    expected_keys = _tensor_keys(expected, label="expected")
    payload = _normalize_meta(_coerce_meta(meta))
    _validate_fixture_inputs(set(input_keys))
    _validate_meta(
        payload,
        expected_outputs=expected_keys,
        expected_model=model,
        expected_case=case,
        meta_path=out_dir / META_FILENAME,
    )

    tmp_dir = out_dir.parent / f".{case}.tmp-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        _save_safetensors(tmp_dir / INPUTS_FILENAME, inputs, label="inputs")
        _save_safetensors(tmp_dir / EXPECTED_FILENAME, expected, label="expected")
        _write_json_atomic(tmp_dir / META_FILENAME, payload)
        summary = validate_fixture(tmp_dir, expected_model=model, expected_case=case)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
        return _relocate_summary(summary, out_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def validate_fixture(
    case_dir: Path,
    *,
    expected_model: str | None = None,
    expected_case: str | None = None,
) -> dict[str, Any]:
    """Validate a fixture case directory and return artifact keys and sha256s."""
    case_dir = case_dir.expanduser().resolve()
    inferred_model, inferred_case = _infer_identity(case_dir)
    expected_model = inferred_model if expected_model is None else expected_model
    expected_case = inferred_case if expected_case is None else expected_case

    inputs_path = case_dir / INPUTS_FILENAME
    expected_path = case_dir / EXPECTED_FILENAME
    meta_path = case_dir / META_FILENAME
    missing = [path.name for path in (inputs_path, expected_path, meta_path) if not path.exists()]
    if missing:
        raise FixtureError(f"{case_dir}: missing fixture artifact(s): {', '.join(missing)}")

    input_keys = _read_safetensor_keys(inputs_path)
    expected_keys = _read_safetensor_keys(expected_path)
    meta = _load_json(meta_path)
    _validate_fixture_inputs(set(input_keys))
    _validate_meta(
        meta,
        expected_outputs=expected_keys,
        expected_model=expected_model,
        expected_case=expected_case,
        meta_path=meta_path,
    )

    return {
        "path": case_dir,
        "inputs": _artifact_summary(inputs_path, input_keys),
        "expected": _artifact_summary(expected_path, expected_keys),
        "meta": _artifact_summary(meta_path),
    }
