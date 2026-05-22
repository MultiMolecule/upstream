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


"""Collect environment metadata for upstream fixture generation.

The PyTorch diagnostics are based on `torch.utils.collect_env`. TensorFlow,
Keras, and JAX do not share the same interface, so this module records their
official runtime/build diagnostics separately.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from importlib import metadata
except ImportError:  # pragma: no cover - legacy upstream environments.
    try:
        import importlib_metadata as metadata  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        metadata = None  # type: ignore[assignment]


PACKAGE_NAMES = (
    "torch",
    "tensorflow",
    "tensorflow-cpu",
    "tensorflow-macos",
    "keras",
    "jax",
    "jaxlib",
    "flax",
    "dm-haiku",
    "optax",
    "numpy",
    "scipy",
    "transformers",
    "huggingface-hub",
    "safetensors",
)

ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING",
    "CONDA_PREFIX",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "KERAS_BACKEND",
    "JAX_PLATFORM_NAME",
    "XLA_FLAGS",
    "TF_CPP_MIN_LOG_LEVEL",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


def package_version(name: str) -> str | None:
    if metadata is None:
        return None
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_python_environment() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def collect_pytorch_environment() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    except Exception as error:
        return {"error": repr(error)}

    payload: dict[str, Any] = {
        "version": getattr(torch, "__version__", package_version("torch")),
    }

    try:
        from torch.utils.collect_env import get_env_info

        payload["collect_env"] = _jsonable(get_env_info())
    except Exception as error:
        payload["collect_env_error"] = repr(error)

    return payload


def collect_tensorflow_environment() -> dict[str, Any] | None:
    try:
        import tensorflow as tf
    except ImportError:
        return None
    except Exception as error:
        return {"error": repr(error)}

    payload: dict[str, Any] = {
        "version": getattr(tf, "__version__", package_version("tensorflow")),
    }

    try:
        payload["build_info"] = _jsonable(tf.sysconfig.get_build_info())
    except Exception as error:
        payload["build_info_error"] = repr(error)

    try:
        payload["physical_devices"] = {
            kind: [device.name for device in tf.config.list_physical_devices(kind)]
            for kind in ("CPU", "GPU", "TPU")
        }
    except Exception as error:
        payload["physical_devices_error"] = repr(error)

    return payload


def collect_keras_environment() -> dict[str, Any] | None:
    try:
        import keras
    except ImportError:
        return None
    except Exception as error:
        return {"error": repr(error)}

    payload: dict[str, Any] = {
        "version": getattr(keras, "__version__", package_version("keras")),
        "env_backend": os.environ.get("KERAS_BACKEND"),
    }

    try:
        payload["backend"] = keras.config.backend()
    except Exception:
        try:
            payload["backend"] = keras.backend.backend()
        except Exception as error:
            payload["backend_error"] = repr(error)

    return payload


def collect_jax_environment() -> dict[str, Any] | None:
    try:
        import jax
    except ImportError:
        return None
    except Exception as error:
        return {"error": repr(error)}

    payload: dict[str, Any] = {
        "version": getattr(jax, "__version__", package_version("jax")),
        "jaxlib_version": package_version("jaxlib"),
    }

    try:
        payload["default_backend"] = jax.default_backend()
    except Exception as error:
        payload["default_backend_error"] = repr(error)

    try:
        payload["devices"] = [str(device) for device in jax.devices()]
    except Exception as error:
        payload["devices_error"] = repr(error)

    try:
        payload["environment_info"] = jax.print_environment_info(return_string=True)
    except Exception as error:
        payload["environment_info_error"] = repr(error)

    return payload


def collect_environment() -> dict[str, Any]:
    """Return JSON-serializable environment metadata for fixture `meta.json`."""
    return {
        "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python": collect_python_environment(),
        "packages": {name: package_version(name) for name in PACKAGE_NAMES},
        "pytorch": collect_pytorch_environment(),
        "tensorflow": collect_tensorflow_environment(),
        "keras": collect_keras_environment(),
        "jax": collect_jax_environment(),
        "env": {name: os.environ.get(name) for name in ENV_VARS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON metadata to this file.")
    args = parser.parse_args()

    payload = json.dumps(collect_environment(), indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
