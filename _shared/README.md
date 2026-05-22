# Shared Upstream Helpers

This directory contains small utilities shared by upstream fixture generators.

Helpers here must not import MultiMolecule model implementations or the
MultiMolecule package itself. Prefer standard-library and third-party runtime
diagnostics only; shared file parsing belongs in `_corpus/` and should use
third-party parsers such as Biopython.

## collect_env.py

`collect_env.py` collects reproducibility metadata for the `generated` section of
fixture `meta.json` files.

Use it from a generator:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from _shared.collect_env import collect_environment

meta = {
    "generated": collect_environment(),
}
```

Or run it directly:

```bash
python _shared/collect_env.py
python _shared/collect_env.py --output out/env.json
```

The helper records interpreter, platform, selected package versions, selected
environment variables, PyTorch diagnostics from `torch.utils.collect_env`,
TensorFlow build information from `tf.sysconfig.get_build_info()`, Keras backend
configuration, and JAX environment information from
`jax.print_environment_info(return_string=True)`. Missing optional packages are
reported as `null` rather than raising.
