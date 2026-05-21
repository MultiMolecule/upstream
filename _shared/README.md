# Shared Upstream Helpers

This directory contains small utilities shared by upstream fixture generators.

Helpers here must not import MultiMolecule model implementations or the
MultiMolecule package itself. Shared file parsing belongs in `_corpus/` and
should use third-party parsers such as Biopython.

## fixture.py

`fixture.py` is the shared v1 writer/validator for a single fixture case
directory:

```text
out/<model>/<case>/
├── inputs.safetensors
├── expected.safetensors
└── meta.json
```

It keeps the contract narrow:

- write `inputs.safetensors`, `expected.safetensors`, and `meta.json` through
  temporary files, staging whole fixture cases before replacing
  `out/<model>/<case>`;
- compute process-cached sha256 digests for written or validated artifacts;
- validate the top-level `meta.json` shape used by `golden/validate.py`;
- raise `FixtureError` / `FixtureContractError` / `FixtureChecksumError` with
  artifact-specific messages instead of bare safetensors or JSON exceptions.

Use it from a generator after constructing tensors and metadata:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from _shared.fixture import fixture_out_dir, write_fixture_artifacts

meta = {
    "version": 1,
    "model": "example",
    "case": "example",
    "outputs": ["logits"],
    "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    "inputs_source": {"type": "synthetic"},
    "upstream": {"repository": "https://example.org/model", "commit": "abc123"},
}

out_dir = fixture_out_dir(ROOT, "example", "example")
artifacts = write_fixture_artifacts(
    out_dir,
    inputs={"input_ids": input_ids},
    expected={"logits": logits},
    meta=meta,
)
print(artifacts["expected"]["sha256"])
```

Fixture writers always use the safetensors torch backend. Generator tensor maps
may contain torch tensors, numpy arrays, or tensor-like objects with `.numpy()`;
the helper converts them before writing `inputs.safetensors` and
`expected.safetensors`.
Generators should compute the standard output path with
`fixture_out_dir(repo_root, model, case)`, then call
`write_fixture_artifacts(out_dir, inputs=..., expected=..., meta=...)`. The
writer infers `model` and `case` from the standard path and stages the whole
case directory before replacing it. Generators should not import safetensors
writer functions directly.
`validate_fixture(path)` reopens the safetensors files, verifies that
`meta["outputs"]` matches `expected.safetensors` keys, checks required top-level
metadata fields and tolerance shape, then returns a plain mapping with file
paths, tensor keys, and sha256 digests.

## alphabet.py

`alphabet.py` contains the small MultiMolecule DNA/RNA vocabularies used by
tokenizer-parity generators. Use `multimolecule_dna_vocabulary()` or
`multimolecule_rna_vocabulary()` when a fixture needs to remap upstream token
columns into the MultiMolecule token order.

## archive.py

`archive.py` contains `safe_extract_tar()` and `safe_extract_zip()`. They reject
archive members that would extract outside the target directory, reject unsafe
tar links, and keep unsupported tar member types out of generator code. Use these
helpers instead of `extractall()` directly.

## checksum.py

`checksum.py` owns the process-local sha256 cache shared by download and fixture
helpers. Generators should usually import `sha256_of_file()` from
`_shared.fixture`; lower-level helpers use `_shared.checksum` directly when they
need to remember a digest after an atomic rename.

## bert_probe.py

`bert_probe.py` contains common logic for BERT-style masked-LM fixture probes:
forcing eager attention, validating shared vocabulary columns, remapping logits
to the MultiMolecule vocabulary subset, stacking hidden states, and preserving
shared input embeddings. New BERT-family generators should use these helpers
instead of reimplementing vocab-remap logic.

## variant.py

`variant.py` is the shared helper for small synthetic variant-pair fixtures. It
loads and validates `crop_variant_pair()` records, provides DNA id and one-hot
encoders, and builds the canonical `corpus_variant_pair` `inputs_source` block
used by splice and variant-effect generators.

## source_tree.py, download.py, and huggingface.py

`source_tree.py`, `download.py`, and `huggingface.py` let generators fetch
original upstream materials directly from their upstream distribution points.
They are cache helpers, not a second checkpoint registry: a generator should
still name the original repository, commit, URL, or upstream path it needs.

`source_tree.py` and `download.py` use `$MULTIMOLECULE_UPSTREAM_CACHE`. If the
environment variable is unset, they use `~/.cache/multimolecule-upstream`.

`download.fetch_http_file()` uses a bounded network contract by default: 60 second
timeouts, a MultiMolecule User-Agent, two retries for transient URL errors,
per-target cache locks, progress in interactive terminals, and sha256
verification after the bytes are written. Callers may override `timeout`,
`user_agent`, `retries`, or `progress` without changing the cached-file API.

`download.fetch_google_drive_file()` uses `gdown` for public Google Drive file or
folder artifacts and follows the same cache, lock, environment override, and
sha256 verification contract as `fetch_http_file()`.

`download.fetch_kaggle_file()` uses the Kaggle API for
`kaggle://datasets/<owner>/<dataset>/<file>` artifacts. Users must configure
`KAGGLE_USERNAME` and `KAGGLE_KEY`, or provide the standard
`~/.kaggle/kaggle.json` or `~/.kaggle/credentials.json` credentials file.

`huggingface.hf_snapshot_dir()` resolves pinned Hub repositories through the
standard Hugging Face cache and may take a model-specific environment override
for offline local snapshots.

`source_tree.ensure_source_tree()` requires at least one sparse checkout path.
It clones into a temporary directory under a cache lock, then renames the
completed checkout into place so interrupted clones do not look like valid
cache hits. Git clone/fetch/checkout failures include the repository, commit,
and cache path in the raised error.

Use it from a generator:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from _shared.download import fetch_google_drive_file, fetch_kaggle_file, fetch_http_file
from _shared.huggingface import hf_snapshot_dir
from _shared.source_tree import ensure_source_tree

source_root = ensure_source_tree(
    "https://github.com/calico/basenji",
    "06ce5d387e20b47184d05433b3983163c5f923cd",
    ("basenji", "manuscripts/cross2020/params_human.json"),
    env_var="MULTIMOLECULE_UPSTREAM_BASENJI_SOURCE",
    cache_prefix="basenji",
)
checkpoint = fetch_http_file(
    "https://storage.googleapis.com/basenji_barnyard2/model_human.h5",
    "model_human.h5",
    cache_prefix="basenji/barnyard2",
    env_var="MULTIMOLECULE_UPSTREAM_BASENJI_CHECKPOINT",
    description="Basenji Cross2020 human Keras checkpoint",
)
google_checkpoint = fetch_google_drive_file(
    "google_drive://1sT6jlv9vrpX0npKmnbFeOqZ1JZDrZTQ2/RNABERT_pretrained.pth",
    "RNABERT_pretrained.pth",
    cache_prefix="rnabert",
    env_var="MULTIMOLECULE_UPSTREAM_RNABERT_CHECKPOINT",
    sha256="c0038a6672191bcb5517f49be94db4692769e6bddcf1dc971a03688c64105d51",
    description="RNABERT official checkpoint",
)
kaggle_checkpoint = fetch_kaggle_file(
    "kaggle://datasets/shujun717/ribonanzanet-weights/RibonanzaNet.pt",
    "RibonanzaNet.pt",
    cache_prefix="ribonanzanet",
    env_var="MULTIMOLECULE_UPSTREAM_RIBONANZANET_CHECKPOINT",
    sha256="c2aa45c14367863ece52d528d6c353ef40b66f7cb41539c19a042e87c7d3f215",
    description="RibonanzaNet official checkpoint",
)
snapshot_root = hf_snapshot_dir(
    "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
    revision="d79fa37e2cd62dd338103c630f95be8f90812d46",
    allow_patterns=("*.json", "*.py", "model.safetensors"),
    env_var="MULTIMOLECULE_UPSTREAM_HYENADNA_SOURCE",
)
```

Generator contract:

- A generator may fetch original upstream files or source trees through these
  helpers.
- A generator may expose model-specific explicit environment overrides for
  local debugging or offline runs.
- `source.yaml` checkpoint/source fields must name original upstream URLs,
  pinned repositories, or cache-resolved artifacts. `local://pretrained/...`,
  `local://parity/...`, and relative `pretrained/...` or `parity/...` defaults
  are migration debt.
- A generator must not default to the main MultiMolecule checkout's
  `pretrained/` or `parity/` directories.
- A generator must not build Docker images, promote fixtures, compare outputs,
  or run disk safety checks during `generate.py`.

Check the current migration state with:

```bash
python run.py doctor --family basenji
python run.py doctor --all --strict-contract
```

Plain `doctor` reports contract violations as warnings so existing scaffolds can
continue to be inspected. `--strict-contract` makes the same warnings return a
non-zero exit code for focused cleanups.
