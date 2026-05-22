# MultiMolecule Upstream

This repository contains the infrastructure used to regenerate MultiMolecule
faithfulness fixtures from original upstream model implementations.

It is a porter-facing repository. End users should use the main
[MultiMolecule](https://github.com/MultiMolecule/multimolecule) library.

## Purpose

MultiMolecule ports many RNA, DNA, and protein models into a shared Python and
Hugging Face ecosystem. For each ported model, we want an auditable answer to a
simple question:

> Given the same deterministic input, does the MultiMolecule implementation
> reproduce the original upstream implementation?

This repository owns the upstream side of that question:

- record where the original code and checkpoints came from;
- preserve enough environment information to rerun upstream inference;
- generate deterministic `inputs.safetensors`, `expected.safetensors`, and
  `meta.json` files;
- open pull requests to
  [MultiMolecule/golden](https://github.com/MultiMolecule/golden).

The main MultiMolecule repository consumes those fixtures. It does not need to
know how each upstream environment was built.

## Repository Layout

```text
upstream/
├── _corpus/
│   ├── README.md
│   ├── corpus.json
│   └── load.py
├── _shared/
│   ├── README.md
│   └── collect_env.py
└── models/
    └── <model>/
        ├── upstream_source.yaml
        ├── notes.md
        ├── Dockerfile
        └── <case>/
            └── generate.py
```

The layout above is the intended stable shape. A model directory may start with
only `upstream_source.yaml` and one `generate.py`; Docker support is optional and
should be added only when the upstream environment is fragile or hard to
recreate.

## Contracts

### No MultiMolecule Model Dependency

Generation scripts must not import MultiMolecule model implementations or the
MultiMolecule package itself. They may use shared helpers from this repository
and third-party packages needed by the upstream implementation.

This keeps the upstream reference independent from the implementation being
tested.

### Single-Sequence Faithfulness

Faithfulness fixtures are generated for `batch_size = 1`.

This is intentional. Some upstream implementations have structural limitations
or known bugs in batched inference. MultiMolecule may correct those batching
paths, but faithfulness fixtures should verify the original single-sequence
behavior. Batch consistency is a separate MultiMolecule internal test.

### Subset Before Saving

If an upstream output is too large, `generate.py` should save a meaningful subset
directly into `expected.safetensors`. The consumer test should only load and
compare tensors; it should not implement model-specific slicing logic.

## upstream_source.yaml

Each model should include an `upstream_source.yaml` file:

```yaml
model: ernierna
upstream:
  repository: https://github.com/Bruce-ywj/ERNIE-RNA
  commit: null
  license: Apache-2.0
checkpoint:
  source: huggingface://multimolecule-upstream/ernierna-raw@v1
  sha256: null
notes:
  - "Record vendored patches, compatibility fixes, or manual setup steps here."
```

Use `null` for fields that are not known yet. The point is to make missing
provenance explicit rather than implicit.

## Generated Artifacts

Every `generate.py` should write one fixture case with this shape:

```text
out/<model>/<case>/
├── inputs.safetensors
├── expected.safetensors
└── meta.json
```

`meta.json` should include at least:

```json
{
  "version": 1,
  "model": "ernierna",
  "case": "default",
  "outputs": ["last_hidden_state", "pooler_output", "attentions"],
  "tolerance": {
    "atol": 0.0001,
    "rtol": 0.0001
  },
  "inputs_source": {
    "type": "corpus_crop",
    "id": "dna/grch38_chr21",
    "record_id": "grch38_chr21",
    "crop": {
      "center": "center",
      "length": 2114,
      "requested_start": 23353934,
      "requested_end": 23356048
    },
    "sha256": "..."
  },
  "upstream": {
    "repository": "https://github.com/Bruce-ywj/ERNIE-RNA",
    "commit": "...",
    "license": "Apache-2.0",
    "checkpoint_source": "huggingface://multimolecule-upstream/ernierna-raw@v1",
    "checkpoint_sha256": "..."
  },
  "generated": {
    "at": "2026-05-22T00:00:00Z",
    "python": {
      "version": "3.11.0",
      "platform": "Linux-..."
    },
    "packages": {
      "torch": "2.4.0",
      "tensorflow": null,
      "jax": null
    },
    "pytorch": {
      "version": "2.4.0",
      "collect_env": {}
    },
    "tensorflow": null,
    "keras": null,
    "jax": null,
    "env": {}
  }
}
```

Use `_shared.collect_env.collect_environment()` to populate the `generated`
section instead of hand-writing interpreter and package metadata in every
generator.

Outputs with repeated structure, such as attention maps, should be saved as a
single stacked tensor when possible. For example, save transformer attentions
under the key `attentions` instead of creating one key per layer.

## Workflow

1. Add or update the model-specific upstream source metadata.
2. Add a deterministic `generate.py` for one case.
3. Run the generator locally or in an upstream Docker image.
4. Review the generated tensors and metadata.
5. Open a pull request to
   [MultiMolecule/golden](https://github.com/MultiMolecule/golden).
6. After the golden PR is merged, bump the `golden` submodule pointer in the
   main MultiMolecule repository.

The first model should be treated as a spike. Keep the implementation small and
copy only the abstractions that survive the second model.
