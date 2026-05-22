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
│   ├── download.py
│   ├── fixture.py
│   └── source_tree.py
└── models/
    └── <family>/
        ├── source.yaml
        ├── generate.py
        └── <case>/generate.py  # only for genuinely per-case generators
```

The default generator path is `models/<family>/generate.py`. A nested
`models/<family>/<case>/generate.py` path is reserved for cases that need
different generator implementations.

## Quick Start

Install the lightweight shared dependencies, then list available generators:

```shell
/opt/conda/bin/python -m pip install -r requirements.txt
/opt/conda/bin/python run.py list --all
```

Run one checkpoint fixture:

```shell
/opt/conda/bin/python run.py generate --checkpoint hyenadna-tiny
```

If Docker is unavailable, force the local Python runtime:

```shell
/opt/conda/bin/python run.py generate --checkpoint hyenadna-tiny --no-docker
```

This works for generators whose upstream dependencies are available in the local
environment. Generators with fragile pinned runtimes should keep using their
declared Docker image.

Run every checkpoint fixture declared for one model:

```shell
/opt/conda/bin/python run.py generate --family openspliceai
```

The generator writes `out/<model>/<checkpoint-id>/`. To copy the generated
artifacts into a sibling `golden` checkout, validate them, and remove the
temporary `out/` copy:

```shell
/opt/conda/bin/python run.py golden \
  --checkpoint hyenadna-tiny \
  --golden-root ../golden \
  --replace \
  --clean-output
```

Generators should read upstream code and checkpoints from explicit local paths
or documented environment variables. Do not build a model-specific Docker image
or download a second copy of a checkpoint when the source artifact is already
available locally.

Model-specific overrides use
`MULTIMOLECULE_UPSTREAM_<CASE_OR_FAMILY>_<KIND>`, for example
`MULTIMOLECULE_UPSTREAM_BASENJI_CHECKPOINT` or
`MULTIMOLECULE_UPSTREAM_RNAFM_MRNA_CHECKPOINT`. `run.py` forwards this prefix
into Docker containers when the variable is set in the host environment.

Public Google Drive artifacts are downloaded through `gdown` and cached under
`$MULTIMOLECULE_UPSTREAM_CACHE`. Kaggle artifacts are downloaded through the
Kaggle API; set `KAGGLE_USERNAME` and `KAGGLE_KEY`, or provide the standard
`~/.kaggle/kaggle.json` credentials file.

For models that require a pinned upstream runtime, run through the declared
Docker image:

```shell
/opt/conda/bin/python run.py golden \
  --checkpoint ernierna-ss \
  --docker \
  --golden-root ../golden \
  --replace \
  --clean-output
```

`run.py generate --docker` and `run.py golden --docker` mount the workspace at
`/work` and run from `/work/upstream`. `source.yaml` declares the public
`docker.image` tag and, when needed, the `docker.dockerfile` that can rebuild
that same tag locally. This keeps local development and public audit runs on one
image name.

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

## source.yaml

Each model should include a `source.yaml` file. The top-level `upstream` block
records the original code or project provenance. Checkpoint records describe the
original artifact locations used by the generator.

```yaml
model: ernierna
upstream:
  repository: https://github.com/Bruce-ywj/ERNIE-RNA
  commit: null
checkpoints:
  ernierna:
    source: huggingface://multimolecule-upstream/ernierna-raw@v1
    sha256: null
```

Use `null` for fields that are not known yet. The point is to make missing
provenance explicit rather than implicit.

`checkpoints:` is always a map keyed by the fixture case id:

```yaml
checkpoints:
  openspliceai-mane-80nt:
    source: https://example.org/model_80nt_rs10.pt
    sha256: null
generator:
  script: generate.py
```

`sha256` is the digest value that the generated `meta.json` records as
`upstream.checkpoint_sha256`. For multi-file upstream distributions, compute a
deterministic aggregate digest in the generator and store that value here.

Supported source prefixes are:

- `https://` or `http://` for direct downloads;
- `huggingface://<repo>@<revision>/<path>` for pinned Hub snapshots;
- `google_drive://<file-or-folder-id>/<path>` for public Drive artifacts;
- `kaggle://datasets/<owner>/<dataset>/<file>` for Kaggle dataset files;
- `zenodo://...` and `figshare://...` when a generator resolves the archive;
- `upstream-file://...` for artifacts shipped inside the pinned upstream source tree.

When `generator.script` is present, `run.py` expands every key in
`checkpoints` into one runnable fixture case and invokes the script with the
case id as its first argument. Use `generator.case_argument` or
`generator.arguments` only when the shared generator needs a different command
line shape.

Run the manifest and generator contract checker before submitting changes:

```shell
/opt/conda/bin/python run.py doctor --all --strict-contract
```

## Generated Artifacts

Every `generate.py` should write one fixture case with this shape:

```text
out/<model>/<case>/
├── inputs.safetensors
├── expected.safetensors
└── meta.json
```

Use the fixture case id declared in `source.yaml` as `<case>`. The main
MultiMolecule repository may decide which cases are published as checkpoints,
but this repository should remain self-contained and auditable without reading
scripts from the main checkout.

`meta.json` should include at least:

```json
{
  "version": 1,
  "model": "ernierna",
  "case": "ernierna",
  "auto_model": "AutoModel",
  "outputs": ["last_hidden_state", "pooler_output"],
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
    "checkpoint_source": "huggingface://multimolecule-upstream/ernierna-raw@v1",
    "checkpoint_sha256": "..."
  }
}
```

Outputs with repeated structure, such as per-layer hidden states, should be
saved as a single stacked tensor when possible. Attention maps are intentionally
not part of the standard golden surface because their `N x N` shape grows
quickly; compare logits, task outputs, hidden states, or embeddings unless a
model has a specific audit reason to expose another tensor.

Use canonical tensor key names:

- inputs: `input_ids`, `attention_mask`, `one_hot`, and case-specific auxiliary
  inputs only when the original model truly requires them, such as
  `alternative_input_ids` for paired variant scoring or `features` for tabular
  covariates;
- expected outputs: `logits`, `hidden_states`, `last_hidden_state`,
  `pooler_output`, `vocab_embeddings`, or task-specific names such as
  `profile_logits`, `count_logits`, `contact_map`, `logits_lm`, `logits_sa`,
  and `logits_ss`.

If the fixture must exercise a task head, set `auto_model` to the loader the
consumer should use, such as `AutoModelForTokenPrediction`. Omit it only when
plain `AutoModel` is correct.

## Checkpoint Scope

Do not maintain a static checkpoint matrix in this repository. The fixture cases
declared in `source.yaml` are the upstream audit surface; the main
MultiMolecule repository decides which converted checkpoints are published. The
durable rules are:

- one golden case validates one source checkpoint;
- repeated random seeds are not separate MultiMolecule releases unless the
  release intentionally exposes them as separate checkpoints;
- ensembles are one checkpoint when the MultiMolecule checkpoint intentionally
  represents the ensemble as one model;
- species, context, task, output head, tokenizer size, and architecture size are
  separate checkpoint identities only when MultiMolecule publishes them as
  separate Hub repositories.

Prefer shared upstream runtime images over one Dockerfile per model. A
model-specific image should be the exception for fragile upstream environments,
patched source trees, or baked checkpoint provenance.

## Workflow

1. Add or update the model-specific upstream source metadata.
2. Add a deterministic `generate.py` for one case.
3. Run the generator through `run.py` locally or, for fragile upstream
   environments, in a shared upstream Docker image.
4. Review the generated tensors and metadata.
5. Open a pull request to
   [MultiMolecule/golden](https://github.com/MultiMolecule/golden).
6. After the golden PR is merged, bump the `golden` submodule pointer in the
   main MultiMolecule repository.

The first model should be treated as a spike. Keep the implementation small and
copy only the abstractions that survive the second model.
