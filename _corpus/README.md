# Canonical Input Corpus

This directory contains stable input records used by upstream fixture
generators. Large physical sequences are fetched remotely and cached on demand;
they are not tracked in git.

The corpus is not the golden artifact. Generated fixtures must store the actual
tokenized tensors in `inputs.safetensors`; corpus files are only the source
material used to create those tensors.

## Files

```text
_corpus/
├── corpus.json
└── load.py
```

`corpus.json` is the metadata catalog. The only physical sequence source is
remote chr21 (`NC_000021.9`), fetched by range into
`~/.cache/multimolecule/upstream/corpus` or `MULTIMOLECULE_CORPUS_CACHE`. DNA,
RNA, protein, and synthetic ref/alt records are all loaded or derived from that
one anchor, which keeps the repository small and makes the faithfulness input
source obvious. `load.py` uses Biopython for FASTA parsing, validates lengths,
anchors, small recommended crops, and variant reference alleles, and provides
shared crop helpers for fixed-length model inputs.

## Usage

List available records:

```bash
python _corpus/load.py --list
```

Load a small full record:

```bash
python _corpus/load.py protein/grch38_chr21_orf
```

Crop a fixed-length DNA window around the chr21 center:

```bash
python _corpus/load.py dna/grch38_chr21 --crop-length 2114 --center center
```

Validate corpus metadata and small recommended crops:

```bash
python _corpus/load.py --check-all
```

Long recommended crops are bounds-checked during `--check-all` and fetched only
when explicitly requested.

Use from a generator:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from _corpus.load import crop_record, load_record_range

crop = crop_record("dna/grch38_chr21", 2114, center="center")
fixed_length_sequence = crop["sequence"]

start = 23354859
end = 23355124
fixed_range_sequence = load_record_range("rna/grch38_chr21_transcribed", start, end)
```

The generator should copy the relevant corpus metadata into `meta.json`:

```json
{
  "inputs_source": {
    "type": "corpus",
    "id": "dna/grch38_chr21",
    "record_id": "grch38_chr21",
    "crop": {
      "center": "center",
      "length": 2114,
      "requested_start": 23353934,
      "requested_end": 23356048
    },
    "sha256": "..."
  }
}
```

## Rules

- Keep physical sequence files out of git unless there is a concrete reason to
  track a tiny local source.
- Store remote accession/fetch metadata and verify fetched ranges by length and
  crop SHA256.
- Use clear provenance and stable accessions or assembly coordinates.
- Validate lengths, anchors, small recommended crops, and variant reference
  alleles with `python _corpus/load.py --check-all`.
- Do not require the main MultiMolecule test suite to parse corpus files.
- If a model needs a modality-specific input, derive it from the chr21 anchor
  unless there is a concrete faithfulness reason to add another physical source.
