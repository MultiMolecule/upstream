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


"""Load canonical corpus records for upstream fixture generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from Bio import SeqIO

CORPUS_ROOT = Path(__file__).resolve().parent
CORPUS_METADATA_PATH = CORPUS_ROOT / "corpus.json"
DEFAULT_CACHE_ROOT = Path("~/.cache/multimolecule/upstream/corpus").expanduser()
VALIDATION_FETCH_LIMIT = 20_000
STANDARD_GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def sequence_sha256(sequence: str) -> str:
    """Return the SHA256 digest for a normalized biological sequence."""
    normalized = "".join(sequence.split()).upper()
    return hashlib.sha256(normalized.encode()).hexdigest()


def default_pad_symbol(record: dict[str, Any]) -> str:
    """Return the default padding symbol for a corpus record."""
    if record["molecule"] == "protein":
        return "X"
    return "N"


def load_corpus_metadata() -> dict[str, Any]:
    """Load corpus metadata."""
    return json.loads(CORPUS_METADATA_PATH.read_text())


def cache_root() -> Path:
    """Return corpus cache root."""
    return Path(os.environ.get("MULTIMOLECULE_CORPUS_CACHE", DEFAULT_CACHE_ROOT)).expanduser()


def find_record_metadata(record_id: str) -> dict[str, Any]:
    """Return corpus metadata for a record id."""
    metadata = load_corpus_metadata()
    matches = [record for record in metadata["records"] if record["id"] == record_id]
    if not matches:
        raise KeyError(f"Unknown corpus record: {record_id}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate corpus record: {record_id}")
    return dict(matches[0])


def read_fasta_sequence(path: Path, record_id: str) -> str:
    """Read one FASTA record by id."""
    fasta_records = {}
    for fasta in SeqIO.parse(path, "fasta"):
        if not fasta.id:
            raise ValueError(f"FASTA record without id in {path}")
        fasta_records[fasta.id] = str(fasta.seq)
    return fasta_records[record_id]


def read_single_fasta_sequence(path: Path) -> str:
    """Read a FASTA file that should contain one record."""
    records = list(SeqIO.parse(path, "fasta"))
    if len(records) != 1:
        raise ValueError(f"{path}: expected one FASTA record, found {len(records)}")
    return str(records[0].seq)


def ncbi_efetch_url(record: dict[str, Any], source_start: int, source_end: int) -> str:
    """Return the NCBI EFetch URL for a 0-based half-open source range."""
    fetch = record["fetch"]
    accession = fetch["id"]
    db = fetch.get("db", "nuccore")
    strand = int(fetch.get("strand", 1))
    if strand != 1:
        raise ValueError(f"{record['id']}: only plus-strand remote fetches are supported")
    base_start = int(fetch["seq_start"])
    seq_start = base_start + source_start
    seq_stop = base_start + source_end - 1
    return (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db={db}&id={accession}&seq_start={seq_start}&seq_stop={seq_stop}"
        f"&strand={strand}&rettype=fasta&retmode=text"
    )


def fetch_ncbi_efetch_chunks(record: dict[str, Any], source_start: int, source_end: int, path: Path) -> None:
    """Fetch a NCBI nucleotide range into a FASTA file in fixed-size chunks."""
    fetch = record["fetch"]
    chunk_size = int(fetch.get("chunk_size", 1_000_000))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        handle.write(
            f">{record['record_id']} molecule={record['molecule']} "
            f"source={fetch['id']}:{source_start}-{source_end}\n"
        )
        line = ""
        for chunk_start in range(source_start, source_end, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, source_end)
            url = ncbi_efetch_url(record, chunk_start, chunk_stop)
            chunk_tmp = tmp_path.with_suffix(f".{chunk_start}.{chunk_stop}.fasta")
            output = subprocess.check_output(
                [
                    "curl",
                    "-fsSL",
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    "120",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "2",
                    url,
                ],
                text=True,
            )
            chunk_tmp.write_text(output)
            try:
                chunk = read_single_fasta_sequence(chunk_tmp).upper()
            finally:
                chunk_tmp.unlink(missing_ok=True)
            for char in chunk:
                line += char
                if len(line) == 80:
                    handle.write(line + "\n")
                    line = ""
        if line:
            handle.write(line + "\n")
    tmp_path.replace(path)


def ensure_remote_sequence(record: dict[str, Any]) -> Path:
    """Return cached FASTA path for a remote sequence, downloading when needed."""
    path = cache_root() / f"{record['record_id']}.fasta"
    if path.exists():
        return path
    fetch = record.get("fetch", {})
    if fetch.get("type") != "ncbi_efetch":
        raise ValueError(f"{record['id']}: unsupported fetch type {fetch.get('type')!r}")
    fetch_ncbi_efetch_chunks(record, 0, int(record["length"]), path)
    return path


def remote_range_path(record: dict[str, Any], source_start: int, source_end: int) -> Path:
    """Return the cache path for a 0-based half-open source range."""
    return cache_root() / "ranges" / record["record_id"] / f"{source_start}_{source_end}.fasta"


def fetch_remote_range(record: dict[str, Any], source_start: int, source_end: int) -> str:
    """Return a remote source range, caching the fetched FASTA."""
    if source_start < 0 or source_end < source_start or source_end > int(record["length"]):
        raise ValueError(f"{record['id']}: invalid source range {source_start}:{source_end}")
    if source_start == source_end:
        return ""
    fetch = record.get("fetch", {})
    if fetch.get("type") != "ncbi_efetch":
        raise ValueError(f"{record['id']}: unsupported fetch type {fetch.get('type')!r}")
    path = remote_range_path(record, source_start, source_end)
    if not path.exists():
        fetch_ncbi_efetch_chunks(record, source_start, source_end, path)
    sequence = read_single_fasta_sequence(path).upper()
    expected_length = source_end - source_start
    if len(sequence) != expected_length:
        raise ValueError(f"{path}: expected {expected_length} bases, got {len(sequence)}")
    return sequence


def load_record_range(record_id: str, start: int, end: int) -> str:
    """Return a 0-based half-open range from a corpus sequence record."""
    record = find_record_metadata(record_id)
    if start < 0 or end < start or end > int(record["length"]):
        raise ValueError(f"{record_id}: invalid range {start}:{end}")
    kind = record.get("kind")
    if kind == "anchor_sequence":
        if "fetch" in record:
            return fetch_remote_range(record, start, end).upper()
        return get_anchor_record(record_id)["sequence"][start:end].upper()
    if kind != "derived_sequence":
        raise ValueError(f"{record_id}: expected sequence record, got {kind!r}")

    transform = record["transform"]
    if transform == "transcribe":
        source = find_record_metadata(record["source_record_id"])
        source_start = int(record.get("source_start", 0)) + start
        source_end = int(record.get("source_start", 0)) + end
        if source.get("kind") == "anchor_sequence" and "fetch" in source:
            return transcribe_dna(fetch_remote_range(source, source_start, source_end))
        return transcribe_dna(get_record(record["source_record_id"])["sequence"][source_start:source_end])
    return get_derived_record(record_id)["sequence"][start:end].upper()


def get_anchor_record(record_id: str) -> dict[str, Any]:
    """Return anchor-sequence corpus metadata plus sequence for a record id."""
    record = find_record_metadata(record_id)
    if record.get("kind") != "anchor_sequence":
        raise ValueError(f"{record_id} is not an anchor sequence")
    if "file" in record:
        path = CORPUS_ROOT / record["file"]
    elif "fetch" in record:
        path = ensure_remote_sequence(record)
    else:
        raise ValueError(f"{record_id}: anchor sequence needs file or fetch metadata")
    sequence = read_fasta_sequence(path, record["record_id"])
    recorded_length = record.get("length")
    if recorded_length is not None and recorded_length != len(sequence):
        raise ValueError(f"Length mismatch for {record_id}: expected {recorded_length}, got {len(sequence)}")
    digest = sequence_sha256(sequence)
    recorded_digest = record.get("sha256")
    if recorded_digest is not None and recorded_digest != digest:
        raise ValueError(f"SHA256 mismatch for {record_id}: expected {recorded_digest}, got {digest}")
    record["sequence"] = sequence
    record["sha256"] = digest
    return record


def transcribe_dna(sequence: str) -> str:
    """Return RNA sequence by replacing thymine with uracil."""
    return sequence.upper().replace("T", "U")


def translate_dna(sequence: str) -> str:
    """Translate DNA with the standard genetic code."""
    sequence = sequence.upper()
    if len(sequence) % 3:
        raise ValueError("translated DNA length must be divisible by 3")
    amino_acids = []
    for index in range(0, len(sequence), 3):
        codon = sequence[index : index + 3]
        try:
            amino_acids.append(STANDARD_GENETIC_CODE[codon])
        except KeyError as error:
            raise ValueError(f"unsupported codon {codon!r} at offset {index}") from error
    return "".join(amino_acids)


def get_derived_record(record_id: str) -> dict[str, Any]:
    """Return corpus metadata plus a sequence derived from another corpus record."""
    record = find_record_metadata(record_id)
    if record.get("kind") != "derived_sequence":
        raise ValueError(f"{record_id} is not a derived sequence")
    source_metadata = find_record_metadata(record["source_record_id"])
    transform = record["transform"]
    if "source_start" in record and "source_length" in record:
        start = int(record["source_start"])
        end = start + int(record["source_length"])
        if source_metadata.get("kind") == "anchor_sequence" and "fetch" in source_metadata:
            source_sequence = fetch_remote_range(source_metadata, start, end)
            source_sha256 = source_metadata.get("sha256")
        else:
            source = get_record(record["source_record_id"])
            source_sequence = source["sequence"][start:end]
            source_sha256 = source["sha256"]
    else:
        source = get_record(record["source_record_id"])
        source_sequence = source["sequence"]
        source_sha256 = source["sha256"]
    if transform == "transcribe":
        sequence = transcribe_dna(source_sequence)
    elif transform == "translate":
        sequence = translate_dna(source_sequence)
    else:
        raise ValueError(f"{record_id}: unsupported transform {transform!r}")
    recorded_length = record.get("length")
    if recorded_length is not None and recorded_length != len(sequence):
        raise ValueError(f"Length mismatch for {record_id}: expected {recorded_length}, got {len(sequence)}")
    digest = sequence_sha256(sequence)
    recorded_digest = record.get("sha256")
    if recorded_digest is not None and recorded_digest != digest:
        raise ValueError(f"SHA256 mismatch for {record_id}: expected {recorded_digest}, got {digest}")
    record["sequence"] = sequence
    record["sha256"] = digest
    record["source_sha256"] = source_sha256
    return record


def get_record(record_id: str) -> dict[str, Any]:
    """Return corpus metadata plus sequence for a corpus record id."""
    record = find_record_metadata(record_id)
    if record.get("kind") == "anchor_sequence":
        return get_anchor_record(record_id)
    if record.get("kind") == "derived_sequence":
        return get_derived_record(record_id)
    raise ValueError(f"{record_id} is not a sequence record")


def get_variant_pair(record_id: str) -> tuple[str, str, dict[str, Any]]:
    """Return reference and alternative anchor sequences for a variant-pair record."""
    record = find_record_metadata(record_id)
    if record.get("kind") != "variant_pair":
        raise ValueError(f"{record_id} is not a variant pair")
    anchor = get_record(record["anchor_record_id"])
    ref_sequence = anchor["sequence"]
    position = int(record["position_in_anchor"])
    ref_allele = record["ref_allele"].upper()
    alt_allele = record["alt_allele"].upper()
    if len(ref_allele) != len(alt_allele):
        raise ValueError(f"{record_id}: ref and alt alleles must have equal length")
    if position < 0 or position + len(ref_allele) > len(ref_sequence):
        raise ValueError(f"{record_id}: position_in_anchor is outside the anchor sequence")
    observed = ref_sequence[position : position + len(ref_allele)].upper()
    if observed != ref_allele:
        raise ValueError(
            f"{record_id}: ref_allele={ref_allele!r} does not match anchor sequence "
            f"{observed!r} at position {position}"
        )
    alt_sequence = ref_sequence[:position] + alt_allele + ref_sequence[position + len(ref_allele) :]
    metadata = dict(record)
    metadata["anchor_sha256"] = anchor["sha256"]
    metadata["ref_sha256"] = sequence_sha256(ref_sequence)
    metadata["alt_sha256"] = sequence_sha256(alt_sequence)
    return ref_sequence, alt_sequence, metadata


def record_length(record: dict[str, Any]) -> int:
    """Return sequence length from loaded sequence or metadata."""
    if "sequence" in record:
        return len(record["sequence"])
    return int(record["length"])


def resolve_anchor(record: dict[str, Any], anchor: str | int) -> int:
    """Resolve a named or integer anchor to a 0-based sequence index."""
    if isinstance(anchor, int):
        return anchor
    if anchor == "center":
        return record_length(record) // 2
    try:
        return int(record["anchors"][anchor])
    except KeyError as error:
        raise KeyError(f"Unknown anchor {anchor!r} for {record['id']}") from error


def crop_plan(
    record: dict[str, Any],
    length: int,
    *,
    center: int | None = None,
    start: int | None = None,
    pad: str = "N",
) -> dict[str, Any]:
    """Return crop coordinates without loading sequence content."""
    if length <= 0:
        raise ValueError("length must be positive")
    if (center is None) == (start is None):
        raise ValueError("exactly one of center or start must be provided")
    requested_start = center - length // 2 if center is not None else start
    if requested_start is None:
        raise AssertionError("requested_start must be resolved")
    requested_end = requested_start + length
    source_length = record_length(record)
    source_start = max(requested_start, 0)
    source_end = min(requested_end, source_length)
    left_pad = max(0, -requested_start)
    right_pad = max(0, requested_end - source_length)
    return {
        "length": length,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "source_start": source_start,
        "source_end": source_end,
        "left_pad": left_pad,
        "right_pad": right_pad,
        "pad": pad,
    }


def assemble_crop(source_sequence: str, crop: dict[str, Any]) -> str:
    """Build a fixed-length crop from a source-range sequence and crop coordinates."""
    expected_source_length = int(crop["source_end"]) - int(crop["source_start"])
    if len(source_sequence) != expected_source_length:
        raise ValueError(f"Expected source range length {expected_source_length}, got {len(source_sequence)}")
    cropped = crop["pad"] * crop["left_pad"] + source_sequence + crop["pad"] * crop["right_pad"]
    if len(cropped) != crop["length"]:
        raise AssertionError(f"Expected crop length {crop['length']}, got {len(cropped)}")
    return cropped


def crop_sequence(
    sequence: str,
    length: int,
    *,
    center: int | None = None,
    start: int | None = None,
    pad: str = "N",
) -> tuple[str, dict[str, Any]]:
    """Crop a fixed-length sequence window, padding when the request crosses bounds."""
    record = {"id": "<sequence>", "sequence": sequence, "length": len(sequence)}
    crop = crop_plan(record, length, center=center, start=start, pad=pad)
    source = sequence[crop["source_start"] : crop["source_end"]]
    cropped = assemble_crop(source, crop)
    crop["sha256"] = sequence_sha256(cropped)
    return cropped, crop


def crop_record(
    record_id: str,
    length: int,
    *,
    center: str | int | None = None,
    start: str | int | None = None,
    pad: str | None = None,
) -> dict[str, Any]:
    """Load a corpus record and return a fixed-length crop plus provenance."""
    metadata = find_record_metadata(record_id)
    if metadata.get("kind") == "anchor_sequence" and "fetch" in metadata:
        return crop_remote_anchor_record(metadata, length, center=center, start=start, pad=pad)
    if metadata.get("kind") == "derived_sequence":
        source_metadata = find_record_metadata(metadata["source_record_id"])
        if (
            metadata["transform"] == "transcribe"
            and source_metadata.get("kind") == "anchor_sequence"
            and "fetch" in source_metadata
        ):
            return crop_transcribed_remote_record(metadata, length, center=center, start=start, pad=pad)
    record = get_record(record_id)
    pad = default_pad_symbol(record) if pad is None else pad
    if center is not None:
        center_index = resolve_anchor(record, center)
        sequence, crop = crop_sequence(record["sequence"], length, center=center_index, pad=pad)
        crop["center"] = center
        crop["center_index"] = center_index
    elif start is not None:
        start_index = resolve_anchor(record, start)
        sequence, crop = crop_sequence(record["sequence"], length, start=start_index, pad=pad)
        crop["start"] = start
        crop["start_index"] = start_index
    else:
        raise ValueError("one of center or start must be provided")
    return {
        "id": record["id"],
        "record_id": record.get("record_id", record["id"]),
        "molecule": record["molecule"],
        "source_sha256": record["sha256"],
        "sequence": sequence,
        "sha256": crop["sha256"],
        "crop": crop,
        "source": {key: record.get(key) for key in ("source", "source_url", "source_query", "reference", "license")},
    }


def crop_remote_anchor_record(
    record: dict[str, Any],
    length: int,
    *,
    center: str | int | None = None,
    start: str | int | None = None,
    pad: str | None = None,
) -> dict[str, Any]:
    """Return a fixed-length crop for a remote anchor without fetching the full sequence."""
    pad = default_pad_symbol(record) if pad is None else pad
    if center is not None:
        center_index = resolve_anchor(record, center)
        crop = crop_plan(record, length, center=center_index, pad=pad)
        crop["center"] = center
        crop["center_index"] = center_index
    elif start is not None:
        start_index = resolve_anchor(record, start)
        crop = crop_plan(record, length, start=start_index, pad=pad)
        crop["start"] = start
        crop["start_index"] = start_index
    else:
        raise ValueError("one of center or start must be provided")
    source = fetch_remote_range(record, crop["source_start"], crop["source_end"])
    sequence = assemble_crop(source, crop)
    crop["sha256"] = sequence_sha256(sequence)
    return {
        "id": record["id"],
        "record_id": record.get("record_id", record["id"]),
        "molecule": record["molecule"],
        "source_sha256": record.get("sha256"),
        "sequence": sequence,
        "sha256": crop["sha256"],
        "crop": crop,
        "source": {key: record.get(key) for key in ("source", "source_url", "source_query", "reference", "license")},
    }


def crop_transcribed_remote_record(
    record: dict[str, Any],
    length: int,
    *,
    center: str | int | None = None,
    start: str | int | None = None,
    pad: str | None = None,
) -> dict[str, Any]:
    """Return a T-to-U crop derived from a remote DNA anchor without fetching the full sequence."""
    source_crop = crop_record(record["source_record_id"], length, center=center, start=start, pad=pad)
    sequence = transcribe_dna(source_crop["sequence"])
    crop = dict(source_crop["crop"])
    crop["sha256"] = sequence_sha256(sequence)
    return {
        "id": record["id"],
        "record_id": record.get("record_id", record["id"]),
        "molecule": record["molecule"],
        "source_sha256": record.get("sha256"),
        "sequence": sequence,
        "sha256": crop["sha256"],
        "crop": crop,
        "source": {key: record.get(key) for key in ("source", "source_url", "source_query", "reference", "license")},
    }


def crop_variant_pair(
    record_id: str,
    length: int,
    *,
    center: str | int | None = None,
    start: str | int | None = None,
    pad: str | None = None,
) -> dict[str, Any]:
    """Load a variant pair and return matched reference and alternative crops."""
    metadata = find_record_metadata(record_id)
    anchor_metadata = find_record_metadata(metadata["anchor_record_id"])
    if anchor_metadata.get("kind") == "anchor_sequence" and "fetch" in anchor_metadata:
        return crop_remote_variant_pair(metadata, anchor_metadata, length, center=center, start=start, pad=pad)
    ref_sequence, alt_sequence, metadata = get_variant_pair(record_id)
    pad = default_pad_symbol(metadata) if pad is None else pad
    anchors = {"variant": metadata["position_in_anchor"], **metadata.get("anchors", {})}
    record = {"id": record_id, "sequence": ref_sequence, "anchors": anchors}
    if center is None and start is None:
        center = "variant"
    if center is not None:
        center_index = resolve_anchor(record, center)
        ref_crop, crop = crop_sequence(ref_sequence, length, center=center_index, pad=pad)
        alt_crop, _ = crop_sequence(alt_sequence, length, center=center_index, pad=pad)
        crop["center"] = center
        crop["center_index"] = center_index
    elif start is not None:
        start_index = resolve_anchor(record, start)
        ref_crop, crop = crop_sequence(ref_sequence, length, start=start_index, pad=pad)
        alt_crop, _ = crop_sequence(alt_sequence, length, start=start_index, pad=pad)
        crop["start"] = start
        crop["start_index"] = start_index
    else:
        raise ValueError("one of center or start must be provided")
    return {
        "id": metadata["id"],
        "kind": metadata["kind"],
        "molecule": metadata["molecule"],
        "anchor_record_id": metadata["anchor_record_id"],
        "position_in_anchor": metadata["position_in_anchor"],
        "ref_allele": metadata["ref_allele"],
        "alt_allele": metadata["alt_allele"],
        "ref_sequence": ref_crop,
        "alt_sequence": alt_crop,
        "ref_sha256": sequence_sha256(ref_crop),
        "alt_sha256": sequence_sha256(alt_crop),
        "crop": crop,
        "source": {key: metadata.get(key) for key in ("source", "source_url", "source_query", "reference", "license")},
    }


def crop_remote_variant_pair(
    metadata: dict[str, Any],
    anchor: dict[str, Any],
    length: int,
    *,
    center: str | int | None = None,
    start: str | int | None = None,
    pad: str | None = None,
) -> dict[str, Any]:
    """Return matched variant crops without materializing the full anchor sequence."""
    pad = default_pad_symbol(metadata) if pad is None else pad
    position = int(metadata["position_in_anchor"])
    ref_allele = metadata["ref_allele"].upper()
    alt_allele = metadata["alt_allele"].upper()
    if len(ref_allele) != len(alt_allele):
        raise ValueError(f"{metadata['id']}: ref and alt alleles must have equal length")
    if position < 0 or position + len(ref_allele) > int(anchor["length"]):
        raise ValueError(f"{metadata['id']}: position_in_anchor is outside the anchor sequence")

    anchors = {"variant": position, **metadata.get("anchors", {})}
    crop_record_metadata = {
        "id": metadata["id"],
        "length": anchor["length"],
        "molecule": metadata["molecule"],
        "anchors": anchors,
    }
    if center is None and start is None:
        center = "variant"
    if center is not None:
        center_index = resolve_anchor(crop_record_metadata, center)
        crop = crop_plan(crop_record_metadata, length, center=center_index, pad=pad)
        crop["center"] = center
        crop["center_index"] = center_index
    elif start is not None:
        start_index = resolve_anchor(crop_record_metadata, start)
        crop = crop_plan(crop_record_metadata, length, start=start_index, pad=pad)
        crop["start"] = start
        crop["start_index"] = start_index
    else:
        raise ValueError("one of center or start must be provided")

    source = fetch_remote_range(anchor, crop["source_start"], crop["source_end"])
    if crop["source_start"] <= position < crop["source_end"]:
        source_offset = position - crop["source_start"]
        observed = source[source_offset : source_offset + len(ref_allele)].upper()
        if observed != ref_allele:
            raise ValueError(
                f"{metadata['id']}: ref_allele={ref_allele!r} does not match anchor sequence "
                f"{observed!r} at position {position}"
            )
    ref_crop = assemble_crop(source, crop)
    alt_chars = list(ref_crop)
    crop_offset = position - crop["requested_start"]
    if 0 <= crop_offset and crop_offset + len(ref_allele) <= len(alt_chars):
        observed = "".join(alt_chars[crop_offset : crop_offset + len(ref_allele)]).upper()
        if observed != ref_allele:
            raise ValueError(
                f"{metadata['id']}: ref_allele={ref_allele!r} does not match crop sequence "
                f"{observed!r} at crop position {crop_offset}"
            )
        alt_chars[crop_offset : crop_offset + len(ref_allele)] = list(alt_allele)
    alt_crop = "".join(alt_chars)
    crop["sha256"] = sequence_sha256(ref_crop)
    return {
        "id": metadata["id"],
        "kind": metadata["kind"],
        "molecule": metadata["molecule"],
        "anchor_record_id": metadata["anchor_record_id"],
        "position_in_anchor": metadata["position_in_anchor"],
        "ref_allele": metadata["ref_allele"],
        "alt_allele": metadata["alt_allele"],
        "ref_sequence": ref_crop,
        "alt_sequence": alt_crop,
        "ref_sha256": sequence_sha256(ref_crop),
        "alt_sha256": sequence_sha256(alt_crop),
        "crop": crop,
        "source": {key: metadata.get(key) for key in ("source", "source_url", "source_query", "reference", "license")},
    }


def validate_sequence_record(record_id: str) -> None:
    """Validate one sequence record without fetching long remote windows unnecessarily."""
    metadata = find_record_metadata(record_id)
    should_materialize = not ("fetch" in metadata or metadata.get("transform") == "transcribe")
    record = get_record(record_id) if should_materialize else metadata
    for name, index in record.get("anchors", {}).items():
        if not isinstance(index, int) or not 0 <= index < record_length(record):
            raise ValueError(f"{record_id}: invalid anchor {name}={index!r}")
    for crop in record.get("recommended_crops", []):
        length = crop["length"]
        pad = crop.get("pad", default_pad_symbol(record))
        if "center" in crop:
            center = resolve_anchor(record, crop["center"])
            if length <= VALIDATION_FETCH_LIMIT:
                crop_record(record_id, length, center=crop["center"], pad=pad)
            else:
                crop_plan(record, length, center=center, pad=pad)
        elif "start" in crop:
            start = resolve_anchor(record, crop["start"])
            if length <= VALIDATION_FETCH_LIMIT:
                crop_record(record_id, length, start=crop["start"], pad=pad)
            else:
                crop_plan(record, length, start=start, pad=pad)
        else:
            raise ValueError(f"{record_id}: recommended crop {crop['name']} has no anchor")


def validate_variant_pair_record(record_id: str) -> None:
    """Load and validate one variant-pair record."""
    metadata = find_record_metadata(record_id)
    anchor_metadata = find_record_metadata(metadata["anchor_record_id"])
    if anchor_metadata.get("kind") == "anchor_sequence" and "fetch" in anchor_metadata:
        ref_sequence = None
        position = int(metadata["position_in_anchor"])
        ref_allele = metadata["ref_allele"].upper()
        observed = fetch_remote_range(anchor_metadata, position, position + len(ref_allele)).upper()
        if observed != ref_allele:
            raise ValueError(
                f"{record_id}: ref_allele={ref_allele!r} does not match anchor sequence "
                f"{observed!r} at position {position}"
            )
    else:
        ref_sequence, _, metadata = get_variant_pair(record_id)
    anchors = {"variant": metadata["position_in_anchor"], **metadata.get("anchors", {})}
    for name, index in anchors.items():
        length = len(ref_sequence) if ref_sequence is not None else int(anchor_metadata["length"])
        if not isinstance(index, int) or not 0 <= index < length:
            raise ValueError(f"{record_id}: invalid anchor {name}={index!r}")
    for crop in metadata.get("recommended_crops", []):
        if "center" in crop:
            if crop["length"] <= VALIDATION_FETCH_LIMIT:
                crop_variant_pair(record_id, crop["length"], center=crop["center"], pad=crop.get("pad"))
            else:
                crop_record_metadata = {
                    "id": record_id,
                    "length": anchor_metadata["length"],
                    "molecule": metadata["molecule"],
                    "anchors": anchors,
                }
                crop_plan(
                    crop_record_metadata,
                    crop["length"],
                    center=resolve_anchor(crop_record_metadata, crop["center"]),
                )
        elif "start" in crop:
            if crop["length"] <= VALIDATION_FETCH_LIMIT:
                crop_variant_pair(record_id, crop["length"], start=crop["start"], pad=crop.get("pad"))
            else:
                crop_record_metadata = {
                    "id": record_id,
                    "length": anchor_metadata["length"],
                    "molecule": metadata["molecule"],
                    "anchors": anchors,
                }
                crop_plan(
                    crop_record_metadata,
                    crop["length"],
                    start=resolve_anchor(crop_record_metadata, crop["start"]),
                )
        else:
            raise ValueError(f"{record_id}: recommended crop {crop['name']} has no anchor")


def validate_corpus() -> int:
    """Load every corpus record and validate metadata against sequence content."""
    metadata = load_corpus_metadata()
    for record in metadata["records"]:
        print(f"checking {record['id']} ...", end=" ", flush=True)
        if record.get("kind") == "variant_pair":
            validate_variant_pair_record(record["id"])
        else:
            validate_sequence_record(record["id"])
        print("OK")
    return len(metadata["records"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_id", nargs="?", help="Corpus record id to print.")
    parser.add_argument("--list", action="store_true", help="List available corpus record ids.")
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Load every record and verify its metadata, length, and sha256.",
    )
    parser.add_argument("--crop-length", type=int, help="Return a fixed-length crop.")
    parser.add_argument("--center", help="Named or integer 0-based center anchor for --crop-length.")
    parser.add_argument("--start", help="Named or integer 0-based start anchor for --crop-length.")
    parser.add_argument("--pad", help="Padding symbol for out-of-bounds crops.")
    args = parser.parse_args()

    if args.list:
        for record in load_corpus_metadata()["records"]:
            print(record["id"])
        return
    if args.check_all:
        count = validate_corpus()
        print(f"\nValidated {count} record(s).")
        return
    if args.record_id is None:
        parser.error("record_id is required unless --list or --check-all is used")
    record = find_record_metadata(args.record_id)
    if args.crop_length is not None:
        center = int(args.center) if args.center is not None and args.center.isdecimal() else args.center
        start = int(args.start) if args.start is not None and args.start.isdecimal() else args.start
        if record.get("kind") == "variant_pair":
            payload = crop_variant_pair(
                args.record_id,
                args.crop_length,
                center=center,
                start=start,
                pad=args.pad,
            )
        else:
            payload = crop_record(
                args.record_id,
                args.crop_length,
                center=center,
                start=start,
                pad=args.pad,
            )
    elif record.get("kind") == "variant_pair":
        ref_sequence, alt_sequence, metadata = get_variant_pair(args.record_id)
        payload = {**metadata, "ref_sequence": ref_sequence, "alt_sequence": alt_sequence}
    else:
        payload = get_record(args.record_id)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
