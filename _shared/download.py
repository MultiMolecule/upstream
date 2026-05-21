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

"""Fetch upstream-distributed files for fixture generators."""

from __future__ import annotations

import contextlib
import inspect
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from _shared.checksum import sha256_of_file

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_USER_AGENT = "MultiMolecule-upstream-fixture/1.0"
PROGRESS_INTERVAL_SECONDS = 5.0
PROGRESS_INTERVAL_BYTES = 64 << 20

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only.
    fcntl = None


def upstream_cache_root() -> Path:
    root = os.environ.get("MULTIMOLECULE_UPSTREAM_CACHE", "~/.cache/multimolecule-upstream")
    return Path(root).expanduser().resolve()


def verify_sha256(path: Path, expected: str | None, *, description: str) -> None:
    if expected is None:
        return
    actual = sha256_of_file(path)
    if actual != expected:
        raise RuntimeError(f"{description} sha256 mismatch: expected {expected}, got {actual}")


def _progress_enabled(progress: bool | None) -> bool:
    if progress is not None:
        return progress
    return sys.stderr.isatty()


@contextlib.contextmanager
def _download_lock(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.lock"
    with lock_path.open("w") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _response_total_bytes(response, initial_bytes: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        if total.isdigit():
            return int(total)
    total_header = response.headers.get("Content-Length")
    if total_header and total_header.isdigit():
        return initial_bytes + int(total_header)
    return None


def _copy_response(
    response,
    destination: Path,
    *,
    label: str,
    progress: bool | None,
    append: bool = False,
) -> None:
    initial_bytes = destination.stat().st_size if append and destination.exists() else 0
    show_progress = _progress_enabled(progress)
    total = _response_total_bytes(response, initial_bytes)
    copied = initial_bytes
    last_report_time = time.monotonic()
    last_report_bytes = initial_bytes

    with destination.open("ab" if append else "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            copied += len(chunk)
            if not show_progress:
                continue
            now = time.monotonic()
            if (
                now - last_report_time < PROGRESS_INTERVAL_SECONDS
                and copied - last_report_bytes < PROGRESS_INTERVAL_BYTES
            ):
                continue
            if total:
                percent = copied / total * 100
                print(
                    f"{label}: downloaded {copied >> 20} MiB / {total >> 20} MiB ({percent:.1f}%)",
                    file=sys.stderr,
                )
            else:
                print(f"{label}: downloaded {copied >> 20} MiB", file=sys.stderr)
            last_report_time = now
            last_report_bytes = copied
    if total is not None and copied != total:
        raise RuntimeError(f"{label} download ended early: expected {total} bytes, got {copied}")


def _temporary_path(parent: Path) -> Path:
    with tempfile.NamedTemporaryFile(dir=parent, delete=False) as handle:
        return Path(handle.name)


def _parse_google_drive_source(source: str) -> tuple[str, str | None]:
    if source.startswith("google_drive://"):
        remainder = source[len("google_drive://") :]
        file_id, separator, inner_path = remainder.partition("/")
        inner_path = inner_path if separator else None
    else:
        file_id = source
        inner_path = None
    if not file_id:
        raise ValueError(f"invalid Google Drive source: {source!r}")
    return file_id, inner_path


def _filename_from_google_source(source: str, filename: str | None) -> str:
    if filename:
        return filename
    _, inner_path = _parse_google_drive_source(source)
    if inner_path:
        return Path(inner_path).name
    raise ValueError("filename is required when Google Drive source has no path component")


def _parse_kaggle_dataset_source(source: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme != "kaggle" or parsed.netloc != "datasets":
        raise ValueError(f"invalid Kaggle dataset source: {source!r}")
    parts = parsed.path.lstrip("/").split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Kaggle dataset source must be kaggle://datasets/<owner>/<dataset>/<file>: {source!r}")
    owner, dataset, file_path = parts
    return f"{owner}/{dataset}", urllib.parse.unquote(file_path)


def _kaggle_credentials_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser()
    return (config_dir / "kaggle.json").is_file() or (config_dir / "credentials.json").is_file()


def _require_kaggle_credentials() -> None:
    if _kaggle_credentials_available():
        return
    raise RuntimeError(
        "Kaggle downloads require API credentials. Set KAGGLE_USERNAME and KAGGLE_KEY, "
        "or place kaggle.json/credentials.json under ~/.kaggle/ or $KAGGLE_CONFIG_DIR."
    )


def fetch_http_file(
    url: str,
    filename: str,
    *,
    cache_prefix: str,
    env_var: str | None = None,
    sha256: str | None = None,
    description: str | None = None,
    timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    retries: int = 2,
    progress: bool | None = None,
) -> Path:
    """Return a cached upstream file, downloading it from its original URL when absent."""
    if retries < 0:
        raise ValueError("retries must be >= 0")

    label = description or filename
    override = os.environ.get(env_var) if env_var else None
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found at ${env_var}: {path}")
        verify_sha256(path, sha256, description=label)
        return path

    destination = upstream_cache_root() / cache_prefix / filename
    if destination.is_file():
        verify_sha256(destination, sha256, description=label)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock(destination):
        if destination.is_file():
            verify_sha256(destination, sha256, description=label)
            return destination
        temporary = _temporary_path(destination.parent)
        try:
            for attempt in range(retries + 1):
                try:
                    headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
                    partial_size = temporary.stat().st_size if temporary.exists() else 0
                    if partial_size:
                        headers["Range"] = f"bytes={partial_size}-"
                    request = urllib.request.Request(url, headers=headers)
                    urlopen_kwargs = {} if timeout is None else {"timeout": timeout}
                    with urllib.request.urlopen(request, **urlopen_kwargs) as response:
                        _copy_response(
                            response,
                            temporary,
                            label=label,
                            progress=progress,
                            append=partial_size > 0 and getattr(response, "status", None) == 206,
                        )
                    break
                except urllib.error.HTTPError as error:
                    if error.code == 416 and temporary.exists():
                        temporary.unlink()
                        if attempt < retries:
                            continue
                    if error.code not in {408, 429} and error.code < 500:
                        raise RuntimeError(f"{label} download failed from {url}: HTTP {error.code}") from error
                    if attempt >= retries:
                        raise RuntimeError(f"{label} download failed from {url}: HTTP {error.code}") from error
                    time.sleep(min(2**attempt, 10))
                except (urllib.error.URLError, OSError) as error:
                    if attempt >= retries:
                        raise RuntimeError(f"{label} download failed from {url}: {error}") from error
                    time.sleep(min(2**attempt, 10))
                except RuntimeError:
                    if attempt >= retries:
                        raise
                    time.sleep(min(2**attempt, 10))
            verify_sha256(temporary, sha256, description=label)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return destination


def _download_google_drive_file(
    file_id: str,
    destination: Path,
    *,
    label: str,
    progress: bool | None,
) -> None:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Google Drive downloads require gdown; install upstream requirements.txt") from error

    result = gdown.download(id=file_id, output=str(destination), quiet=not _progress_enabled(progress))
    if result is None or not destination.is_file():
        raise RuntimeError(f"{label} Google Drive file download failed for id {file_id}")


def _google_drive_folder_kwargs(download_folder, **kwargs):
    parameters = inspect.signature(download_folder).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def _download_google_drive_folder_file(
    folder_id: str,
    inner_path: str,
    destination: Path,
    *,
    label: str,
    progress: bool | None,
) -> None:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Google Drive downloads require gdown; install upstream requirements.txt") from error

    try:
        files = gdown.download_folder(
            **_google_drive_folder_kwargs(
                gdown.download_folder,
                id=folder_id,
                output=str(destination.parent),
                quiet=not _progress_enabled(progress),
                remaining_ok=True,
                skip_download=True,
            )
        )
    except Exception:
        files = None
    for item in files or ():
        item_path = Path(getattr(item, "path", "") or getattr(item, "local_path", ""))
        if item_path.as_posix() != inner_path and item_path.name != Path(inner_path).name:
            continue
        file_id = getattr(item, "id", None)
        if not file_id:
            continue
        _download_google_drive_file(file_id, destination, label=label, progress=progress)
        return

    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".gdrive-folder-") as tmp_dir:
        output_dir = Path(tmp_dir)
        result = gdown.download_folder(
            **_google_drive_folder_kwargs(
                gdown.download_folder,
                id=folder_id,
                output=str(output_dir),
                quiet=not _progress_enabled(progress),
                remaining_ok=True,
            )
        )
        if result is None:
            raise RuntimeError(f"{label} Google Drive folder download failed for id {folder_id}")
        candidates = [output_dir / inner_path]
        candidates.extend(path for path in output_dir.rglob(Path(inner_path).name) if path.is_file())
        for candidate in candidates:
            if candidate.is_file():
                shutil.copyfile(candidate, destination)
                return
    raise RuntimeError(f"{label} Google Drive folder {folder_id} did not contain {inner_path!r}")


def fetch_google_drive_file(
    source: str,
    filename: str | None = None,
    *,
    cache_prefix: str,
    env_var: str | None = None,
    sha256: str | None = None,
    description: str | None = None,
    progress: bool | None = None,
) -> Path:
    """Return a cached Google Drive artifact, downloading it with gdown when absent.

    `source` may be either a raw Google Drive file/folder id or
    `google_drive://<id>/<path>`. The helper first tries `<id>` as a file id.
    If that fails and a path is present, it falls back to downloading the
    public folder and extracting the named path.
    """
    label = description or filename or source
    override = os.environ.get(env_var) if env_var else None
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found at ${env_var}: {path}")
        verify_sha256(path, sha256, description=label)
        return path

    file_id, inner_path = _parse_google_drive_source(source)
    destination = upstream_cache_root() / cache_prefix / _filename_from_google_source(source, filename)
    if destination.is_file():
        verify_sha256(destination, sha256, description=label)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock(destination):
        if destination.is_file():
            verify_sha256(destination, sha256, description=label)
            return destination
        temporary = _temporary_path(destination.parent)
        try:
            try:
                _download_google_drive_file(file_id, temporary, label=label, progress=progress)
            except Exception:
                temporary.unlink(missing_ok=True)
                if not inner_path:
                    raise
                temporary = _temporary_path(destination.parent)
                _download_google_drive_folder_file(
                    file_id,
                    inner_path,
                    temporary,
                    label=label,
                    progress=progress,
                )
            verify_sha256(temporary, sha256, description=label)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return destination


def _copy_kaggle_download(download_dir: Path, file_path: str, destination: Path) -> None:
    candidates = [download_dir / file_path, download_dir / Path(file_path).name]
    candidates.extend(path for path in download_dir.rglob(Path(file_path).name) if path.is_file())
    for candidate in candidates:
        if candidate.is_file():
            shutil.copyfile(candidate, destination)
            return

    for archive_path in download_dir.glob("*.zip"):
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            match = next((name for name in names if name == file_path), None)
            if match is None:
                match = next(
                    (name for name in names if Path(name).name == Path(file_path).name),
                    None,
                )
            if match is None:
                continue
            with archive.open(match) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            return

    raise RuntimeError(f"Kaggle download did not contain {file_path!r}")


def _download_kaggle_dataset_file(
    dataset: str,
    file_path: str,
    destination: Path,
    *,
    label: str,
    progress: bool | None,
) -> None:
    _require_kaggle_credentials()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise RuntimeError("Kaggle downloads require kaggle; install upstream requirements.txt") from error

    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".kaggle-") as tmp_dir:
        download_dir = Path(tmp_dir)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(
            dataset,
            file_name=file_path,
            path=str(download_dir),
            force=True,
            quiet=not _progress_enabled(progress),
        )
        _copy_kaggle_download(download_dir, file_path, destination)


def fetch_kaggle_file(
    source: str,
    filename: str | None = None,
    *,
    cache_prefix: str,
    env_var: str | None = None,
    sha256: str | None = None,
    description: str | None = None,
    progress: bool | None = None,
    retries: int = 2,
) -> Path:
    """Return a cached Kaggle dataset file, downloading it with the Kaggle API when absent."""
    if retries < 0:
        raise ValueError("retries must be >= 0")

    dataset, file_path = _parse_kaggle_dataset_source(source)
    label = description or filename or file_path
    override = os.environ.get(env_var) if env_var else None
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found at ${env_var}: {path}")
        verify_sha256(path, sha256, description=label)
        return path

    destination = upstream_cache_root() / cache_prefix / (filename or Path(file_path).name)
    if destination.is_file():
        verify_sha256(destination, sha256, description=label)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock(destination):
        if destination.is_file():
            verify_sha256(destination, sha256, description=label)
            return destination
        temporary = _temporary_path(destination.parent)
        try:
            for attempt in range(retries + 1):
                try:
                    _download_kaggle_dataset_file(dataset, file_path, temporary, label=label, progress=progress)
                    break
                except Exception:
                    temporary.unlink(missing_ok=True)
                    if attempt >= retries:
                        raise
                    time.sleep(min(2**attempt, 10))
            verify_sha256(temporary, sha256, description=label)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return destination
