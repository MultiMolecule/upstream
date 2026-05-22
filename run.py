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

"""Run upstream fixture generators through one audit-friendly entry point."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.fixture import (
    ARTIFACT_FILENAMES,
    FixtureError,
    sha256_of_file,
    validate_fixture,
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUT = ROOT / "out"
DOCKER_PLATFORM = "linux/amd64"
DOCKER_CORPUS_CACHE = "/work/upstream/.cache/multimolecule-corpus"
DOCKER_UPSTREAM_CACHE = "/work/upstream/.cache/multimolecule-upstream"
UPSTREAM_ENV_PREFIX = "MULTIMOLECULE_UPSTREAM_"


@dataclass(frozen=True)
class FixtureCase:
    model: str
    case: str
    script: Path
    arguments: tuple[str, ...] = ()
    docker_image: str | None = None
    dockerfile: Path | None = None
    docker_context: Path | None = None
    origin: str = "generate.py"
    kaggle_artifacts: tuple[str, ...] = ()

    @property
    def output_dir(self) -> Path:
        return OUT / self.model / self.case

    @property
    def has_docker_config(self) -> bool:
        return bool(self.docker_image or self.dockerfile)

    def command(self, python: str) -> list[str]:
        return [python, str(self.script), *self.arguments]


@dataclass
class ContractIssue:
    path: Path
    line: int
    message: str
    excerpt: str


WORKSPACE_DEFAULT_RE = re.compile(
    r"WORKSPACE_ROOT[\s()]*/\s*[\"'](?P<root>parity|pretrained)[\"']",
    re.DOTALL,
)
DOWNSTREAM_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+multimolecule(?:\.|\s)|import\s+multimolecule(?:\.|\s|$))",
    re.MULTILINE,
)
SOURCE_YAML_LOCAL_URI_RE = re.compile(r"local://(?P<root>parity|pretrained)(?:/|$)")
SOURCE_YAML_RELATIVE_PATH_RE = re.compile(r":\s*[\"']?(?P<root>parity|pretrained)/")
KAGGLE_ARTIFACT_SCHEMES = ("kaggle://",)
MODEL_ENV_LITERAL_RE = re.compile(
    r"[\"'](?P<env>[A-Z][A-Z0-9_]*(?:CKPT|SRC|CHECKPOINT|SOURCE|ROOT|DIR|SNAPSHOT|ARCHIVE|MODEL_JSON)[A-Z0-9_]*)[\"']"
)


def split_csv(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read source.yaml generator metadata") from error

    payload = yaml.safe_load(path.read_text())
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a YAML mapping")
    return payload


def expand_argument_template(template: str, *, model: str, case: str, checkpoint: dict[str, Any]) -> str:
    values = {"model": model, "case": case, **checkpoint}
    return template.format_map(values)


def iter_checkpoint_cases(source_yaml: Path, checkpoints: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(checkpoints, dict):
        items = []
        for case, checkpoint in sorted(checkpoints.items()):
            if checkpoint is None:
                checkpoint = {}
            if not isinstance(checkpoint, dict):
                raise RuntimeError(f"{source_yaml}: checkpoint {case} must be a mapping")
            items.append((str(case), checkpoint))
        return items
    raise RuntimeError(f"{source_yaml}: checkpoints must be a mapping")


def iter_source_checkpoint_cases(source_yaml: Path, source: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    checkpoints = source.get("checkpoints")
    if checkpoints is not None:
        return iter_checkpoint_cases(source_yaml, checkpoints)
    return []


def artifact_sources(payload: Any, schemes: tuple[str, ...]) -> tuple[str, ...]:
    """Return sources in `payload` that use one of `schemes`."""
    sources: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if isinstance(value, str) and value.startswith(schemes):
            sources.append(value)

    visit(payload)
    return tuple(dict.fromkeys(sources))


def kaggle_artifact_sources(payload: Any) -> tuple[str, ...]:
    """Return checkpoint sources that require Kaggle API credentials."""
    return artifact_sources(payload, KAGGLE_ARTIFACT_SCHEMES)


def discover_source_yaml_cases() -> list[FixtureCase]:
    cases = []
    for source_yaml in sorted(MODELS.glob("*/source.yaml")):
        source = load_yaml(source_yaml)
        generator = source.get("generator")
        if generator is None:
            continue
        if not isinstance(generator, dict):
            raise RuntimeError(f"{source_yaml}: generator must be a mapping")
        checkpoint_cases = iter_source_checkpoint_cases(source_yaml, source)
        if not checkpoint_cases:
            raise RuntimeError(f"{source_yaml}: generator requires checkpoints or checkpoint metadata")

        model = str(source.get("model") or source_yaml.parent.name)
        docker_config = docker_config_from_source(source, source_yaml)
        script_template = generator.get("script")
        if not script_template:
            raise RuntimeError(f"{source_yaml}: generator.script is required")

        argument_templates = generator.get("arguments")
        if argument_templates is None:
            argument_templates = [generator.get("case_argument", "{case}")]
        elif isinstance(argument_templates, str):
            argument_templates = [argument_templates]
        elif not isinstance(argument_templates, list):
            raise RuntimeError(f"{source_yaml}: generator.arguments must be a string or list")

        for case, checkpoint in checkpoint_cases:
            script_name = expand_argument_template(
                str(script_template), model=model, case=str(case), checkpoint=checkpoint
            )
            script = source_yaml.parent / script_name
            if not script.is_file():
                raise RuntimeError(f"{source_yaml}: generator script not found: {script}")
            arguments = tuple(
                expand_argument_template(str(template), model=model, case=str(case), checkpoint=checkpoint)
                for template in argument_templates
            )
            cases.append(
                FixtureCase(
                    model=model,
                    case=str(case),
                    script=script,
                    arguments=arguments,
                    docker_image=docker_config["image"],
                    dockerfile=docker_config["dockerfile"],
                    docker_context=docker_config["context"],
                    origin=f"{source_yaml.relative_to(ROOT)}:generator",
                    kaggle_artifacts=kaggle_artifact_sources(checkpoint),
                )
            )
    return cases


def docker_context_for(source_yaml: Path, dockerfile: Path) -> Path:
    if dockerfile.parent == source_yaml.parent.resolve():
        return source_yaml.parent.resolve()
    return dockerfile.parent


def docker_config_from_source(source: dict[str, Any], source_yaml: Path | None = None) -> dict[str, Any]:
    docker = source.get("docker")
    if not isinstance(docker, dict):
        return {"image": None, "dockerfile": None, "context": None}
    image = docker.get("image")
    dockerfile_value = docker.get("dockerfile")
    dockerfile = None
    context = None
    if dockerfile_value:
        dockerfile = Path(str(dockerfile_value))
        if source_yaml is not None:
            dockerfile = (source_yaml.parent / dockerfile).resolve()
            context = docker_context_for(source_yaml, dockerfile)
    return {
        "image": str(image) if image else None,
        "dockerfile": dockerfile,
        "context": context or (dockerfile.parent if dockerfile is not None else None),
    }


def discover_cases() -> list[FixtureCase]:
    cases_by_id = {}
    for case in discover_source_yaml_cases():
        key = (case.model, case.case)
        existing = cases_by_id.get(key)
        if existing is not None:
            if existing.script == case.script and existing.arguments == case.arguments:
                continue
            print(
                f"WARNING: duplicate fixture case {case.model}/{case.case}; "
                f"using {existing.origin}, ignoring {case.origin}",
                file=sys.stderr,
            )
            continue
        cases_by_id[key] = case
    return [cases_by_id[key] for key in sorted(cases_by_id)]


def select_cases(
    cases: list[FixtureCase],
    *,
    families: set[str],
    checkpoints: set[str],
    all_cases: bool,
    default_all: bool = False,
) -> list[FixtureCase]:
    if default_all and not all_cases and not families and not checkpoints:
        return cases
    if not all_cases and not families and not checkpoints:
        raise SystemExit("Select at least one checkpoint with --checkpoint, --family, or --all. ")
    selected = []
    for case in cases:
        if all_cases or case.model in families or case.case in checkpoints:
            selected.append(case)
    if not selected:
        selectors = sorted(families | checkpoints)
        raise SystemExit(f"No fixture generators matched: {', '.join(selectors)}")
    return selected


def check_output(case: FixtureCase) -> None:
    missing = [name for name in ARTIFACT_FILENAMES if not (case.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"{case.output_dir}: generator did not write required artifacts: {', '.join(missing)}")
    validate_fixture(case.output_dir, expected_model=case.model, expected_case=case.case)


def promote_output(case: FixtureCase, golden_root: Path, *, replace: bool) -> Path:
    check_output(case)
    destination = golden_root / "models" / case.model / case.case
    if destination.exists():
        if not replace:
            raise RuntimeError(f"{destination} already exists; pass --replace to overwrite it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILENAMES:
        shutil.copy2(case.output_dir / name, destination / name)
        source_digest = sha256_of_file(case.output_dir / name)
        destination_digest = sha256_of_file(destination / name)
        if destination_digest != source_digest:
            raise RuntimeError(
                f"{destination / name}: copied artifact sha256 mismatch: "
                f"expected {source_digest}, got {destination_digest}"
            )
    try:
        validate_fixture(destination, expected_model=case.model, expected_case=case.case)
    except FixtureError as error:
        raise RuntimeError(f"{destination}: invalid promoted fixture artifacts: {error}") from error
    return destination


def cleanup_output(case: FixtureCase, *, dry_run: bool = False) -> None:
    if not case.output_dir.exists():
        return
    if dry_run:
        print(f"    would remove: {case.output_dir}")
        return
    shutil.rmtree(case.output_dir)
    for parent in (OUT / case.model, OUT):
        with contextlib.suppress(OSError):
            parent.rmdir()


def docker_command(case: FixtureCase, image: str, python: str) -> list[str]:
    script = case.script.relative_to(ROOT)
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        DOCKER_PLATFORM,
        "-e",
        f"MULTIMOLECULE_CORPUS_CACHE={DOCKER_CORPUS_CACHE}",
        "-e",
        f"MULTIMOLECULE_UPSTREAM_CACHE={DOCKER_UPSTREAM_CACHE}",
        "-v",
        f"{ROOT.parent}:/work",
        "-w",
        "/work/upstream",
    ]
    if case.kaggle_artifacts:
        command.extend(kaggle_docker_args())
    command.extend(model_env_docker_args())
    command.extend([image, python, str(script), *case.arguments])
    return command


def docker_workspace_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(ROOT.parent)
    except ValueError as error:
        raise RuntimeError(f"{path}: cannot be mapped into Docker workspace mounted at {ROOT.parent}") from error
    return f"/work/{relative.as_posix()}"


def docker_chown_command(image: str, case: FixtureCase) -> list[str]:
    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        return []
    container_out_root = docker_workspace_path(OUT)
    container_model_dir = docker_workspace_path(case.output_dir.parent)
    container_case_dir = docker_workspace_path(case.output_dir)
    script = (
        f"for path in \"$1\" \"$2\"; do "
        f"if [ -e \"$path\" ]; then "
        f"chown {os.getuid()}:{os.getgid()} \"$path\" && chmod u+rwx,go+rx \"$path\"; "
        f"fi; "
        f"done; "
        f"path=\"$3\"; "
        f"if [ -e \"$path\" ]; then "
        f"chown -R {os.getuid()}:{os.getgid()} \"$path\" && chmod -R u+rwX,go+rX \"$path\"; "
        f"fi"
    )
    return [
        "docker",
        "run",
        "--rm",
        "--platform",
        DOCKER_PLATFORM,
        "--user",
        "0:0",
        "-v",
        f"{ROOT.parent}:/work",
        "-w",
        "/work/upstream",
        image,
        "sh",
        "-c",
        script,
        "sh",
        container_out_root,
        container_model_dir,
        container_case_dir,
    ]


def normalize_docker_output_permissions(case: FixtureCase, image: str, *, strict: bool = True) -> None:
    command = docker_chown_command(image, case)
    if not command:
        return
    print("    normalize output ownership", flush=True)
    print_command(command)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except Exception as error:
        if strict:
            raise
        print(f"WARNING: unable to normalize Docker output ownership after generator failure: {error}", file=sys.stderr)


def model_env_docker_args() -> list[str]:
    args = []
    for name in sorted(os.environ):
        if not name.startswith(UPSTREAM_ENV_PREFIX):
            continue
        if name == "MULTIMOLECULE_UPSTREAM_CACHE":
            continue
        args.extend(["-e", name])
    return args


def kaggle_docker_args() -> list[str]:
    args = []
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        args.extend(["-e", "KAGGLE_USERNAME", "-e", "KAGGLE_KEY"])
        return args

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser()
    if (config_dir / "kaggle.json").is_file() or (config_dir / "credentials.json").is_file():
        args.extend(["-e", "KAGGLE_CONFIG_DIR=/root/.kaggle", "-v", f"{config_dir}:/root/.kaggle:ro"])
    return args


def docker_image_exists(image: str) -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def ensure_docker_available() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required for this command but was not found on PATH")


def select_docker_image(case: FixtureCase, override: str | None) -> str:
    if override:
        return override
    if case.docker_image:
        return case.docker_image
    raise RuntimeError(f"{case.model}/{case.case}: no Docker image declared in source.yaml")


def wants_docker(case: FixtureCase, runtime: str | None, docker_image: str | None) -> bool:
    if docker_image:
        return True
    if runtime == "docker":
        return True
    if runtime == "local":
        return False
    return case.has_docker_config


def print_command(command: list[str | Path]) -> None:
    print("    " + shlex.join(str(part) for part in command), flush=True)


def run_case(
    case: FixtureCase,
    python: str,
    *,
    dry_run: bool,
    runtime: str | None,
    docker_python: str,
    docker_image: str | None,
) -> None:
    image = None
    if wants_docker(case, runtime, docker_image):
        image = select_docker_image(case, docker_image)
        command = docker_command(case, image, docker_python)
    else:
        command = case.command(python)
    print(f"\n>>> {case.model}/{case.case}")
    print_command(command)
    if dry_run:
        return
    if image is not None:
        ensure_docker_available()
    run_succeeded = False
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        run_succeeded = True
    finally:
        if image is not None:
            normalize_docker_output_permissions(case, image, strict=run_succeeded)
    check_output(case)


def validate_golden(golden_root: Path, python: str, *, dry_run: bool = False) -> None:
    validate_py = golden_root / "validate.py"
    if not validate_py.is_file():
        raise RuntimeError(f"{golden_root}: missing validate.py")
    command = [python, str(validate_py), "--upstream-root", str(ROOT)]
    print("\n>>> validate golden", flush=True)
    print_command(command)
    if dry_run:
        return
    subprocess.run(command, cwd=golden_root, check=True)


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(ROOT.parent))
        except ValueError:
            return str(path)


def runtime_label(case: FixtureCase) -> str:
    return "docker" if case.has_docker_config else "python"


def kaggle_auth_status(case: FixtureCase) -> str:
    if not case.kaggle_artifacts:
        return "not required"

    package_state = "installed" if importlib.util.find_spec("kaggle") is not None else "missing package"
    has_env_credentials = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser()
    legacy_config = config_dir / "kaggle.json"
    oauth_config = config_dir / "credentials.json"
    config_credentials = legacy_config if legacy_config.is_file() else oauth_config if oauth_config.is_file() else None
    if has_env_credentials:
        auth_state = "env credentials"
    elif config_credentials is not None:
        auth_state = f"config {config_credentials}"
    else:
        auth_state = "missing credentials"
    suffix = "s" if len(case.kaggle_artifacts) != 1 else ""
    return f"{package_state}; {auth_state}; {len(case.kaggle_artifacts)} source{suffix}"


def output_status(case: FixtureCase) -> str:
    pieces = []
    for artifact in ARTIFACT_FILENAMES:
        pieces.append(f"{artifact}={'yes' if (case.output_dir / artifact).is_file() else 'no'}")
    if all((case.output_dir / artifact).is_file() for artifact in ARTIFACT_FILENAMES):
        try:
            validate_fixture(case.output_dir, expected_model=case.model, expected_case=case.case)
            pieces.append("valid=yes")
        except FixtureError as error:
            pieces.append(f"valid=no ({str(error).splitlines()[0]})")
    return ", ".join(pieces)


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def line_excerpt(text: str, line: int) -> str:
    lines = text.splitlines()
    if line < 1 or line > len(lines):
        return ""
    return lines[line - 1].strip()


def contract_issue_label(issue: ContractIssue) -> str:
    return f"{relative_path(issue.path)}:{issue.line}: " f"{issue.message} ({issue.excerpt})"


def scan_script_contract(script: Path) -> list[ContractIssue]:
    if not script.is_file():
        return []
    text = script.read_text()
    issues = []
    for match in WORKSPACE_DEFAULT_RE.finditer(text):
        root = match.group("root")
        line = line_number(text, match.start("root"))
        issues.append(
            ContractIssue(
                path=script,
                line=line,
                message=(
                    f"default path uses WORKSPACE_ROOT/{root}; " "use an explicit env override or upstream cache helper"
                ),
                excerpt=line_excerpt(text, line),
            )
        )
    for match in DOWNSTREAM_IMPORT_RE.finditer(text):
        line = line_number(text, match.start())
        issues.append(
            ContractIssue(
                path=script,
                line=line,
                message="generator imports downstream multimolecule package; use upstream-local helpers instead",
                excerpt=line_excerpt(text, line),
            )
        )
    for match in MODEL_ENV_LITERAL_RE.finditer(text):
        env_name = match.group("env")
        if env_name.startswith(UPSTREAM_ENV_PREFIX) or env_name.startswith("KAGGLE_"):
            continue
        line = line_number(text, match.start("env"))
        issues.append(
            ContractIssue(
                path=script,
                line=line,
                message=f"model override env {env_name!r} must use {UPSTREAM_ENV_PREFIX}<CASE>_<KIND>",
                excerpt=line_excerpt(text, line),
            )
        )
    return issues


def case_name_contract_issues(source_yaml: Path, model: str, checkpoints: dict[str, Any]) -> list[ContractIssue]:
    issues = []
    lines = source_yaml.read_text().splitlines()
    for case in checkpoints:
        allowed_prefix = case == model or case.startswith(f"{model}-") or case.startswith(f"{model}.")
        dot_is_length = "." not in case or case.rsplit(".", 1)[1].isdigit()
        if allowed_prefix and dot_is_length:
            continue
        line = next((index for index, content in enumerate(lines, start=1) if content.strip() == f"{case}:"), 1)
        issues.append(
            ContractIssue(
                path=source_yaml,
                line=line,
                message="fixture case must be named <family>, <family>-<variant>, or <family>[-<variant>].<length>",
                excerpt=line_excerpt(source_yaml.read_text(), line),
            )
        )
    return issues


def checkpoint_case_for_line(
    content: str,
    *,
    in_checkpoints: bool,
    current_case: str | None,
) -> tuple[bool, str | None]:
    stripped = content.strip()
    indent = len(content) - len(content.lstrip())
    if indent == 0:
        return stripped == "checkpoints:", None
    if in_checkpoints and indent == 2 and stripped.endswith(":"):
        return True, stripped[:-1].strip("'\"")
    return in_checkpoints, current_case


def scan_source_yaml_contract(source_yaml: Path, *, case: str | None = None) -> list[ContractIssue]:
    if not source_yaml.is_file():
        return []
    text = source_yaml.read_text()
    issues = []
    source = load_yaml(source_yaml)
    docker = source.get("docker")
    docker_config = docker if isinstance(docker, dict) else {}
    has_unbuildable_image = all(
        (
            docker_config.get("image"),
            not docker_config.get("dockerfile"),
        )
    )
    if has_unbuildable_image:
        line = next(
            (index for index, content in enumerate(text.splitlines(), start=1) if content.strip().startswith("image:")),
            1,
        )
        issues.append(
            ContractIssue(
                path=source_yaml,
                line=line,
                message=("docker.image is declared without docker.dockerfile; add a buildable image path"),
                excerpt=line_excerpt(text, line),
            )
        )
    model = str(source.get("model") or source_yaml.parent.name)
    checkpoints = source.get("checkpoints")
    if not isinstance(checkpoints, dict):
        line = next(
            (
                index
                for index, content in enumerate(text.splitlines(), start=1)
                if content.strip().startswith("checkpoint")
            ),
            1,
        )
        issues.append(
            ContractIssue(
                path=source_yaml,
                line=line,
                message="source.yaml must declare checkpoints as a mapping keyed by fixture case",
                excerpt=line_excerpt(text, line),
            )
        )
    else:
        issues.extend(case_name_contract_issues(source_yaml, model, checkpoints))
    generator = source.get("generator")
    if isinstance(generator, dict):
        script = generator.get("script")
        if isinstance(script, str) and script == f"{model}/generate.py":
            line = next(
                (index for index, content in enumerate(text.splitlines(), start=1) if "script:" in content),
                1,
            )
            issues.append(
                ContractIssue(
                    path=source_yaml,
                    line=line,
                    message="default generator should live at generate.py; nested scripts are for per-case generators",
                    excerpt=line_excerpt(text, line),
                )
            )
    in_checkpoints = False
    current_case: str | None = None
    for line, content in enumerate(text.splitlines(), start=1):
        in_checkpoints, current_case = checkpoint_case_for_line(
            content,
            in_checkpoints=in_checkpoints,
            current_case=current_case,
        )
        for pattern in (SOURCE_YAML_LOCAL_URI_RE, SOURCE_YAML_RELATIVE_PATH_RE):
            match = pattern.search(content)
            if match is None:
                continue
            if case is not None and current_case is not None and current_case != case:
                break
            root = match.group("root")
            issues.append(
                ContractIssue(
                    path=source_yaml,
                    line=line,
                    message=(
                        f"source.yaml references sibling {root}/ artifacts; "
                        "resolve through the upstream cache contract"
                    ),
                    excerpt=content.strip(),
                )
            )
            break
    return issues


def contract_issues_for_case(case: FixtureCase) -> list[ContractIssue]:
    issues = scan_script_contract(case.script)
    issues.extend(scan_source_yaml_contract(MODELS / case.model / "source.yaml", case=case.case))
    return issues


def selected_from_args(
    cases: list[FixtureCase], args: argparse.Namespace, *, default_all: bool = False
) -> list[FixtureCase]:
    return select_cases(
        cases,
        families=split_csv(getattr(args, "families", None)),
        checkpoints=split_csv(getattr(args, "checkpoints", None)),
        all_cases=getattr(args, "all", False),
        default_all=default_all,
    )


def list_cases(cases: list[FixtureCase], args: argparse.Namespace) -> None:
    selected = selected_from_args(cases, args, default_all=True)
    for case in selected:
        print(f"{case.case}\t{case.model}\t{relative_path(case.script)}\t" f"{case.origin}\t{runtime_label(case)}")


def docker_image_status(image: str | None) -> str:
    if not image:
        return "not declared"
    if shutil.which("docker") is None:
        return f"{image} (unknown: docker not found)"
    return f"{image} ({'present' if docker_image_exists(image) else 'missing'})"


def dockerfile_status(case: FixtureCase) -> str:
    if case.dockerfile is None:
        return "not declared"
    state = "exists" if case.dockerfile.is_file() else "missing"
    return f"{relative_path(case.dockerfile)} ({state})"


def run_doctor(cases: list[FixtureCase], args: argparse.Namespace) -> int:
    selected = selected_from_args(cases, args)
    contract_issues = {case: contract_issues_for_case(case) for case in selected}
    print(f"selected checkpoints: {len(selected)}")
    for case in selected:
        print(f"\n{case.case} ({case.model})")
        print(f"  script: {relative_path(case.script)}")
        print(f"  origin: {case.origin}")
        print(f"  runtime: {runtime_label(case)}")
        print(f"  kaggle.auth: {kaggle_auth_status(case)}")
        print(f"  docker.image: {docker_image_status(case.docker_image)}")
        print(f"  docker.dockerfile: {dockerfile_status(case)}")
        if case.docker_context is not None:
            print(f"  docker.context: {relative_path(case.docker_context)}")
        print(f"  output: {relative_path(case.output_dir)} ({'exists' if case.output_dir.exists() else 'missing'})")
        print(f"  artifacts: {output_status(case)}")
        issues = contract_issues[case]
        if issues:
            print(f"  contract: WARN ({len(issues)} issue{'s' if len(issues) != 1 else ''})")
            for issue in issues:
                print(f"    - {contract_issue_label(issue)}")
        else:
            print("  contract: ok")
    warning_cases = [(case, issues) for case, issues in contract_issues.items() if issues]
    if warning_cases:
        print("\ncontract warnings:")
        for case, issues in warning_cases:
            print(f"  {case.model}/{case.case}: {len(issues)} issue{'s' if len(issues) != 1 else ''}")
        if args.strict_contract:
            return 1
    return 0


def unique_docker_cases(cases: list[FixtureCase]) -> list[FixtureCase]:
    unique = []
    seen = set()
    for case in cases:
        if not case.has_docker_config:
            continue
        key = (
            case.docker_image,
            str(case.dockerfile) if case.dockerfile is not None else None,
            str(case.docker_context) if case.docker_context is not None else None,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(case)
    return unique


def prepare_docker(cases: list[FixtureCase], *, dry_run: bool) -> None:
    docker_cases = unique_docker_cases(cases)
    if not docker_cases:
        print("No Docker runtime declared for selected checkpoints.")
        return

    for case in docker_cases:
        label = case.docker_image or "<unnamed>"
        print(f"\n>>> prepare docker image: {label}")

        if case.dockerfile is not None:
            if not case.dockerfile.is_file():
                raise RuntimeError(f"{case.model}: declared dockerfile does not exist: {case.dockerfile}")
            if not case.docker_image:
                raise RuntimeError(f"{case.model}: docker.dockerfile requires docker.image")
            if docker_image_exists(case.docker_image):
                print(f"    ready: {case.docker_image}")
                continue
            context = case.docker_context or case.dockerfile.parent
            command = [
                "docker",
                "build",
                "--platform",
                DOCKER_PLATFORM,
                "-t",
                case.docker_image,
                "-f",
                str(case.dockerfile),
                str(context),
            ]
            print_command(command)
            if dry_run:
                continue
            ensure_docker_available()
            subprocess.run(command, cwd=ROOT, check=True)
            continue

        if case.docker_image:
            command = [
                "docker",
                "pull",
                "--platform",
                DOCKER_PLATFORM,
                case.docker_image,
            ]
            print_command(command)
            if dry_run:
                continue
            ensure_docker_available()
            subprocess.run(command, cwd=ROOT, check=True)
            continue

        raise RuntimeError(f"{case.model}: local Docker image is missing and no dockerfile or public image is declared")


def promote_case(case: FixtureCase, golden_root: Path, *, replace: bool, dry_run: bool) -> Path:
    destination = golden_root / "models" / case.model / case.case
    print(f"\n>>> promote {case.model}/{case.case}")
    print(f"    {relative_path(case.output_dir)} -> {relative_path(destination)}")
    if dry_run:
        return destination
    promoted = promote_output(case, golden_root, replace=replace)
    print(f"    promoted: {promoted}")
    return promoted


def run_generate(cases: list[FixtureCase], args: argparse.Namespace) -> int:
    selected = selected_from_args(cases, args)
    for case in selected:
        run_case(
            case,
            args.python,
            dry_run=args.dry_run,
            runtime=args.runtime,
            docker_python=args.docker_python,
            docker_image=args.docker_image,
        )
        if args.clean_output:
            cleanup_output(case, dry_run=args.dry_run)
    return 0


def run_promote(cases: list[FixtureCase], args: argparse.Namespace) -> int:
    selected = selected_from_args(cases, args)
    golden_root = args.golden_root.expanduser().resolve()
    for case in selected:
        promote_case(case, golden_root, replace=args.replace, dry_run=args.dry_run)
    return 0


def run_prepare(cases: list[FixtureCase], args: argparse.Namespace) -> int:
    selected = selected_from_args(cases, args)
    prepare_docker(selected, dry_run=args.dry_run)
    return 0


def run_validate(args: argparse.Namespace) -> int:
    validate_golden(args.golden_root.expanduser().resolve(), args.python, dry_run=args.dry_run)
    return 0


def run_golden(cases: list[FixtureCase], args: argparse.Namespace) -> int:
    selected = selected_from_args(cases, args)
    golden_root = args.golden_root.expanduser().resolve()
    prepare_docker(selected, dry_run=args.dry_run)
    for case in selected:
        docker_image = args.docker_image
        run_case(
            case,
            args.python,
            dry_run=args.dry_run,
            runtime=args.runtime,
            docker_python=args.docker_python,
            docker_image=docker_image,
        )
        promote_case(case, golden_root, replace=args.replace, dry_run=args.dry_run)
        if args.clean_output:
            cleanup_output(case, dry_run=args.dry_run)
    validate_golden(golden_root, args.python, dry_run=args.dry_run)
    return 0


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        help="Checkpoint id to select. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--family",
        dest="families",
        action="append",
        help="Model family to select. Can be repeated or comma-separated.",
    )
    parser.add_argument("--all", action="store_true", help="Select all discovered fixture generators.")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument(
        "--docker",
        dest="runtime",
        action="store_const",
        const="docker",
        help="Force Docker runtime.",
    )
    runtime.add_argument(
        "--no-docker",
        dest="runtime",
        action="store_const",
        const="local",
        help="Force local Python.",
    )
    parser.set_defaults(runtime=None)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run local generators.",
    )
    parser.add_argument("--docker-image", help="Override the Docker image declared by source.yaml.")
    parser.add_argument(
        "--docker-python",
        default="python",
        help="Python executable inside the Docker image.",
    )


def add_golden_root_argument(parser: argparse.ArgumentParser, *, default: Path | None) -> None:
    kwargs: dict[str, Any] = {"type": Path, "help": "Golden checkout root."}
    if default is not None:
        kwargs["default"] = default
    parser.add_argument("--golden-root", **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Shortcut for the list subcommand.")

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    list_parser = subparsers.add_parser("list", help="List checkpoint/family/script/origin/runtime.")
    add_selection_arguments(list_parser)

    doctor_parser = subparsers.add_parser("doctor", help="Read-only preflight for selected checkpoints.")
    add_selection_arguments(doctor_parser)
    doctor_parser.add_argument(
        "--strict-contract",
        action="store_true",
        help="Return non-zero on source/artifact contract warnings.",
    )

    prepare_parser = subparsers.add_parser("prepare", help="Prepare declared Docker images only.")
    add_selection_arguments(prepare_parser)
    prepare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Docker build/pull commands without running them.",
    )

    generate_parser = subparsers.add_parser("generate", help="Run selected upstream generators.")
    add_selection_arguments(generate_parser)
    add_runtime_arguments(generate_parser)
    generate_parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove upstream/out artifacts after generation.",
    )
    generate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generator commands without running them.",
    )

    promote_parser = subparsers.add_parser("promote", help="Copy existing upstream/out artifacts into golden.")
    add_selection_arguments(promote_parser)
    add_golden_root_argument(promote_parser, default=ROOT.parent / "golden")
    promote_parser.add_argument("--replace", action="store_true", help="Replace an existing golden case.")
    promote_parser.add_argument("--dry-run", action="store_true", help="Print copy operations without writing.")

    validate_parser = subparsers.add_parser("validate", help="Run golden/validate.py --upstream-root upstream.")
    add_golden_root_argument(validate_parser, default=ROOT.parent / "golden")
    validate_parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run golden/validate.py.",
    )
    validate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the validation command without running it.",
    )

    golden_parser = subparsers.add_parser("golden", help="Run prepare, generate, promote, and validate.")
    add_selection_arguments(golden_parser)
    add_runtime_arguments(golden_parser)
    add_golden_root_argument(golden_parser, default=ROOT.parent / "golden")
    golden_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing golden cases during promotion.",
    )
    golden_parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove upstream/out artifacts after promotion.",
    )
    golden_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full workflow without writes or heavy execution.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cases = discover_cases()
    command = "list" if args.list else args.command
    if command == "list":
        list_cases(cases, args)
        return 0
    if command == "doctor":
        return run_doctor(cases, args)
    if command == "prepare":
        return run_prepare(cases, args)
    if command == "generate":
        return run_generate(cases, args)
    if command == "promote":
        return run_promote(cases, args)
    if command == "validate":
        return run_validate(args)
    if command == "golden":
        return run_golden(cases, args)
    raise SystemExit("specify a subcommand: list|doctor|prepare|generate|promote|validate|golden")


if __name__ == "__main__":
    raise SystemExit(main())
