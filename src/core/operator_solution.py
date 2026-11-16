"""Explicit source manifests and framework-owned KernelBench solution packaging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.atomic_file import atomic_write_bytes


def _encode(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _read_spec(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("candidate specification must not be a symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("candidate specification must be an object")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("candidate specification requires a non-empty sources list")
    return value


def _source_paths(spec: dict[str, Any], root: Path, reserved: set[Path]) -> list[Path]:
    paths: list[Path] = []
    for entry in spec["sources"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("each source requires a relative path")
        relative = Path(entry["path"])
        if (
            not entry["path"]
            or relative == Path(".")
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError(f"invalid source path: {entry['path']}")
        target = root / relative
        if target.resolve() in reserved or not target.resolve().is_relative_to(root):
            raise ValueError(f"source path escapes or names a reserved artifact: {relative}")
        if any(
            parent.is_symlink()
            for parent in (target, *target.parents)
            if parent.is_relative_to(root) and parent != root
        ):
            raise ValueError(f"source path contains a symlink: {relative}")
        if target in paths:
            raise ValueError(f"duplicate source path: {relative}")
        paths.append(target)
    return paths


def initialize_operator_solution(
    solution: str | Path, task_dir: str | Path | None = None
) -> dict[str, Any]:
    """Import a legacy embedded candidate once without overwriting existing edits."""
    requested_path = Path(solution).expanduser()
    solution_path = requested_path.parent.resolve() / requested_path.name
    spec_path = solution_path.with_name("solution_spec.json")
    result: dict[str, Any] = {
        "operation": "operator_artifact_initialize",
        "initialized": False,
        "errors": [],
    }
    if spec_path.exists() or not solution_path.is_file():
        return result
    try:
        if solution_path.is_symlink():
            raise ValueError("solution.json must not be a symlink")
        spec = _read_spec(solution_path)
        task_root = (
            Path(task_dir).expanduser().resolve()
            if task_dir is not None else solution_path.parent
        )
        reserved = {
            solution_path, spec_path,
            task_root / "definition.json", task_root / "workload.jsonl",
        }
        paths = _source_paths(spec, solution_path.parent, reserved)
        for entry, path in zip(spec["sources"], paths):
            if not path.exists() and not isinstance(entry.get("content"), str):
                raise ValueError(f"legacy source has no content: {entry['path']}")
        for entry, path in zip(spec["sources"], paths):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(path, entry["content"].encode("utf-8"))
            entry.pop("content", None)
        atomic_write_bytes(spec_path, _encode(spec))
        result["initialized"] = True
    except (OSError, ValueError, UnicodeError) as exc:
        result["errors"] = [str(exc)]
    return result


def _published_source_path(path: Path, task_dir: Path, root: Path) -> str:
    """Path a packed solution should declare for a staged source file.

    KernelBench stages ``solution.sources[*].path`` under ``<work>/sources`` and
    loads the entry point from there. Candidates are authored with paths relative
    to the task root (the prompt prefers ``sources/``), so strip that prefix when
    the file really lives under ``<task_dir>/sources``; otherwise keep the path
    relative to the specification directory, which is the legacy layout.
    """
    resolved = path.resolve()
    sources_root = (task_dir.expanduser().resolve() / "sources")
    try:
        return resolved.relative_to(sources_root).as_posix()
    except ValueError:
        return resolved.relative_to(root).as_posix()


def solution_payload_errors(payload: Any) -> list[str]:
    """Return the KernelBench ``Solution`` contract violations in a payload."""
    if not isinstance(payload, dict):
        return ["solution.json must contain an object"]
    errors: list[str] = []
    for field in ("name", "definition", "author"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"solution.json requires non-empty {field}")
    build = payload.get("spec")
    if not isinstance(build, dict):
        errors.append("solution.json requires a spec object")
        build = {}
    for field in ("languages", "target_hardware"):
        value = build.get(field)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item for item in value
        ):
            errors.append(f"spec.{field} must be a non-empty string list")
    entry = build.get("entry_point")
    if not isinstance(entry, str) or entry.count("::") != 1 or not all(entry.split("::")):
        errors.append("spec.entry_point must have the form '<path>::<symbol>'")
    build_method = build.get("build_method")
    if build_method is not None and not isinstance(build_method, str):
        errors.append("spec.build_method must be a string")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("solution.json requires a non-empty sources list")
        sources = []
    paths: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            errors.append("sources must contain objects")
            continue
        path = source.get("path")
        content = source.get("content")
        if not isinstance(path, str) or not path:
            errors.append("each source requires a non-empty path")
            continue
        if not isinstance(content, str) or not content:
            errors.append(f"source content must be a non-empty string: {path}")
        if path in paths:
            errors.append(f"duplicate source path: {path}")
        paths.add(path)
    if isinstance(entry, str) and "::" in entry:
        entry_file = entry.split("::", 1)[0]
        if entry_file not in paths:
            errors.append(f"entry source is not in sources: {entry_file}")
    return errors


def _build_solution_payload(
    spec: dict[str, Any],
    task_dir: Path,
    solution_path: Path,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a KernelBench Solution, accepting legacy flat phase-3 specs too."""
    definition = spec.get("definition") or spec.get("operator") or task_dir.name
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError("cannot determine solution definition")
    source_languages = [
        entry["language"]
        for entry in spec["sources"]
        if isinstance(entry, dict)
        and isinstance(entry.get("language"), str)
        and entry["language"]
    ]
    build = spec.get("spec")
    if isinstance(build, dict):
        build = dict(build)
        entry = build.get("entry_point") or spec.get("entry_point")
    else:
        entry = spec.get("entry_point")
        if not entry and spec.get("name") and sources:
            entry = f"{sources[0]['path']}::run"
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("solution_spec.json requires spec.entry_point or entry_point")
        solution_type = spec.get("solution_type") or "ascendc"
        build = {
            "languages": spec.get("languages") or source_languages or [solution_type],
            "target_hardware": spec.get("target_hardware") or ["ascend910b"],
            "entry_point": entry,
            "destination_passing_style": spec.get("destination_passing_style", False),
            "build_method": spec.get("build_method") or "cmake",
        }
    if not isinstance(entry, str) or entry.count("::") != 1:
        raise ValueError("solution entry_point must have the form '<path>::<symbol>'")
    entry_file, symbol = entry.split("::", 1)
    published_paths = {source["path"] for source in sources}
    by_original = {
        entry_spec["path"]: source["path"]
        for entry_spec, source in zip(spec["sources"], sources)
        if isinstance(entry_spec, dict) and isinstance(entry_spec.get("path"), str)
    }
    if entry_file in published_paths:
        published_entry = entry_file
    elif entry_file in by_original:
        published_entry = by_original[entry_file]
    else:
        entry_path = solution_path.parent / entry_file
        if not entry_path.is_file():
            raise ValueError(f"entry source does not exist: {entry_file}")
        published_entry = _published_source_path(entry_path, task_dir, solution_path.parent)
    if published_entry not in published_paths:
        raise ValueError(f"entry source is not in sources: {published_entry}")
    build["entry_point"] = f"{published_entry}::{symbol}"
    build.setdefault("languages", source_languages or [spec.get("solution_type") or "ascendc"])
    build.setdefault("target_hardware", spec.get("target_hardware") or ["ascend910b"])
    build.setdefault("destination_passing_style", False)
    if "build_method" not in build:
        build["build_method"] = spec.get("build_method") or "cmake"
    payload: dict[str, Any] = {
        "name": spec.get("name") or f"{definition.lower()}_{spec.get('solution_type') or 'ascendc'}",
        "definition": definition,
        "author": spec.get("author") or "seam-agent",
        "spec": build,
        "sources": sources,
    }
    description = spec.get("description")
    if isinstance(description, str) and description:
        payload["description"] = description
    errors = solution_payload_errors(payload)
    if errors:
        raise ValueError("invalid KernelBench solution: " + "; ".join(errors))
    return payload


def pack_operator_solution(solution: str | Path, task_dir: str | Path) -> dict[str, Any]:
    """Pack declared source files atomically, preserving the old solution on failure."""
    requested_path = Path(solution).expanduser()
    solution_path = requested_path.parent.resolve() / requested_path.name
    spec_path = solution_path.with_name("solution_spec.json")
    result: dict[str, Any] = {
        "operation": "operator_artifact_pack",
        "solution_path": str(solution_path),
        "spec_path": str(spec_path),
        "packed": False,
        "errors": [],
    }
    try:
        if solution_path.is_symlink():
            raise ValueError("solution.json must not be a symlink")
        spec = _read_spec(spec_path)
        if any(
            "content" in entry
            for entry in spec["sources"] if isinstance(entry, dict)
        ):
            raise ValueError("solution_spec.json must declare paths without embedded content")
        reserved = {solution_path, spec_path}
        reserved.update(
            Path(task_dir).expanduser().resolve() / name
            for name in ("definition.json", "workload.jsonl")
        )
        paths = _source_paths(spec, solution_path.parent, reserved)
        before = [path.read_bytes() for path in paths]
        sources = [
            {
                **entry,
                "path": _published_source_path(path, Path(task_dir), solution_path.parent),
                "content": content.decode("utf-8"),
            }
            for entry, path, content in zip(spec["sources"], paths, before)
        ]
        packed = _build_solution_payload(
            spec, Path(task_dir).expanduser().resolve(), solution_path, sources
        )
        if (
            before != [path.read_bytes() for path in paths]
            or spec != _read_spec(spec_path)
        ):
            raise ValueError("candidate inputs changed during packaging")
        payload = _encode(packed)
        atomic_write_bytes(solution_path, payload)
        result.update(
            {
                "packed": True,
                "source_count": len(paths),
                "solution_sha256": hashlib.sha256(payload).hexdigest(),
                "source_sha256": {
                    entry["path"]: hashlib.sha256(content).hexdigest()
                    for entry, content in zip(packed["sources"], before)
                },
            }
        )
    except (OSError, ValueError, UnicodeError) as exc:
        result["errors"] = [str(exc)]
    return result
