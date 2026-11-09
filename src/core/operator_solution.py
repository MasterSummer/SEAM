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
        packed = dict(spec)
        packed["sources"] = [
            {
                **entry,
                "path": _published_source_path(path, Path(task_dir), solution_path.parent),
                "content": content.decode("utf-8"),
            }
            for entry, path, content in zip(spec["sources"], paths, before)
        ]
        spec_block = packed.get("spec")
        if isinstance(spec_block, dict):
            entry_point = spec_block.get("entry_point")
            if isinstance(entry_point, str) and entry_point:
                entry_file, _, symbol = entry_point.partition("::")
                entry_path = solution_path.parent / entry_file
                if entry_path.is_file():
                    published = _published_source_path(
                        entry_path, Path(task_dir), solution_path.parent
                    )
                    packed["spec"] = {
                        **spec_block,
                        "entry_point": f"{published}::{symbol}" if symbol else published,
                    }
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
