#!/usr/bin/env python3
"""Compile-check the generated host/kernel skeleton (no algorithm, no device)."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

from npu_kernelbench.builder.ascendc_builder import AscendCBuilder
from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.solution import Solution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    task = Path(args.task).resolve()
    card = task / "task_card"
    if not card.is_dir():
        raise SystemExit(f"task_card missing: {card}")
    definition = Definition.model_validate(json.loads((task / "definition.json").read_text()))
    name = definition.name
    sources = []
    sources.append({"path": "main.py", "content": "def run(*args, **kwargs):\n    raise NotImplementedError\n"})
    for sub in ("op_host", "op_kernel"):
        directory = card / "question" / sub
        for path in sorted(directory.glob("*.cpp")):
            sources.append({"path": str(path.relative_to(card / "question")), "content": path.read_text()})
    solution = Solution.model_validate({
        "name": f"{name.lower()}_skeleton",
        "definition": name,
        "author": "skeleton-check",
        "spec": {
            "languages": ["ascendc"],
            "target_hardware": ["ascend910b"],
            "entry_point": "main.py::run",
            "destination_passing_style": False,
            "build_method": "cmake",
        },
        "sources": sources,
    })
    work = card / "skeleton_build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for entry in solution.sources:
        dst = work / "sources" / entry.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(entry.content)
    try:
        AscendCBuilder().build(definition, solution, work_dir=str(work), compile_timeout=600)
        result = {"status": "success", "work": str(work), "log": str(work / "cmake_compile.log")}
    except Exception as exc:  # noqa: BLE001
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "work": str(work),
                  "log": str(work / "cmake_compile.log")}
    (card / "skeleton_compile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
