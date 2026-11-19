#!/usr/bin/env python3
"""Build a KernelBench solution directly with AscendCBuilder (no official eval)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from npu_kernelbench.builder.ascendc_builder import AscendCBuilder
from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.solution import Solution


def main() -> int:
    task = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2]).resolve()
    solution = Solution.model_validate(json.loads((task / "solution.json").read_text()))
    definition = Definition.model_validate(json.loads((task / "definition.json").read_text()))
    work.mkdir(parents=True, exist_ok=True)
    sources_dir = work / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    for entry in solution.sources:
        dst = sources_dir / entry.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(entry.content)
    AscendCBuilder().build(definition, solution, work_dir=str(work), compile_timeout=600)
    print(json.dumps({
        "status": "success",
        "work": str(work),
        "kernel": str(work / "kernel.so"),
        "sources": str(work / "sources"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
