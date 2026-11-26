#!/usr/bin/env python3
"""Snapshot a locally verified candidate. No build, no evaluation."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    task = Path(args.task).resolve()
    solution = task / "solution.json"
    if not solution.is_file():
        print(json.dumps({"error": "solution.json missing"}))
        return 1
    label = args.label or time.strftime("%Y%m%d-%H%M%S")
    target = task / "checkpoints" / label
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(solution, target / "solution.json")
    for name in ("main.py", "op_host", "op_kernel", "notes.md", "experiments.md", "research.md"):
        source = task / name
        if source.is_dir():
            shutil.copytree(source, target / name)
        elif source.is_file():
            shutil.copy2(source, target / name)
    (target / "meta.json").write_text(json.dumps({
        "label": label,
        "task": task.name,
        "created": time.strftime("%F %T"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
