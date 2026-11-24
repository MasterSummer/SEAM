#!/usr/bin/env python3
"""Implementation review: source sync, debug leftovers, local case coverage."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def latest_feedback(path: Path) -> dict:
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    task = Path(args.task).resolve()
    critical, warnings = [], []
    solution_path = task / "solution.json"
    if not solution_path.is_file():
        critical.append("solution.json missing")
        solution = {}
    else:
        solution = json.loads(solution_path.read_text())

    # 1. solution sources vs workspace files
    for entry in solution.get("sources") or []:
        path = task / entry.get("path", "")
        content = entry.get("content", "")
        if not path.is_file():
            critical.append(f"source missing: {entry.get('path')}")
        elif path.read_text(errors="replace") != content:
            critical.append(f"source drifted from solution.json: {entry.get('path')}")

    # 2. debug/probe/TODO kernels
    kernel_dir = task / "op_kernel"
    if kernel_dir.is_dir():
        for path in kernel_dir.glob("*.cpp"):
            text = path.read_text(errors="replace")
            if "probe" in path.name.lower():
                critical.append(f"probe named as final kernel: {path.name}")
            if "TODO" in text:
                critical.append(f"TODO left in kernel: {path.name}")
            if "debug build" in text.lower():
                critical.append(f"debug build submitted as final kernel: {path.name}")
            elif "DEBUG" in text:
                warnings.append(f"debug marker in kernel: {path.name}")
        if not any("TODO" not in p.read_text(errors="replace") for p in kernel_dir.glob("*.cpp")):
            critical.append("op_kernel contains only TODO/empty sources")

    # 3. local smoke result and required case coverage
    local = {}
    result_path = task / "fast_eval" / "result.json"
    if result_path.is_file():
        local = json.loads(result_path.read_text())
        if local.get("status") != "passed":
            warnings.append("latest local smoke did not pass")
    else:
        warnings.append("no fast_eval/result.json")
    tested = set(local.get("tested_uuids") or [])
    required_path = task / "task_card" / "required_cases.csv"
    missing_required = []
    if required_path.is_file():
        with required_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                uuid = row.get("uuid")
                if uuid and uuid not in tested:
                    missing_required.append({"uuid": uuid, "reason": row.get("reason")})
        if missing_required:
            warnings.append(f"required local cases not tested: {len(missing_required)}")
    else:
        warnings.append("task_card/required_cases.csv missing")

    status = "fail" if critical else ("warn" if warnings else "pass")
    report = {
        "status": status,
        "critical": critical,
        "warnings": warnings,
        "local_smoke": local.get("status"),
        "tested_uuids": sorted(tested),
        "missing_required": missing_required,
    }
    output = Path(args.output).resolve() if args.output else task / "review.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if status != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
