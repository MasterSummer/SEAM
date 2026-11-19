#!/usr/bin/env python3
"""Fast local iteration for an already-staged Ascend C workspace.

This reuses the official evaluator's build_tmp tree and generated eval.py, but:
  - rebuilds incrementally with cmake --build;
  - runs only selected workloads;
  - disables reference profiling / reduces iterations;
  - emits a small JSON summary instead of a full trace envelope.

It is diagnostic only. Official kernel_eval.py smoke/full remain authoritative.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TASK_FILES = ("main.py", "solution.json", "definition.json", "workload.jsonl")
PYTHON = os.environ.get("FAST_PYTHON", "/home/yiding/npu-kernelbench/.venv/bin/python")


def find_staged(task: Path) -> Path:
    candidates = sorted(
        (p.parent for p in (task / "evaluations").glob("**/work/*/eval.py")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(
            (p.parent.parent for p in (task / "build_tmp").glob("*/build/CMakeCache.txt")),
            key=lambda p: p.stat().st_mtime,
        )
    if not candidates:
        raise SystemExit(
            "no staged build found; run one official smoke first so the evaluator stages the build"
        )
    return candidates[-1]


def copy_source(task: Path, staged: Path) -> None:
    for name in TASK_FILES:
        src = task / name
        if src.is_file() and name != "workload.jsonl":
            shutil.copy2(src, staged / name)
    for sub in ("op_host", "op_kernel"):
        src = task / sub
        if src.is_dir():
            dst = staged / "sources" / sub
            dst.mkdir(parents=True, exist_ok=True)
            for path in src.iterdir():
                if path.is_file():
                    shutil.copy2(path, dst / path.name)
    main = task / "main.py"
    if main.is_file():
        (staged / "sources").mkdir(exist_ok=True)
        shutil.copy2(main, staged / "sources" / "main.py")


def sync_eval_dir(task: Path, staged: Path, selected: list[dict], device: int) -> Path:
    fast = task / "fast_eval"
    if fast.exists():
        shutil.rmtree(fast)
    fast.mkdir(parents=True)
    for name in ("eval.py", "solution.json", "definition.json", "kernel.so"):
        src = staged / name
        if src.is_file():
            shutil.copy2(src, fast / name)
    if (staged / "sources").is_dir():
        shutil.copytree(staged / "sources", fast / "sources")
    with (fast / "workload.jsonl").open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    config = {
        "benchmark_reference": False,
        "warmup_runs": 0,
        "iterations": 1,
        "npu_id": 0,
        "npu_ids": [0],
        "enable_anti_hack": ["ast_scan", "kernel_signature"],
    }
    (fast / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    return fast


def build(task: Path, staged: Path, jobs: int) -> dict:
    copy_source(task, staged)
    started = time.time()
    proc = subprocess.run(
        ["cmake", "--build", "build", "-j", str(jobs)],
        cwd=staged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log = staged / "fast-build.log"
    log.write_text(proc.stdout, encoding="utf-8")
    lib = staged / "build" / "libkernel.so"
    if lib.is_file():
        shutil.copy2(lib, staged / "kernel.so")
    return {
        "status": "success" if proc.returncode == 0 else "failed",
        "exit_code": proc.returncode,
        "duration_s": round(time.time() - started, 1),
        "log": str(log),
        "kernel": str(staged / "kernel.so") if lib.is_file() else None,
    }


def run(task: Path, staged: Path, limit: int, workloads: str, device: int, timeout: int) -> dict:
    all_workloads = [
        json.loads(line) for line in (task / "workload.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if workloads:
        wanted = {item.strip() for item in workloads.split(",") if item.strip()}
        selected = [item for item in all_workloads if item.get("uuid") in wanted]
    else:
        selected = all_workloads[:max(1, limit)]
    if not selected:
        raise SystemExit("no workloads selected")
    fast = sync_eval_dir(task, staged, selected, device)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    env["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    proc = subprocess.run(
        [PYTHON, "eval.py"],
        cwd=fast,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    (fast / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (fast / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    traces = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    statuses: dict[str, int] = {}
    max_abs = 0.0
    max_rel = 0.0
    failures = []
    for trace in traces:
        evaluation = trace.get("evaluation") or {}
        status = str(evaluation.get("status") or "UNKNOWN")
        statuses[status] = statuses.get(status, 0) + 1
        correctness = evaluation.get("correctness") or {}
        try:
            max_abs = max(max_abs, float(correctness.get("max_absolute_error") or 0))
            max_rel = max(max_rel, float(correctness.get("max_relative_error") or 0))
        except (TypeError, ValueError):
            pass
        if status != "PASSED":
            failures.append({
                "workload": (trace.get("workload") or {}).get("uuid"),
                "status": status,
                "max_abs": correctness.get("max_absolute_error"),
                "max_rel": correctness.get("max_relative_error"),
            })
    passed = statuses.get("PASSED", 0)
    return {
        "status": "passed" if traces and passed == len(traces) else "failed",
        "duration_s": round(time.time() - started, 1),
        "tested": len(traces),
        "passed": passed,
        "statuses": statuses,
        "max_abs": max_abs if max_abs else None,
        "max_rel": max_rel if max_rel else None,
        "failures": failures,
        "stdout": str(fast / "stdout.log"),
        "stderr": str(fast / "stderr.log"),
        "exit_code": proc.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--task", required=True)
        cmd.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
        cmd.add_argument("--limit", type=int, default=2)
        cmd.add_argument("--workloads", default="")
        cmd.add_argument("--device", type=int, default=0)
        cmd.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    task = Path(args.task).expanduser().resolve()
    staged = find_staged(task)
    if args.command == "build":
        print(json.dumps(build(task, staged, args.jobs), ensure_ascii=False))
        return 0
    result = run(task, staged, args.limit, args.workloads, args.device, args.timeout)
    (task / "fast_eval" / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
