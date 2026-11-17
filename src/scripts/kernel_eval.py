#!/usr/bin/env python3
"""KernelBench smoke/full evaluator for the thin agent loop.

The agent calls this via bash.  Every call returns one FeedbackEnvelope JSON
object and appends it to feedback.jsonl when requested.  Smoke is diagnostic;
only full is authoritative.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

SCHEMA = "kernelbench.feedback.v1"


def sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_expected(task_dir: Path) -> int:
    path = task_dir / "workload.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def summarize_traces(traces: Path) -> tuple[Counter, list[dict], dict]:
    statuses: Counter = Counter()
    failures: list[dict] = []
    max_abs = 0.0
    max_rel = 0.0
    if not traces.is_file():
        return statuses, failures, {"max_abs": None, "max_rel": None}
    for line in traces.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        evaluation = item.get("evaluation") or {}
        status = str(evaluation.get("status") or "UNKNOWN")
        statuses[status] += 1
        correctness = evaluation.get("correctness") or {}
        try:
            max_abs = max(max_abs, float(correctness.get("max_absolute_error") or 0))
            max_rel = max(max_rel, float(correctness.get("max_relative_error") or 0))
        except (TypeError, ValueError):
            pass
        if status != "PASSED":
            workload = item.get("workload") or {}
            failures.append({
                "workload": workload.get("uuid") or item.get("uuid") or "unknown",
                "phase": (
                    "runtime" if status in {"RUNTIME_ERROR", "TIMEOUT", "LAUNCH_ERROR"}
                    else "correctness" if status == "INCORRECT_NUMERICAL"
                    else "unknown"
                ),
                "status": status,
                "message": str(evaluation.get("error") or evaluation.get("message") or "")[:500],
                "max_absolute_error": correctness.get("max_absolute_error"),
                "max_relative_error": correctness.get("max_relative_error"),
                "shape": (workload.get("inputs") or {}),
                "details_path": str(traces),
            })
    return statuses, failures[:8], {
        "max_abs": max_abs if max_abs else None,
        "max_rel": max_rel if max_rel else None,
    }


def attribution(stdout: str, stderr: str, traces: Path) -> str:
    evidence = (stdout + "\n" + stderr).lower()
    if any(marker in evidence for marker in (
        "fault kernel_name=", "aicore exception", "device fault", "kernel execute failed",
    )):
        return "candidate"
    if any(marker in evidence for marker in (
        "reference run() failed", "build_baseline", "gen_inputs", "aclnninplaceuniform",
    )) and ("error" in evidence or "runtimeerror" in evidence or "failed" in evidence):
        return "reference"
    if any(marker in evidence for marker in (
        "get device count of npu failed", "invalid device id", "dcmi module initialize failed",
        "drvret", "get platform info failed", "device_count", "no npu",
    )):
        return "environment"
    if "no such file" in evidence or "permission denied" in evidence:
        return "environment"
    if not traces.is_file():
        return "harness"
    return "candidate"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "full"])
    parser.add_argument("--task", required=True, help="operator workspace/task dir")
    parser.add_argument("--solution", default="")
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--kernelbench-root", default="/home/yiding/npu-kernelbench")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--feedback", default="")
    args = parser.parse_args()
    args.timeout = args.timeout or (1800 if args.mode == "full" else 900)

    task = Path(args.task).expanduser().resolve()
    solution = Path(args.solution).expanduser().resolve() if args.solution else task / "solution.json"
    root = Path(args.kernelbench_root).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else task / "evaluations"
    eval_id = f"{args.mode}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = out_root / eval_id
    out.mkdir(parents=True, exist_ok=True)
    traces = out / "traces.jsonl"
    stdout_path = out / "stdout.log"
    stderr_path = out / "stderr.log"
    work = out / "work"
    profile = out / "profile"
    work.mkdir(exist_ok=True)
    profile.mkdir(exist_ok=True)

    started = time.time()
    envelope: dict = {
        "schema": SCHEMA,
        "kind": args.mode,
        "candidate_id": sha256(solution)[:16],
        "status": "failure",
        "command": [],
        "exit_code": None,
        "duration_s": 0,
        "summary": "",
        "metrics": {},
        "failures": [],
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "traces": str(traces),
            "work": str(work),
            "profile": str(profile),
            "solution": str(solution),
        },
        "attribution": "harness",
    }

    if not task.is_dir():
        envelope.update({"status": "blocked", "summary": f"task dir missing: {task}"})
        print(json.dumps(envelope, ensure_ascii=False))
        return 2
    if not solution.is_file():
        envelope.update({"status": "blocked", "summary": f"solution missing: {solution}"})
        print(json.dumps(envelope, ensure_ascii=False))
        return 2

    workload_timeout = max(10, int(args.timeout * 0.75))
    command = [
        "uv", "run", "npu-kernelbench", "eval", str(task),
        "--solution", str(solution),
        "--runner", "persistent",
        "--timeout", str(workload_timeout),
        "-o", str(traces),
        "--verbose",
        "--work-dir", str(work),
        "--profile-dir", str(profile),
        "--npu-ids", str(args.device),
    ]
    if args.mode == "smoke" and args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    envelope["command"] = command

    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    env["PYTHONUNBUFFERED"] = "1"
    uv_cache = task / ".uv-cache"
    uv_cache.mkdir(parents=True, exist_ok=True)
    env["UV_CACHE_DIR"] = str(uv_cache)
    env["UV_NO_PROGRESS"] = "1"
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)

    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(
                command, cwd=str(root), env=env, stdout=stdout, stderr=stderr,
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=args.timeout + 60)
                envelope["exit_code"] = rc
                envelope["status"] = "completed" if rc == 0 else "failure"
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=10)
                envelope["exit_code"] = 124
                envelope["status"] = "timeout"

    except Exception as exc:
        envelope["status"] = "failure"
        envelope["summary"] = f"evaluator launch failed: {exc}"

    envelope["duration_s"] = round(time.time() - started, 1)
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    statuses, failures, errors = summarize_traces(traces)
    full_expected = load_expected(task)
    expected = (
        min(args.limit, full_expected)
        if args.mode == "smoke" and args.limit > 0
        else full_expected
    )
    total = sum(statuses.values())
    passed = statuses.get("PASSED", 0)
    complete = expected > 0 and total == expected
    smoke_passed = args.mode == "smoke" and complete and passed == total
    full_passed = args.mode == "full" and complete and passed == total
    metrics = {
        "expected_workloads": expected,
        "full_expected_workloads": full_expected,
        "total_workloads": total,
        "passed_workloads": passed,
        "workload_pass_rate": round(passed / total, 4) if total else 0.0,
        "trace_complete": complete,
        "smoke_passed": smoke_passed,
        "operator_passed": full_passed,
        "statuses": dict(statuses),
        "max_abs": errors["max_abs"],
        "max_rel": errors["max_rel"],
    }
    attribution_value = attribution(stdout_text, stderr_text, traces)
    envelope.update({
        "metrics": metrics,
        "failures": failures,
        "attribution": attribution_value,
        "failure_category": {
            "reference": "invalid_reference",
            "environment": "environment",
            "harness": "harness",
            "candidate": "candidate",
        }.get(attribution_value, "candidate"),
    })
    if args.mode == "smoke":
        envelope["summary"] = f"smoke {passed}/{total} passed" + (" (complete)" if complete else " (partial)")
    else:
        envelope["summary"] = f"full {passed}/{total} passed" + (" (complete)" if complete else " (incomplete)")
    feedback_path = Path(args.feedback).expanduser().resolve() if args.feedback else None
    if feedback_path:
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with feedback_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    (out / "feedback.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(envelope, ensure_ascii=False))
    return 0 if envelope["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
