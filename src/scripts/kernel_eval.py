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
import re
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


def summarize_traces(traces: Path) -> tuple[Counter, list[dict], dict, dict[str, dict]]:
    statuses: Counter = Counter()
    failures: list[dict] = []
    tested: dict[str, dict] = {}
    max_abs = 0.0
    max_rel = 0.0
    if not traces.is_file():
        return statuses, failures, {"max_abs": None, "max_rel": None}, tested
    for line in traces.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        workload = item.get("workload") or {}
        uuid = str(workload.get("uuid") or item.get("uuid") or "unknown")
        tested[uuid] = workload
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
            failures.append({
                "workload": uuid,
                "phase": (
                    "runtime" if status in {"RUNTIME_ERROR", "TIMEOUT", "LAUNCH_ERROR"}
                    else "correctness" if status == "INCORRECT_NUMERICAL"
                    else "compile" if status == "COMPILE_ERROR"
                    else "unknown"
                ),
                "status": status,
                "message": str(evaluation.get("error") or evaluation.get("message") or "")[:1000],
                "max_absolute_error": correctness.get("max_absolute_error"),
                "max_relative_error": correctness.get("max_relative_error"),
                "has_nan": correctness.get("has_nan"),
                "has_inf": correctness.get("has_inf"),
                "shape": (workload.get("inputs") or {}),
                "details_path": str(traces),
            })
    return statuses, failures, {
        "max_abs": max_abs if max_abs else None,
        "max_rel": max_rel if max_rel else None,
    }, tested


def load_workloads(task: Path) -> list[dict]:
    path = task / "workload.jsonl"
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def coverage_report(expected: list[dict], tested: dict[str, dict]) -> dict:
    dtypes: set[str] = set()
    ranks: set[int] = set()
    has_tail = False
    has_empty = False
    has_non_last = False
    max_numel = 0
    size_bytes = {
        "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
        "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
    }
    for workload in tested.values():
        inputs = workload.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue
        for spec in inputs.values():
            if not isinstance(spec, dict):
                continue
            dtype = str(spec.get("dtype") or "")
            shape = spec.get("shape")
            if dtype:
                dtypes.add(dtype)
            if not isinstance(shape, list):
                continue
            ranks.add(len(shape))
            try:
                numel = 1
                for dim in shape:
                    numel *= int(dim)
                max_numel = max(max_numel, numel)
            except (TypeError, ValueError):
                pass
            if any(int(dim) == 0 for dim in shape if isinstance(dim, int)):
                has_empty = True
            if len(shape) >= 2 and any(int(dim) > 1 for dim in shape[:-1] if isinstance(dim, int)):
                has_non_last = True
            if shape:
                try:
                    last = int(shape[-1])
                    width = size_bytes.get(dtype, 4)
                    if last * width % 32 != 0:
                        has_tail = True
                except (TypeError, ValueError):
                    pass
    expected_uuids = [str(item.get("uuid") or "") for item in expected]
    tested_uuids = set(tested)
    missing = [uuid for uuid in expected_uuids if uuid and uuid not in tested_uuids]
    return {
        "tested_workloads": len(tested),
        "expected_workloads": len(expected),
        "missing_workloads": missing,
        "dtypes": sorted(dtypes),
        "ranks": sorted(ranks),
        "has_tail": has_tail,
        "has_empty": has_empty,
        "has_non_last_dim": has_non_last,
        "max_numel": max_numel,
    }


def build_report(statuses: Counter, stderr_text: str, traces: Path) -> dict:
    compile_errors = int(statuses.get("COMPILE_ERROR", 0))
    timeout = bool(re.search(r"(?:cmake )?build timed out", stderr_text, re.IGNORECASE))
    warnings = len(re.findall(r"\bwarning:", stderr_text, re.IGNORECASE))
    errors = len(re.findall(r"\berror:", stderr_text, re.IGNORECASE))
    if timeout:
        status = "timeout"
    elif compile_errors or (errors and not traces.is_file()):
        status = "failed"
    elif traces.is_file():
        status = "success"
    else:
        status = "unknown"
    return {
        "status": status,
        "compile_errors": compile_errors,
        "errors": errors,
        "warnings": warnings,
        "timeout": timeout,
    }


def best_score_path(task: Path) -> Path:
    return task / "best" / "score.json"


def load_best(task: Path) -> dict:
    path = best_score_path(task)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def failure_diff(best: dict, statuses: Counter, failures: list[dict], metrics: dict) -> dict:
    if not best:
        return {}
    current_failures = {item.get("workload") for item in failures if item.get("workload")}
    previous_failures = set(best.get("failure_uuids") or [])
    return {
        "best_candidate_id": best.get("candidate_id"),
        "best_kind": best.get("kind"),
        "best_pass_rate": best.get("workload_pass_rate"),
        "pass_rate_delta": round(
            float(metrics.get("workload_pass_rate") or 0) - float(best.get("workload_pass_rate") or 0), 4
        ),
        "passed_delta": int(metrics.get("passed_workloads") or 0) - int(best.get("passed_workloads") or 0),
        "new_failures": sorted(current_failures - previous_failures),
        "fixed": sorted(previous_failures - current_failures),
        "statuses": dict(statuses),
    }


def update_best(task: Path, solution: Path, envelope: dict) -> None:
    if envelope.get("kind") != "full":
        return
    metrics = envelope.get("metrics") or {}
    if not metrics.get("trace_complete"):
        return
    previous = load_best(task)
    rate = float(metrics.get("workload_pass_rate") or 0)
    passed = int(metrics.get("passed_workloads") or 0)
    if not previous and passed <= 0 and not metrics.get("operator_passed"):
        return
    previous_rate = float(previous.get("workload_pass_rate") or 0)
    previous_passed = int(previous.get("passed_workloads") or 0)
    if previous and not (rate > previous_rate or (rate == previous_rate and passed > previous_passed)):
        return
    best_dir = task / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    if solution.is_file():
        (best_dir / "solution.json").write_bytes(solution.read_bytes())
    failures = [str(item.get("workload")) for item in envelope.get("failures") or [] if item.get("workload")]
    data = {
        "candidate_id": envelope.get("candidate_id"),
        "kind": "full",
        "status": envelope.get("status"),
        "workload_pass_rate": rate,
        "passed_workloads": passed,
        "total_workloads": int(metrics.get("total_workloads") or 0),
        "failure_uuids": failures,
        "max_abs": metrics.get("max_abs"),
        "max_rel": metrics.get("max_rel"),
        "source_solution": str(solution),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    best_score_path(task).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
        "--compile-timeout", "600",
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
    statuses, failures, errors, tested = summarize_traces(traces)
    workloads = load_workloads(task)
    full_expected = len(workloads) if workloads else load_expected(task)
    expected = (
        min(args.limit, full_expected)
        if args.mode == "smoke" and args.limit > 0
        else full_expected
    )
    expected_workloads = workloads[:expected] if expected else []
    total = sum(statuses.values())
    passed = statuses.get("PASSED", 0)
    complete = expected > 0 and total == expected
    smoke_passed = args.mode == "smoke" and complete and passed == total
    full_passed = args.mode == "full" and complete and passed == total
    previous_best = load_best(task)
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
        "failure_count": len(failures),
        "failure_uuids": [item.get("workload") for item in failures],
        "max_abs": errors["max_abs"],
        "max_rel": errors["max_rel"],
        "coverage": coverage_report(expected_workloads, tested),
    }
    attribution_value = attribution(stdout_text, stderr_text, traces)
    envelope.update({
        "metrics": metrics,
        "failures": failures,
        "build": build_report(statuses, stderr_text, traces),
        "diff_from_best": failure_diff(previous_best, statuses, failures, metrics),
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
    update_best(task, solution, envelope)
    (out / "feedback.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(envelope, ensure_ascii=False))
    return 0 if envelope["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
