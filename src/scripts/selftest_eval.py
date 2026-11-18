#!/usr/bin/env python3
"""Run an agent-authored selftest suite and emit a structured envelope.

Contract for selftest/run_selftest.py:
  - run from the operator workspace;
  - write a JSON object to $SELFTEST_JSON_OUT (and optionally stdout);
  - JSON shape:
      {"status": "passed|failed", "metrics": {...}, "cases": [...],
       "coverage": {...}, "failures": [...]}

This runner is deliberately separate from kernel_eval.py: selftests are
diagnostic and are never authoritative; official smoke/full remain the judge.
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
from pathlib import Path

from selftest_guard import guard_state, lock_path, suite_hash, write_lock

SCHEMA = "kernelbench.feedback.v1"


def sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_json_object(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_cases(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, dict):
            result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="operator workspace")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--feedback", default="")
    args = parser.parse_args()

    task = Path(args.task).expanduser().resolve()
    suite = task / "selftest"
    script = suite / "run_selftest.py"
    out_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else task / "evaluations"
    run_id = f"selftest-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    stdout_path = out / "stdout.log"
    stderr_path = out / "stderr.log"
    result_path = out / "result.json"

    envelope: dict = {
        "schema": SCHEMA,
        "kind": "selftest",
        "candidate_id": sha256(task / "solution.json")[:16],
        "status": "blocked",
        "command": [],
        "exit_code": None,
        "duration_s": 0,
        "summary": "",
        "metrics": {},
        "cases": [],
        "failures": [],
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "result": str(result_path),
        },
        "attribution": "selftest",
        "failure_category": "selftest",
        "test_hash": "",
        "test_changed": False,
        "guard": {},
    }

    def finish() -> int:
        feedback = Path(args.feedback).expanduser().resolve() if args.feedback else None
        if feedback:
            feedback.parent.mkdir(parents=True, exist_ok=True)
            with feedback.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(envelope, ensure_ascii=False) + "\n")
        (out / "feedback.json").write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(envelope, ensure_ascii=False))
        return {"passed": 0, "blocked": 2}.get(envelope["status"], 1)

    if not task.is_dir():
        envelope["summary"] = f"task dir missing: {task}"
        return finish()
    if not script.is_file():
        envelope["summary"] = (
            "selftest/run_selftest.py missing; write it only if tests are useful, "
            "or skip selftest"
        )
        return finish()

    state = guard_state(task)
    envelope["guard"] = state
    envelope["test_hash"] = state["current_hash"]
    envelope["test_changed"] = state["test_changed"]
    if state["state"] == "blocked":
        envelope["summary"] = (
            "selftest suite changed after freeze; submit a change request before running: "
            "python src/scripts/selftest_guard.py --task <workspace> request --reason ..."
        )
        return finish()
    if state["state"] == "unfrozen":
        write_lock(task, state["current_hash"], source="first-selftest")
        state = guard_state(task)
        envelope["guard"] = state
        envelope["test_changed"] = False

    started = time.time()
    env = os.environ.copy()
    env["SELFTEST_JSON_OUT"] = str(result_path)
    env["SELFTEST_TASK"] = str(task)
    env["SELFTEST_OUTPUT_DIR"] = str(out)
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, str(script)]
    envelope["command"] = command

    rc = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(
                command, cwd=str(task), env=env, stdout=stdout, stderr=stderr,
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=max(10, args.timeout))
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=10)
                rc = 124
    except Exception as exc:
        envelope.update({"status": "failure", "summary": f"selftest launch failed: {exc}"})
        envelope["duration_s"] = round(time.time() - started, 1)
        return finish()

    envelope["exit_code"] = rc
    envelope["duration_s"] = round(time.time() - started, 1)
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    result = {}
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {}
    if not result:
        result = last_json_object(stdout_text) or {}

    cases = normalize_cases(result.get("cases"))
    failures = normalize_cases(result.get("failures"))
    if not failures:
        failures = [
            case for case in cases
            if str(case.get("status", "")).upper() not in {"PASSED", "PASS", "OK"}
        ]
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    passed = metrics.get("passed")
    total = metrics.get("total")
    if passed is None:
        passed = sum(1 for case in cases if str(case.get("status", "")).upper() in {"PASSED", "PASS", "OK"})
    if total is None:
        total = len(cases)

    status = str(result.get("status") or "").lower()
    if rc == 124:
        status = "timeout"
    elif not status:
        status = "passed" if rc == 0 and not failures else "failed"
    if status not in {"passed", "failed", "timeout"}:
        status = "failed"
    if status == "passed" and failures:
        status = "failed"

    envelope.update({
        "status": status,
        "summary": str(result.get("summary") or f"selftest {passed}/{total} passed"),
        "metrics": {
            **metrics,
            "total": total,
            "passed": passed,
            "failed": len(failures),
            "coverage": result.get("coverage") or metrics.get("coverage") or {},
        },
        "cases": cases,
        "failures": failures,
        "attribution": str(result.get("attribution") or "selftest"),
        "failure_category": str(result.get("failure_category") or "selftest"),
        "test_hash": suite_hash(task),
        "test_changed": False,
    })
    if rc == 124:
        envelope["summary"] = f"selftest timeout after {args.timeout}s"
    elif rc not in (0, None) and status == "passed":
        envelope["status"] = "failed"
        envelope["summary"] = f"selftest exited {rc} despite reporting passed"
    return finish()


if __name__ == "__main__":
    sys.exit(main())
