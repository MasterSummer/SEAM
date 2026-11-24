#!/usr/bin/env python3
"""Judge a timeout/aborted task: continue, stop, retry API, or retry on a clean card."""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path

API_MARKERS = ("transport", "stream idle", "request timed out", "rate_limit", "429")
ENV_MARKERS = ("device fault", "aicore exception", "rtstreamsynchronizewithtimeout", "507014", "507034", "suspect remote error")


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def last_jsonl(path: Path) -> dict:
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--probe", action="store_true", help="also probe the device")
    args = parser.parse_args()
    task = Path(args.task).resolve()
    run = read_json(task.parent / "run.json")
    final = last_jsonl(task / "final-feedback.jsonl")
    metrics = final.get("metrics", {})
    local = read_json(task / "fast_eval" / "result.json")
    log = (task.parent / "agent.log").read_text(errors="replace").lower() if (task.parent / "agent.log").is_file() else ""
    notes = task / "notes.md"
    solution = (task / "best" / "solution.json").is_file() or (task / "solution.json").is_file()
    agent_exit = run.get("agent_exit")
    statuses = metrics.get("statuses") or {}
    origins = metrics.get("failure_origins") or {}
    local_passed = int(local.get("passed") or 0)
    local_tested = int(local.get("tested") or 0)
    full_passed = int(metrics.get("passed_workloads") or 0)
    pass_rate = float(metrics.get("workload_pass_rate") or 0.0)
    evidence = {
        "agent_exit": agent_exit,
        "duration_s": run.get("agent_duration_s"),
        "solution": solution,
        "local_tested": local_tested,
        "local_passed": local_passed,
        "full_passed": full_passed,
        "pass_rate": pass_rate,
        "statuses": statuses,
        "origins": origins,
        "notes_bytes": notes.stat().st_size if notes.is_file() else 0,
    }

    api = any(marker in log for marker in API_MARKERS)
    env = final.get("attribution") == "environment" or any(marker in log for marker in ENV_MARKERS)
    progress = solution or local_tested > 0 or full_passed > 0 or evidence["notes_bytes"] > 200
    runtime_n = int(statuses.get("RUNTIME_ERROR") or 0)
    shape_n = int(statuses.get("INCORRECT_SHAPE") or 0)
    numerical_n = int(statuses.get("INCORRECT_NUMERICAL") or 0)

    if api:
        decision = "retry_api"
        reason = "agent ended on an API transport/idle/request timeout; resume same workspace"
    elif env:
        decision = "retry_clean_card"
        reason = "device/stream failure; move the same workspace to a clean card, do not rerun agent from scratch"
    elif agent_exit == 124 and progress:
        decision = "continue"
        reason = "timeout with measurable progress; continue in the same workspace"
    elif agent_exit == 124 and not progress:
        decision = "stop"
        reason = "timeout without solution or local progress; likely spinning"
    elif solution:
        decision = "continue"
        reason = "solution exists; finish verification or fix remaining cases"
    else:
        decision = "stop"
        reason = "no solution and no measurable progress"

    if runtime_n >= max(5, len(statuses) and sum(statuses.values()) // 5):
        goal = "stabilize runtime: workspace/blockDim/UB and shape dispatch before full"
    elif shape_n:
        goal = "fix output shape/layout first, then rerun tiny cases"
    elif numerical_n:
        goal = "fix the remaining numerical cases"
    elif not local_tested:
        goal = "run one tiny and one largest local case, then continue"
    else:
        goal = "finish the remaining local cases"

    result = {
        "decision": decision,
        "reason": reason,
        "resume_recommended": decision in {"continue", "retry_api", "retry_clean_card"},
        "resume_goal": goal if decision != "stop" else "",
        "evidence": evidence,
    }
    output = Path(args.output).resolve() if args.output else task / "triage.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
