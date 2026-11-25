#!/usr/bin/env python3
"""Build a compact failure card from a finished task workspace."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ERROR_RE = re.compile(r"(?P<file>[^\s:]+\.(?:cpp|cc|cxx|h|hpp)):(?P<line>\d+)(?::(?P<col>\d+))?:\s*(?P<msg>.+)")


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


def evidence(task: Path) -> dict:
    run = read_json(task.parent / "run.json")
    final = last_jsonl(task / "final-feedback.jsonl")
    metrics = final.get("metrics", {})
    failures = final.get("failures") or []
    local = read_json(task / "fast_eval" / "result.json")
    notes = task / "notes.md"
    return {
        "task": run.get("task") or task.parent.name,
        "agent_exit": run.get("agent_exit"),
        "duration_s": run.get("agent_duration_s"),
        "full_exit": run.get("full_exit"),
        "summary": final.get("summary"),
        "attribution": final.get("attribution"),
        "statuses": metrics.get("statuses") or {},
        "origins": metrics.get("failure_origins") or {},
        "local_status": local.get("status"),
        "local_tested": local.get("tested"),
        "local_passed": local.get("passed"),
        "notes_bytes": notes.stat().st_size if notes.is_file() else 0,
        "failure_count": len(failures),
        "failures": failures,
    }


def parse_build_error(task: Path) -> dict:
    candidates = [task / "fast_build" / "cmake_compile.log"]
    candidates += sorted(task.glob("evaluations/*/stderr.log"), reverse=True)
    candidates += sorted(task.glob("evaluations/*/stdout.log"), reverse=True)
    for path in candidates:
        if not path.is_file():
            continue
        lines = path.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            match = ERROR_RE.search(line)
            if not match:
                continue
            source = Path(match.group("file"))
            start = max(0, index - 2)
            snippet = lines[start:index + 4]
            context = []
            if source.is_file():
                src_lines = source.read_text(errors="replace").splitlines()
                line_no = int(match.group("line"))
                context = [
                    f"{n + 1}: {src_lines[n]}"
                    for n in range(max(0, line_no - 3), min(len(src_lines), line_no + 2))
                ]
            return {
                "log": str(path),
                "file": match.group("file"),
                "line": int(match.group("line")),
                "column": int(match.group("col")) if match.group("col") else None,
                "message": match.group("msg").strip(),
                "log_context": snippet,
                "source_context": context,
            }
    return {}


def probe_one(task: Path, uuid: str, device: int) -> dict:
    if not uuid:
        return {}
    venv = os.environ.get("FAST_PYTHON", "/home/yiding/npu-kernelbench/.venv/bin/python")
    cmd = [
        venv, str(Path(__file__).with_name("fast_iter.py")), "run",
        "--task", str(task), "--workloads", uuid, "--limit", "1",
        "--device", str(device), "--timeout", "300",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=360)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    result = read_json(task / "fast_eval" / "result.json")
    for record in result.get("records") or []:
        if str(record.get("workload")) == uuid:
            return {
                "uuid": uuid,
                "status": record.get("status"),
                "candidate_outputs": record.get("candidate_outputs"),
                "reference_outputs": record.get("reference_outputs"),
                "candidate_error": record.get("candidate_error"),
            }
    return {"uuid": uuid, "error": "no probe record"}

def first_by_status(failures: list[dict], *statuses: str) -> dict:
    wanted = set(statuses)
    for failure in failures:
        if failure.get("status") in wanted:
            return failure
    return failures[0] if failures else {}


def classify(ev: dict, build_error: dict) -> tuple[str, str]:
    if build_error or "COMPILE_ERROR" in ev["statuses"] or ev["attribution"] == "build":
        return "E0_build", "compile/build failure"
    if ev["attribution"] == "environment":
        return "E4_environment", "device/stream environment failure"
    if "RUNTIME_ERROR" in ev["statuses"] or "LAUNCH_ERROR" in ev["statuses"]:
        return "E1_runtime", "runtime/launch failure"
    if "INCORRECT_SHAPE" in ev["statuses"]:
        return "E2_shape", "output/input shape failure"
    if "INCORRECT_NUMERICAL" in ev["statuses"]:
        return "E3_numerical", "numerical failure with shape/runtime not blocking"
    return "E6_process", "no usable final result"


def process_flags(task: Path, ev: dict) -> list[str]:
    flags = []
    if ev["notes_bytes"] == 0:
        flags.append("notes_empty")
    if not ev["summary"] or ev["full_exit"] == 2:
        flags.append("no_final")
    if not (task / "fast_eval" / "result.json").is_file():
        flags.append("no_local_smoke")
    solution = task / "solution.json"
    if solution.is_file() and ev["statuses"].get("COMPILE_ERROR"):
        flags.append("build_failed_after_last_rewrite")
    if solution.is_file():
        data = read_json(solution)
        for entry in data.get("sources") or []:
            path = task / entry.get("path", "")
            if not path.is_file() or path.read_text(errors="replace") != entry.get("content", ""):
                flags.append("source_drift")
                break
    return flags


def build_questions(klass: str, build_error: dict, failure: dict) -> list[str]:
    if klass == "E0_build":
        return [
            f"Which source line emits the first error: {build_error.get('file', '?')}:{build_error.get('line', '?')}?",
            "Is it a missing include, undeclared type, template/type error, or link error?",
            "Did the last rewrite remove an extern C/AscendC entry or required header?",
        ]
    if klass == "E2_shape":
        shape = failure.get("shape") or {}
        return [
            "What is candidate output shape/dtype vs reference on one tiny case?",
            f"Input dim/shape: {json.dumps(shape, ensure_ascii=False)[:500]}",
            "Is negative dim normalized, and is the 2D/3D/4D branch correct?",
            "Does TilingFunc/GET_TILING_DATA carry the same rank and output shape?",
        ]
    if klass == "E1_runtime":
        return [
            "What is the first runtime error and failing UUID?",
            "Does the same UUID fail on a second clean card?",
            "Check workspace size, blockDim, UB/queue, 32B alignment, and launch args.",
        ]
    if klass == "E3_numerical":
        return [
            "Take one failing UUID; compare one element and one reduction axis.",
            "Is output shape/dtype already equal to reference?",
            "Which single change fixes this case without touching other cases?",
        ]
    if klass == "E4_environment":
        return ["Retry the same workspace on a clean card; do not inspect numerics yet."]
    return ["Resume the task or collect a final result first."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--uuid", default="")
    parser.add_argument("--no-notes", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    task = Path(args.task).resolve()
    ev = evidence(task)
    build_error = parse_build_error(task)
    failure = first_by_status(ev["failures"], "RUNTIME_ERROR", "INCORRECT_SHAPE", "INCORRECT_NUMERICAL")
    if args.uuid:
        failure = next((f for f in ev["failures"] if f.get("workload") == args.uuid), failure)
    klass, reason = classify(ev, build_error)
    flags = process_flags(task, ev)
    shape_evidence = {}
    if args.probe and klass not in ("E0_build", "E4_environment"):
        shape_evidence = probe_one(task, str(failure.get("workload") or ""), args.device)
    questions = build_questions(klass, build_error, failure)
    out_dir = Path(args.output).resolve() if args.output else task / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "class": klass,
        "reason": reason,
        "first_failure": {
            "uuid": failure.get("workload"),
            "status": failure.get("status"),
            "shape": failure.get("shape"),
            "max_abs": failure.get("max_absolute_error"),
            "max_rel": failure.get("max_relative_error"),
        },
        "build_error": build_error,
        "shape_evidence": shape_evidence,
        "evidence": ev,
        "process_flags": flags,
        "questions": questions,
        "next_single_check": questions[0] if questions else "",
    }
    (out_dir / "evidence.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card = [
        f"### {ev['task']}/{failure.get('workload', '?')}", "",
        f"- class: {klass}",
        f"- reason: {reason}",
        f"- first_error: {failure.get('status')}",
        f"- first_message: {failure.get('message') or ''}",
        f"- build_file_line: {build_error.get('file', '')}:{build_error.get('line', '')}",
        f"- build_message: {build_error.get('message', '')}",
        f"- candidate_output: {json.dumps((shape_evidence.get('candidate_outputs') if shape_evidence else None), ensure_ascii=False)}",
        f"- reference_output: {json.dumps((shape_evidence.get('reference_outputs') if shape_evidence else None), ensure_ascii=False)}",
        f"- candidate_error: {shape_evidence.get('candidate_error') if shape_evidence else None}",
        f"- local_result: {ev['local_status']} tested={ev['local_tested']} passed={ev['local_passed']}",
        f"- process_flags: {', '.join(flags) if flags else 'none'}",
        f"- next_single_check: {payload['next_single_check']}",
        "", "#### First failure shape", "",
        "```json", json.dumps(failure.get("shape") or {}, ensure_ascii=False, indent=2)[:3000], "```",
    ]
    if build_error:
        card += ["", "#### Build error", "", "```text", build_error.get("message", ""), "```"]
        if build_error.get("source_context"):
            card += ["", "```cpp", *build_error["source_context"], "```"]
    card += ["", "#### Questions", ""] + [f"- {q}" for q in questions]
    (out_dir / "failure-card.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    if not args.no_notes:
        notes = task / "notes.md"
        with notes.open("a", encoding="utf-8") as stream:
            stream.write("\n\n---\n\n")
            stream.write((out_dir / "failure-card.md").read_text(encoding="utf-8"))
    print(json.dumps({"class": klass, "card": str(out_dir / "failure-card.md"), "flags": flags}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
