#!/usr/bin/env python3
"""Build a replay manifest from the latest full feedback.

A full failure is only useful for the next iteration if the failing UUIDs are
turned into local cases.  This script writes that list without modifying any
agent state; `--run` additionally executes fast_iter on the replay set.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def latest_envelope(path: Path) -> dict:
    if not path.is_file():
        return {}
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                pass
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--feedback", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    task = Path(args.task).expanduser().resolve()
    feedback = Path(args.feedback).expanduser().resolve() if args.feedback else task / "final-feedback.jsonl"
    envelope = latest_envelope(feedback)
    failures = envelope.get("failures") or []
    uuids = []
    for item in failures:
        uuid = item.get("workload")
        if uuid and uuid not in uuids:
            uuids.append(str(uuid))
    data = {
        "task": task.name,
        "feedback": str(feedback),
        "status": envelope.get("status"),
        "summary": envelope.get("summary"),
        "failed_uuids": uuids,
        "failure_statuses": {},
        "command": "",
    }
    for item in failures:
        status = str(item.get("status") or "UNKNOWN")
        data["failure_statuses"][status] = data["failure_statuses"].get(status, 0) + 1
    if uuids:
        data["command"] = (
            f"python3 src/scripts/fast_iter.py run --task {task} "
            f"--workloads {','.join(uuids)} --device {args.device} "
            f"--feedback {task / 'replay-feedback.jsonl'}"
        )
    output = Path(args.output).expanduser().resolve() if args.output else task / "replay.json"
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False))

    if args.run and uuids:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "fast_iter.py"),
            "run", "--task", str(task), "--workloads", ",".join(uuids),
            "--device", str(args.device), "--feedback", str(task / "replay-feedback.jsonl"),
        ]
        return subprocess.run(cmd, cwd=str(task.parent)).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
