#!/usr/bin/env python3
"""Failure-analyst subagent: propose the next bounded investigation, no code edits."""
from __future__ import annotations
import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

CONFIG = Path("/home/yiding/seam/.opencode/opencode.jsonc")


def load_provider() -> tuple[str, str, str]:
    base = os.environ.get("SEAM_ANALYST_BASE", "https://apicz.boyuerichdata.com/v1")
    model = os.environ.get("SEAM_ANALYST_MODEL", "qwen3.8-27b")
    key = os.environ.get("SEAM_ANALYST_KEY", "")
    if not key and CONFIG.is_file():
        text = CONFIG.read_text()
        index = text.find('"chuangzhi"')
        match = re.search(r'"apiKey"\s*:\s*"([^"]+)"', text[index:])
        if match:
            key = match.group(1)
    return base, model, key


def fallback(task: Path) -> dict:
    card = task / "analysis" / "failure-card.md"
    questions = []
    qpath = task / "analysis" / "questions.json"
    if qpath.is_file():
        try:
            questions = [str(x) for x in json.loads(qpath.read_text())]
        except json.JSONDecodeError:
            questions = []
    return {
        "root_cause_hypotheses": [],
        "missing_evidence": questions[:3],
        "next_experiments": ([{"name": "deterministic next check", "command": "", "expected": questions[0]}] if questions else []),
        "priority": "shape_before_numerical",
        "next_single_change": questions[0] if questions else "",
        "fallback": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    task = Path(args.task).resolve()
    analysis = task / "analysis"
    card_path = analysis / "failure-card.md"
    if not card_path.is_file():
        print(json.dumps({"error": "failure-card missing"}))
        return 1
    card = card_path.read_text(errors="replace")[:8000]
    base, model, key = load_provider()
    prompt = """You are a failure-analyst subagent for an Ascend C kernel task.
Do not edit code. Do not run full evaluation. Use only the failure card below.
Return JSON only with this schema:
{
  "root_cause_hypotheses": ["..."],
  "missing_evidence": ["..."],
  "next_experiments": [{"name": "...", "command": "...", "expected": "..."}],
  "priority": "build|shape|runtime|numerical|environment",
  "next_single_change": "..."
}
Rules:
- At most 3 hypotheses.
- At most 3 missing evidence items.
- Exactly 1 next experiment.
- If the class is build, focus on the first compiler error/source line.
- If shape, ask for candidate vs reference shape/dtype and dim/tiling mapping.
- If runtime, ask for the first runtime UUID and clean-card reproduction.
- If numerical, ask for one element/reduction comparison.
- If environment, say retry clean card, no code change.
Failure card:
""" + card
    fallback_value = fallback(task)
    if not key:
        result = fallback_value
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON. You are a read-only failure analyst."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 700,
            "temperature": 0,
        }
        try:
            request = urllib.request.Request(
                base.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                data = json.loads(response.read())
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            match = re.search(r"\{.*\}", content, re.S)
            result = json.loads(match.group(0)) if match else fallback_value
        except Exception as exc:  # noqa: BLE001
            result = fallback_value
            result["error"] = f"{type(exc).__name__}: {exc}"[:300]
    output = Path(args.output).resolve() if args.output else analysis / "plan.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
