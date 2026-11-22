#!/usr/bin/env python3
"""Check one NPU card is healthy enough to start a task."""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

PYTHON = os.environ.get("FAST_PYTHON", "/home/yiding/npu-kernelbench/.venv/bin/python")


def probe(card: int) -> tuple[bool, str]:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(card)
    code = "import torch, torch_npu; assert torch.npu.is_available(); assert torch.npu.device_count()==1; x=torch.randn(1024, device='npu'); print(float(x.sum().cpu()))"
    try:
        p = subprocess.run([PYTHON, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "probe timeout"
    if p.returncode != 0:
        return False, (p.stderr or p.stdout)[-500:]
    return True, p.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    for attempt in range(max(1, args.retries)):
        ok, message = probe(args.card)
        print(f"card={args.card} attempt={attempt + 1} ok={ok} {message}", flush=True)
        if ok:
            return 0
        time.sleep(5)
    return 1


if __name__ == "__main__":
    sys.exit(main())
