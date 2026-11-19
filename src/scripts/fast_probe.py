#!/usr/bin/env python3
"""Run selected workloads against a locally built kernel.so (diagnostic only)."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.solution import Solution
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.default import DefaultEvaluator


def status_value(value: object) -> str:
    return str(getattr(value, "value", value))


def main() -> int:
    task = Path(sys.argv[1]).resolve()
    solution = Solution.model_validate(json.loads((task / "solution.json").read_text()))
    definition = Definition.model_validate(json.loads((task / "definition.json").read_text()))
    workloads = [
        Workload.model_validate(json.loads(line))
        for line in (task / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    torch.ops.load_library(str(task / "kernel.so"))

    import importlib.util

    entry = task / "sources" / solution.get_entry_path()
    spec = importlib.util.spec_from_file_location("fast_candidate", entry)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    fn = getattr(module, solution.get_entry_symbol())

    config = BenchmarkConfig(
        benchmark_reference=False,
        warmup_runs=0,
        iterations=1,
        npu_id=0,
        npu_ids=[0],
        enable_anti_hack=[],
        solution_source_dir=str(task / "sources"),
    )
    evaluator = DefaultEvaluator()
    for workload in workloads:
        try:
            baseline = evaluator.build_baseline(definition, workload, "npu:0", config)
            trace = evaluator.evaluate(definition, workload, fn, baseline, config)
            evaluation = trace.evaluation
            correctness = evaluation.correctness
            print(json.dumps({
                "workload": workload.uuid,
                "status": status_value(evaluation.status),
                "max_abs": getattr(correctness, "max_absolute_error", None),
                "max_rel": getattr(correctness, "max_relative_error", None),
                "has_nan": getattr(correctness, "has_nan", None),
                "has_inf": getattr(correctness, "has_inf", None),
                "log": (evaluation.log or "")[:500],
            }, ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({
                "workload": workload.uuid,
                "status": "RUNTIME_ERROR",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "traceback": traceback.format_exc()[-1000:],
            }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
