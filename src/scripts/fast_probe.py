#!/usr/bin/env python3
"""Run selected workloads against a locally built kernel.so (diagnostic only)."""
from __future__ import annotations

import json
import sys
import traceback
import importlib.util
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.solution import Solution
from npu_kernelbench.data.workload import ToleranceSpec, Workload
from npu_kernelbench.evaluator.default import DefaultEvaluator, _run_reference
from npu_kernelbench.evaluator.evaluator import BaselineHandle


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

    # Task-card validation hook: per-case tolerance overrides.
    checks_path = task / "validation" / "checks.json"
    checks = {}
    if checks_path.is_file():
        for item in json.loads(checks_path.read_text()).get("cases", []):
            if item.get("uuid"):
                checks[item["uuid"]] = item.get("tolerance") or {}
    # A task-specific reference module may replace definition.reference.
    module_path = task / "validation" / "module.py"
    if module_path.is_file():
        definition = definition.model_copy(update={"reference": module_path.read_text()})

    prepare = None
    prepare_path = task / "validation" / "prepare_inputs.py"
    if prepare_path.is_file():
        prep_spec = importlib.util.spec_from_file_location("fast_prepare_inputs", prepare_path)
        prep_module = importlib.util.module_from_spec(prep_spec)
        assert prep_spec.loader is not None
        prep_spec.loader.exec_module(prep_module)
        prepare = getattr(prep_module, "prepare", None)

    torch.ops.load_library(str(task / "kernel.so"))

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
        if workload.uuid in checks:
            workload = workload.model_copy(update={"tolerance": ToleranceSpec(**checks[workload.uuid])})
        try:
            inputs = None
            if callable(prepare):
                inputs = prepare(workload, "npu:0", config.seed)
            if inputs is not None:
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]
                outputs = _run_reference(definition.reference, list(inputs), list(definition.outputs.keys()))
                baseline = BaselineHandle(
                    definition_name=definition.name, device="npu:0",
                    workload_uuid=workload.uuid, inputs=list(inputs), outputs=outputs,
                )
            else:
                baseline = evaluator.build_baseline(definition, workload, "npu:0", config)
            candidate_outputs = None
            reference_outputs = None
            candidate_error = None
            try:
                candidate_value = fn(*list(baseline.inputs))
                if torch.is_tensor(candidate_value):
                    candidate_value = [candidate_value]
                if isinstance(candidate_value, dict):
                    candidate_value = [candidate_value[k] for k in definition.outputs]
                candidate_outputs = [
                    {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
                    for t in candidate_value if torch.is_tensor(t)
                ]
                reference_outputs = [
                    {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
                    for t in (baseline.outputs or {}).values() if torch.is_tensor(t)
                ]
            except Exception as exc:  # noqa: BLE001
                candidate_error = f"{type(exc).__name__}: {exc}"[:300]
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
                "candidate_outputs": candidate_outputs,
                "reference_outputs": reference_outputs,
                "candidate_error": candidate_error,
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
