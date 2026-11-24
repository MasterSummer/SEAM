#!/usr/bin/env python3
"""Generate a task card from definition.json + workload.jsonl.

The card is intentionally built without any solution/answer.  It makes the
task contract explicit: semantics, input/output descriptions, a per-case
shape/dtype/attr table, and a grouped shape matrix.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

GE_DTYPE = {
    "float32": "ge::DT_FLOAT", "float16": "ge::DT_FLOAT16", "bfloat16": "ge::DT_BF16",
    "int64": "ge::DT_INT64", "int32": "ge::DT_INT32", "int16": "ge::DT_INT16",
    "int8": "ge::DT_INT8", "uint8": "ge::DT_UINT8", "bool": "ge::DT_BOOL",
}
SIZE_BYTES = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1,
    "uint8": 1, "bool": 1,
}


def load_workloads(task: Path) -> list[dict]:
    result = []
    for line in (task / "workload.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def tensor_specs(workload: dict) -> list[tuple[str, list[int], str]]:
    result = []
    inputs = workload.get("inputs") or {}
    for name, spec in inputs.items():
        if not isinstance(spec, dict):
            continue
        shape = spec.get("shape")
        dtype = spec.get("dtype", "")
        if isinstance(shape, list) and dtype:
            result.append((name, [int(dim) for dim in shape], str(dtype)))
    return result


def attrs(workload: dict) -> dict[str, Any]:
    result = {}
    for name, spec in (workload.get("inputs") or {}).items():
        if isinstance(spec, dict) and spec.get("type") == "scalar":
            result[name] = spec.get("value")
    return result


def classify(workload: dict) -> dict:
    specs = tensor_specs(workload)
    ranks = sorted({len(shape) for _, shape, _ in specs})
    dtypes = sorted({dtype for _, _, dtype in specs})
    tail = False
    empty = False
    non_last = False
    max_numel = 0
    for _, shape, dtype in specs:
        if any(dim == 0 for dim in shape):
            empty = True
        if len(shape) >= 2 and any(dim > 1 for dim in shape[:-1]):
            non_last = True
        if shape:
            width = SIZE_BYTES.get(dtype, 4)
            if int(shape[-1]) * width % 32 != 0:
                tail = True
        numel = 1
        for dim in shape:
            numel *= dim
        max_numel = max(max_numel, numel)
    return {
        "shapes": specs,
        "dtypes": dtypes,
        "ranks": ranks,
        "attrs": attrs(workload),
        "tail": tail,
        "empty": empty,
        "non_last": non_last,
        "max_numel": max_numel,
    }


def make_card(task: Path, output: Path) -> dict:
    definition = json.loads((task / "definition.json").read_text(encoding="utf-8"))
    workloads = load_workloads(task)
    output.mkdir(parents=True, exist_ok=True)
    (output / "question" / "op_host").mkdir(parents=True, exist_ok=True)
    (output / "question" / "op_kernel").mkdir(parents=True, exist_ok=True)
    (output / "validation").mkdir(parents=True, exist_ok=True)

    name = definition.get("name", task.name)
    inputs = definition.get("inputs") or {}
    outputs = definition.get("outputs") or {}
    description = definition.get("description") or ""

    lines = [f"# {name}", "", "## Functional Description", "", description.strip(), ""]
    lines += ["## Inputs", "", "| name | description |", "|---|---|"]
    for key, value in inputs.items():
        lines.append(f"| `{key}` | {str((value or {}).get('description', '')).replace('|', '/')} |")
    lines += ["", "## Outputs", "", "| name | description |", "|---|---|"]
    for key, value in outputs.items():
        lines.append(f"| `{key}` | {str((value or {}).get('description', '')).replace('|', '/')} |")
    lines += ["", "## Reference semantics", "", "```python", str(definition.get("reference", "")).strip(), "```", ""]
    lines += [
        "## Case matrix",
        "",
        "See `cases.csv` for every workload (shape/dtype/attr), and",
        "`shape_matrix.csv` for the grouped coverage matrix.",
        "",
        "## Packaging contract",
        "",
        "- `main.py::run` is the Python entry point.",
        "- Host registration and kernel are under `op_host/` and `op_kernel/`.",
        "- `solution.json` must embed all sources and the build spec.",
        "- Do not use the reference implementation as the candidate.",
        "- Shape, dtype, attribute, tail, and empty cases are explicit in `cases.csv`.",
        "",
    ]
    (output / "api_desc.md").write_text("\n".join(lines), encoding="utf-8")

    (output / "validation").mkdir(parents=True, exist_ok=True)
    (output / "cases.jsonl").write_text("", encoding="utf-8")
    case_rows = []
    matrix_groups: dict[tuple, list[str]] = {}
    check_rows = []
    for index, workload in enumerate(workloads):
        info = classify(workload)
        uuid = str(workload.get("uuid") or f"case-{index:03d}")
        shapes = "; ".join(f"{name}:{shape}" for name, shape, _ in info["shapes"])
        dtypes = ",".join(info["dtypes"])
        attrs_text = json.dumps(info["attrs"], ensure_ascii=False, sort_keys=True)
        tolerance = workload.get("tolerance") or {}
        case_rows.append({
            "case_id": index,
            "uuid": uuid,
            "shapes": shapes,
            "dtypes": dtypes,
            "attrs": attrs_text,
            "ranks": ",".join(map(str, info["ranks"])),
            "tail": int(info["tail"]),
            "empty": int(info["empty"]),
            "non_last": int(info["non_last"]),
            "max_numel": info["max_numel"],
            "atol": tolerance.get("max_atol"),
            "rtol": tolerance.get("max_rtol"),
            "required_ratio": tolerance.get("required_matched_ratio"),
            "max_error_cap": tolerance.get("max_error_cap"),
        })
        check_rows.append({"case_id": index, "uuid": uuid, "tolerance": tolerance})
        key = (dtypes, tuple(info["ranks"]), info["tail"], info["empty"], info["non_last"])
        matrix_groups.setdefault(key, []).append(uuid)
        with (output / "cases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"case_id": index, **workload}, ensure_ascii=False) + "\n")

    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]) if case_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(case_rows)

    (output / "validation" / "cases.csv").write_text(
        (output / "cases.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (output / "validation" / "cases.jsonl").write_text(
        (output / "cases.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    (output / "validation" / "checks.json").write_text(
        json.dumps({"cases": check_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "validation" / "README.md").write_text(
        "# Validation contract\n\n"
        "`cases.csv` is the shape/dtype/attr/tolerance matrix; `checks.json` keeps the\n"
        "per-case tolerance. Task-specific custom checks belong here, not in a global evaluator.\n\n"
        "`prepare_inputs.py` may define `prepare(workload, device, seed)` returning a list\n"
        "of tensors in definition input order. Return None to use the official generator.\n",
        encoding="utf-8")
    (output / "validation" / "prepare_inputs.py").write_text(
        "def prepare(workload, device=\"npu:0\", seed=200):\n"
        "    \"\"\"Return a list of inputs in definition order, or None to use the official generator.\"\"\"\n"
        "    return None\n",
        encoding="utf-8")
    with (output / "shape_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dtypes", "ranks", "tail", "empty", "non_last", "cases", "count"])
        for (dtypes, ranks, tail, empty, non_last), uuids in sorted(matrix_groups.items()):
            writer.writerow([dtypes, ",".join(map(str, ranks)), int(tail), int(empty), int(non_last),
                             ";".join(uuids), len(uuids)])

    # Implementation checklist + required local cases (no answer).
    op_name = name
    lowered = op_name.lower()
    checks = [
        "Output shape and dtype exactly match the reference.",
        "Every reduction axis is specified explicitly (dim/axis/keepdim).",
        "All input ranks and empty tensors are handled.",
        "Tail elements use a padded 32B window with masking; no sub-32B DataCopyPad.",
        "Scratch/UB buffer sizes are checked against the largest tile.",
        "dtype cast/round mode matches the reference (especially f32->f16/bf16).",
        "solution.json sources are byte-identical to the workspace sources.",
        "No debug/probe/TODO kernel is submitted as the final solution.",
        "Workspace, blockDim, and UB/queue sizes are checked for the largest case before full.",
        "One tiny and one largest case both ran without RUNTIME_ERROR before full.",
        "At least one tiny, one tail/edge, and one large local case passed.",
    ]
    if "quant" in lowered:
        checks += [
            "Quantization: scale = max_abs / 127, clamped to >= 1e-10.",
            "Quantization: round is ties-to-even, then clamp to [-128,127].",
        ]
    if any(key in inputs for key in ("index", "indices")):
        checks += [
            "Index semantics: axis/dim, bounds, and duplicate indices are explicit.",
            "Scatter/index-put duplicate behavior is defined (last-wins or accumulate).",
        ]
    if "iou" in lowered or "nms" in lowered:
        checks += ["Zero-area boxes produce the defined result (no accidental 0/0 NaN)."]
    if "cumsum" in lowered:
        checks += ["Accumulation dtype and large-value overflow are explicit."]
    if any((spec or {}).get("type") == "scalar" for spec in inputs.values()):
        checks += ["Scalar attrs are passed as attrs, not tensor inputs."]
    checklist = ["# Implementation checklist", ""]
    checklist += [f"- [ ] {item}" for item in checks]
    checklist += ["", "Run `review_solution.py --task <workspace>` before final full.", ""]
    (output / "checklist.md").write_text("\n".join(checklist), encoding="utf-8")

    # Required local cases: min, tail, empty, non-last, largest, one per dtype.
    required = {}
    if case_rows:
        def add(reason, row):
            if row and row.get("uuid") not in required:
                required[row["uuid"]] = reason
        add("tiny", min(case_rows, key=lambda r: int(r.get("max_numel") or 0)))
        add("largest", max(case_rows, key=lambda r: int(r.get("max_numel") or 0)))
        tail = next((r for r in case_rows if r.get("tail") == "1"), None)
        add("tail", tail)
        empty = next((r for r in case_rows if r.get("empty") == "1"), None)
        add("empty", empty)
        nonlast = next((r for r in case_rows if r.get("non_last") == "1"), None)
        add("non_last", nonlast)
        by_dtype = {}
        for row in case_rows:
            for dtype in str(row.get("dtypes") or "").split(","):
                by_dtype.setdefault(dtype, row)
        for dtype, row in sorted(by_dtype.items()):
            add(f"dtype:{dtype}", row)
    with (output / "required_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["reason", "uuid"])
        for uuid, reason in required.items():
            writer.writerow([reason, uuid])

    # Engineering skeletons only: no algorithm, no answer.
    input_names = list(inputs)
    output_names = list(outputs)
    scalar_names = set()
    if workloads:
        for name in input_names:
            specs = [(workload.get("inputs") or {}).get(name) or {} for workload in workloads]
            if specs and all(spec.get("type") == "scalar" for spec in specs):
                scalar_names.add(name)
    tensor_inputs = [name for name in input_names if name not in scalar_names]
    all_dtypes = []
    for workload in workloads:
        for name in tensor_inputs:
            spec = (workload.get("inputs") or {}).get(name) or {}
            dtype = spec.get("dtype")
            if dtype and dtype not in all_dtypes:
                all_dtypes.append(dtype)
    ge_dtypes = [GE_DTYPE.get(dtype, "ge::DT_FLOAT16") for dtype in all_dtypes] or ["ge::DT_FLOAT16"]
    class_name = op_name[:1].upper() + op_name[1:]
    host = [
        '#include "register/op_def_registry.h"',
        '#include "tiling/platform/platform_ascendc.h"',
        '',
        '// SKELETON: fill tiling data and launch parameters; no algorithm here.',
        'namespace optiling {',
        'static ge::graphStatus TilingFunc(gert::TilingContext *context) {',
        '    context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());',
        '    return ge::GRAPH_SUCCESS;',
        '}',
        '}  // namespace optiling',
        '',
        'namespace ops {',
        f'class {class_name} : public OpDef {{',
        'public:',
        f'    explicit {class_name}(const char *name) : OpDef(name) {{',
    ]
    for input_name in tensor_inputs:
        host += [f'        this->Input("{input_name}")',
                 '            .ParamType(REQUIRED)',
                 f'            .DataType({{{", ".join(ge_dtypes)}}})',
                 '            .Format({ge::FORMAT_ND});']
    for output_name in output_names:
        host += [f'        this->Output("{output_name}")',
                 '            .ParamType(REQUIRED)',
                 f'            .DataType({{{", ".join(ge_dtypes)}}})',
                 '            .Format({ge::FORMAT_ND});']
    for scalar_name in sorted(scalar_names):
        host += [f'        this->Attr("{scalar_name}").AttrType(OPTIONAL).Int(0);']
    host += [
        '        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");',
        '    }',
        '};',
        f'OP_ADD({class_name});',
        '}  // namespace ops',
        '',
    ]
    host_path = output / "question" / "op_host" / f"{op_name.lower()}.cpp"
    host_path.write_text("\n".join(host), encoding="utf-8")
    kernel_args = ", ".join([f"GM_ADDR {item}" for item in tensor_inputs] + [f"GM_ADDR {item}" for item in output_names])
    kernel = [
        '#include "kernel_operator.h"',
        '',
        '// SKELETON: fill GET_TILING_DATA, queues, Compute, and tail handling.',
        '// For a tail smaller than 32B, pad to a 32B window and mask lanes;',
        '// do not issue a sub-32B DataCopyPad.',
        f'extern "C" __global__ __aicore__ void {op_name.lower()}({kernel_args}, GM_ADDR workspace, GM_ADDR tiling) {{',
        '    // TODO: GET_TILING_DATA(tilingData, tiling);',
        '    // TODO: allocate queues and implement CopyIn / Compute / CopyOut.',
        '}',
        '',
    ]
    kernel_path = output / "question" / "op_kernel" / f"{op_name.lower()}.cpp"
    kernel_path.write_text("\n".join(kernel), encoding="utf-8")
    (output / "question" / "README.md").write_text(
        "# Packaging skeleton\n\n"
        "`op_host/<name>.cpp` and `op_kernel/<name>.cpp` are skeletons only.\n"
        "Fill tiling, queues, compute, and tail handling; do not copy reference/answer.\n",
        encoding="utf-8",
    )
    metadata = {
        "name": name,
        "task": str(task),
        "num_cases": len(workloads),
        "inputs": inputs,
        "outputs": outputs,
        "shape_matrix_cases": len(matrix_groups),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    task = Path(args.task).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else task / "task_card"
    metadata = make_card(task, output)
    print(json.dumps({"status": "success", "output": str(output), **metadata}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
