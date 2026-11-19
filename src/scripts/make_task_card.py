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

    case_rows = []
    matrix_groups: dict[tuple, list[str]] = {}
    (output / "cases.jsonl").write_text("", encoding="utf-8")
    for index, workload in enumerate(workloads):
        info = classify(workload)
        uuid = str(workload.get("uuid") or f"case-{index:03d}")
        shapes = "; ".join(f"{name}:{shape}" for name, shape, _ in info["shapes"])
        dtypes = ",".join(info["dtypes"])
        attrs_text = json.dumps(info["attrs"], ensure_ascii=False, sort_keys=True)
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
        })
        key = (dtypes, tuple(info["ranks"]), info["tail"], info["empty"], info["non_last"])
        matrix_groups.setdefault(key, []).append(uuid)
        with (output / "cases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"case_id": index, **workload}, ensure_ascii=False) + "\n")

    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]) if case_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(case_rows)

    with (output / "shape_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dtypes", "ranks", "tail", "empty", "non_last", "cases", "count"])
        for (dtypes, ranks, tail, empty, non_last), uuids in sorted(matrix_groups.items()):
            writer.writerow([dtypes, ",".join(map(str, ranks)), int(tail), int(empty), int(non_last),
                             ";".join(uuids), len(uuids)])

    (output / "question" / "README.md").write_text(
        "# Packaging skeleton\n\n"
        "Implement the operator here; do not copy the reference implementation.\n"
        "Use `api_desc.md` and `cases.csv` as the contract.\n",
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
