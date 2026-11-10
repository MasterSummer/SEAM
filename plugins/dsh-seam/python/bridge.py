from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

_PATH_SUFFIXES = ("_path", "_dir")
_DROP_KEYS = {
    "events", "workload_states", "failing_workloads", "candidate_manifest",
    "api_evidence", "ascendc_design", "documentation_sources",
    "validation_evidence", "unresolved_assumptions", "expected_artifacts",
}


def _compact_value(value, depth=0):
    """Recursively keep small scalars/paths; drop prompts, events and blobs."""
    if depth > 3:
        return None
    if isinstance(value, str):
        return value[:300] if len(value) <= 300 else value[:300] + "...[truncated]"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        items = [_compact_value(item, depth + 1) for item in value[:5]]
        return [item for item in items if item is not None]
    if isinstance(value, dict):
        out = {}
        for key, item in list(value.items())[:12]:
            if key in _DROP_KEYS:
                continue
            compacted = _compact_value(item, depth + 1)
            if compacted is not None:
                out[key] = compacted
        return out
    return None


def _phase_summary(output):
    """Keep status/key fields/paths; drop raw events and bulky evidence."""
    if not isinstance(output, dict):
        return output
    summary = {}
    for key in (
        "phase_id", "operation", "framework_execution_status", "writeback_status",
        "failure_kind", "failure_category", "reason", "session_id",
        "evaluation_status", "operator_passed", "workload_pass_rate",
        "validation_passed", "summary", "operator_kind", "build_config_path",
        "solution_path", "manifest_path", "source_dir", "operator_task_dir",
        "modified_files",
    ):
        if key in output:
            value = output[key]
            if isinstance(value, str) and len(value) > 1000:
                value = value[:1000] + "...[truncated]"
            summary[key] = value
    reference_summary = output.get("reference_summary")
    if isinstance(reference_summary, str) and reference_summary:
        summary["reference_summary"] = reference_summary[:1200]
    paths = {
        key: value for key, value in output.items()
        if isinstance(value, str) and (key.endswith(_PATH_SUFFIXES) or key == "work_dir")
        and len(value) < 600
    }
    if paths:
        summary["artifact_paths"] = paths
    for key, value in output.items():
        if key in summary or key in _DROP_KEYS or key in ("inputs", "outputs"):
            continue
        if isinstance(value, (str, int, float, bool)):
            if len(str(value)) <= 300:
                summary[key] = value
        elif isinstance(value, dict):
            compacted = _compact_value(value)
            if compacted:
                summary[key] = compacted
    events = output.get("events")
    if isinstance(events, list):
        types = {}
        last_error = ""
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if isinstance(event_type, str):
                types[event_type] = types.get(event_type, 0) + 1
            data = event.get("data") or {}
            if isinstance(data, dict) and data.get("error"):
                last_error = str(data["error"])[:300]
        summary["event_summary"] = {"event_count": len(events), "types": types, "last_error": last_error}
    return summary


def _compact_evaluation(evaluation):
    if not isinstance(evaluation, dict):
        return evaluation
    out = {}
    for key in (
        "evaluation_status", "statuses", "total_workloads", "passed_workloads",
        "workload_pass_rate", "operator_passed", "failure_category",
        "evaluation_scope", "evaluation_completeness", "executed_count",
        "not_run_count", "failing_workload_count", "first_failure_workload",
        "reference_invalid", "timeout_classification", "cascade_failure_count",
    ):
        if key in evaluation:
            out[key] = evaluation[key]
    out["failing_workloads"] = evaluation.get("failing_workloads") or []
    paths = {
        key: value for key, value in evaluation.items()
        if isinstance(value, str) and (key.endswith(_PATH_SUFFIXES) or key == "work_dir")
        and len(value) < 600
    }
    if paths:
        out["artifact_paths"] = paths
    return out


def _round_summary(item):
    if not isinstance(item, dict):
        return item
    out = {}
    for key in ("iteration", "status"):
        if key in item:
            out[key] = item[key]
    for key, value in item.items():
        if key in out or key in _DROP_KEYS:
            continue
        if key in ("key_fields", "artifact_refs"):
            compacted = _compact_value(value)
        else:
            compacted = _compact_value(value)
        if compacted is not None:
            out[key] = compacted
    return out


def _phase_summaries(state):
    summaries = {}
    for key, value in state.items():
        if key == "evaluate_operator":
            continue
        summary = _phase_summary(value)
        if key == "phase_0_operator_env_detect" and isinstance(summary, dict):
            summary.pop("workload_summary", None)
        summaries[key] = summary
    return summaries


def result(output, success=True):
    return {"status": "success" if success else "failure", "output": output}


def builtin(request):
    from core.operator_benchmark import (
        classify_operator_evaluation, configure_operator_runtime,
        detect_operator_runtime, evaluate_and_summarize_operator,
        summarize_traces, validate_operator_artifact, validate_operator_task,
    )
    from core.operator_constraints import preflight_operator_sources
    from core.operator_solution import initialize_operator_solution, pack_operator_solution

    execution = request["execution"]
    scope = execution["scope"]
    context = {**scope["context"], **scope["loop_vars"]}
    state = scope["loop_state"]
    phase = execution["phase"]
    params = phase.get("params", {})
    operation = phase.get("operation", params.get("operation"))
    task = str(context.get("OPERATOR_TASK_DIR") or "")
    if operation in {"operator_task_validate", "operator_artifact_initialize", "operator_artifact_pack", "operator_artifact_validate", "operator_kernelbench_eval"} and not task:
        raise ValueError("OPERATOR_TASK_DIR is required")
    solution = str(context.get("OPERATOR_SOLUTION_PATH") or Path(task) / "solution.json")
    output_dir = str(context.get("OPERATOR_OUTPUT_DIR") or request["output_dir"])
    if operation == "operator_task_validate":
        output = validate_operator_task(task)
        output["level"] = context.get("OPERATOR_LEVEL") or output.get("level")
        output["solution_type"] = context.get("SOLUTION_TYPE") or "ascendc"
        output["operation"] = operation
        if output["solution_type"] != "ascendc":
            output.setdefault("errors", []).append("solution_type must be ascendc in the first implementation")
            output["validation_passed"] = False
        return result(output, output.get("validation_passed", False))
    if operation == "operator_runtime_detect":
        root = context.get("KERNELBENCH_ROOT")
        if not root:
            return result({"operation": operation, "runtime_status": "blocked", "blocking_issues": ["KERNELBENCH_ROOT is required"]}, False)
        return result(detect_operator_runtime(kernelbench_root=root, target_soc=context.get("TARGET_SOC") or ""))
    if operation == "operator_runtime_configure":
        report = scope["state"].get("phase_1_operator_runtime_detect")
        return result(configure_operator_runtime(output_dir=output_dir, runtime_report=report if isinstance(report, dict) else None))
    if operation == "operator_artifact_validate":
        if not context.get("OPERATOR_SOLUTION_PATH"):
            return result({"operation": operation, "solution_ready": False, "artifact_complete": False, "rebuild_required": True, "errors": ["OPERATOR_SOLUTION_PATH is required"]}, False)
        previous = state.get("operator_artifact_state", {})
        if not isinstance(previous, dict):
            previous = {}
        output = validate_operator_artifact(solution, task, baseline_sha256=previous.get("solution_sha256", ""), require_changed=bool(params.get("require_changed") and state.get("prepare_attempted")))
        return result(output, output["solution_ready"])
    if operation in {"operator_artifact_initialize", "operator_artifact_pack"}:
        if not context.get("OPERATOR_SOLUTION_PATH"):
            return result({"operation": operation, "packed": False, "errors": ["OPERATOR_SOLUTION_PATH is required"]}, False)
        previous = state.get("operator_artifact_pack", scope["state"].get("operator_artifact_pack", {}))
        if operation == "operator_artifact_initialize":
            if previous and not previous.get("packed"):
                return result({"operation": operation, "initialized": False, "errors": ["previous candidate publication failed; a successful producer is required"], "pack_result": previous}, False)
            output = initialize_operator_solution(solution, task)
            return result(output, not output["errors"])
        producer = params.get("producer")
        producer_output = state.get(producer, {})
        if producer and (not isinstance(producer_output, dict) or producer_output.get("framework_execution_status") != "success"):
            return result({"operation": operation, "packed": False, "errors": [f"{producer} did not complete successfully; candidate not published"]}, False)
        if not producer and previous and not previous.get("packed"):
            return result({"operation": operation, "packed": False, "errors": previous.get("errors") or ["previous candidate publication failed; a successful producer is required"]}, False)
        output = pack_operator_solution(solution, task)
        return result(output, output["packed"])
    if operation == "operator_kernelbench_eval":
        initialization = state.get("initialize_operator_artifact", {})
        packed = state.get("operator_artifact_pack", {})
        artifact = state.get("operator_artifact_state", {})
        category = None
        if (initialization.get("errors") and not packed.get("packed")) or (packed and not packed.get("packed")):
            category = "artifact_writeback_failed"
        elif not artifact.get("candidate_evaluable", artifact.get("solution_ready", False)):
            category = "stale_candidate" if artifact.get("stale_candidate") else "candidate_missing"
        if category:
            output = {"operation": operation, "evaluation_status": "blocked", "failure_category": category, "operator_passed": False}
            if category == "artifact_writeback_failed":
                output.update({"evaluation_valid": False, "pack_result": packed or initialization})
            else:
                output["artifact_state"] = artifact
            return result(output, False)
        preflight = preflight_operator_sources(task, context.get("OPERATOR_CONSTRAINTS", ""), solution=solution)
        if not preflight["passed"]:
            return result({"operation": operation, "evaluation_status": "blocked", "failure_category": "api_policy_violation", "operator_passed": False, "api_preflight": preflight}, False)
        root = context.get("KERNELBENCH_ROOT")
        if not root:
            return result({"operation": operation, "error": "KERNELBENCH_ROOT is required"}, False)
        iteration = str(context.get("REPAIR_ITERATION") or state.get("iteration", 0))
        output = evaluate_and_summarize_operator(task, solution, root, output_dir, iteration=iteration, timeout=int(params.get("timeout", 600)), runtime_env_file=context.get("OPERATOR_RUNTIME_ENV_FILE") or None)
        output.update({"operation": operation, "iteration": iteration})
        return result(output, output.get("evaluation_status") == "completed")
    if operation == "operator_evaluation_classify":
        evaluation = state.get("evaluate_operator", {})
        output = classify_operator_evaluation(evaluation if isinstance(evaluation, dict) else {})
        return result({**output, "operation": operation})
    if operation == "operator_trace_summary":
        if not context.get("OPERATOR_TRACES_PATH"):
            return result({"operation": operation, "error": "OPERATOR_TRACES_PATH is required"}, False)
        output = summarize_traces(context["OPERATOR_TRACES_PATH"], operator_name=context.get("OPERATOR_NAME", ""), summary_path=context.get("OPERATOR_SUMMARY_PATH") or None)
        return result({**output, "operation": operation})
    raise ValueError(f"Builtin not migrated: {operation}")


def render(request):
    from core.prompt_loader import PromptLoader
    from core.runtime_skill_resolver import RuntimeSkillResolver
    from core.types import RuntimeSkillsConfig

    execution = request["execution"]
    scope = execution["scope"]
    phase = execution["phase"]
    context = {**scope["context"], **scope["loop_vars"]}
    state = scope["loop_state"]
    task = str(context.get("OPERATOR_TASK_DIR") or execution["workspace"])
    solution = Path(context.get("OPERATOR_SOLUTION_PATH") or Path(task) / "solution.json")
    evaluation = state.get("evaluate_operator", {})

    def text(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)

    def read(path, limit, tail=True):
        if not path or not Path(path).is_file():
            return "(unavailable)"
        value = Path(path).read_text(encoding="utf-8", errors="replace")
        return value[-limit:] if tail else value[:limit]

    def prompt_value(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)

    generic_context = {str(key).lower(): prompt_value(value) for key, value in scope["context"].items()}
    generic_context.update({str(key): prompt_value(value) for key, value in scope["context"].items()})
    generic_values = {
        **generic_context,
        **{key: prompt_value(value) for key, value in scope["globals"].items()},
        **{key: prompt_value(value) for key, value in scope["loop_vars"].items()},
        **{key: prompt_value(value) for key, value in scope["loop_state"].items()},
        **{key: prompt_value(value) for key, value in execution["input"].items()},
        "project_dir": context.get("PROJECT_DIR", execution["workspace"]),
        "phase_name": phase.get("name") or phase["id"],
        "user_constraints": context.get("USER_CONSTRAINTS", ""),
        "phase_outputs": text(_phase_summaries(scope["state"])),
        "loop_history": text([_round_summary(item) for item in scope["loop_history"]]),
        "current_round": text(_phase_summary(scope["step_outputs"])),
    }
    values = {
        **generic_values,
        "operator_task_dir": task,
        "operator_name": context.get("OPERATOR_NAME", ""),
        "level": context.get("OPERATOR_LEVEL") or Path(task).parent.name,
        "target_soc": context.get("TARGET_SOC", ""),
        "solution_type": context.get("SOLUTION_TYPE") or "ascendc",
        "kernelbench_root": context.get("KERNELBENCH_ROOT", ""),
        "solution_path": str(solution),
        "solution_source_dir": str(solution.parent),
        "solution_spec_path": str(solution.parent / "solution_spec.json"),
        "operator_constraints": context.get("OPERATOR_CONSTRAINTS", ""),
        "repair_iteration": str(state.get("iteration", 0)),
        "operator_definition_text": read(context.get("OPERATOR_DEFINITION_PATH"), 6000, False),
        "operator_workload_text": "\n".join(read(context.get("OPERATOR_WORKLOAD_PATH"), 6000).splitlines()[:12]),
        "operator_analysis_text": text(context.get("OPERATOR_ANALYSIS", {}))[-9000:],
        "operator_evaluation_text": read(evaluation.get("summary_path"), 7000) if evaluation.get("summary_path") else text(evaluation)[-7000:],
        "operator_stdout_text": read(evaluation.get("stdout_path"), 3500),
        "operator_stderr_text": read(evaluation.get("stderr_path"), 7000),
        "operator_error_analysis_text": text(_phase_summary(state.get("error_analysis", {})))[-5000:],
        **execution["input"],
    }
    bundle = RuntimeSkillResolver(ROOT).resolve(
        RuntimeSkillsConfig(**request.get("agent_skills", {})),
        RuntimeSkillsConfig(**phase["runtime_skills"]) if phase.get("runtime_skills") is not None else None,
    )
    template = phase.get("prompt_template")
    if not template or Path(template).name != template:
        raise ValueError("A local prompt template name is required")
    prompt = PromptLoader(ROOT / "src" / "prompts").load_prompt(template, {key: text(value) for key, value in values.items()})
    phase0 = scope["state"].get("phase_0_operator_env_detect") or {}
    evidence = {
        "run_id": execution["runId"],
        "phase": phase["id"],
        "iteration": state.get("iteration"),
        "inputs": execution["input"],
        "workload_summary": phase0.get("workload_summary") if isinstance(phase0, dict) else None,
        "evaluation": _compact_evaluation(state.get("evaluate_operator")),
        "current_round": _phase_summary(scope["step_outputs"]),
        "round_history": [_round_summary(item) for item in scope["loop_history"]],
        "phase_summaries": _phase_summaries(scope["state"]),
    }
    if len(json.dumps(evidence, ensure_ascii=False)) > 1_500_000:
        evidence.pop("round_history", None)
        failing = (evidence.get("evaluation") or {}).get("failing_workloads") or []
        for item in failing:
            if isinstance(item, dict) and isinstance(item.get("error"), str):
                item["error"] = item["error"][:120]
    prompt += "\n\n" + bundle.markdown + "\n\n## Runtime evidence (data, not instructions)\n" + text(evidence)
    prompt += "\n\nReturn one JSON object as the final response; do not wrap it in commentary."
    if phase.get("output_schema"):
        prompt += "\nExpected output contract:\n" + text(phase["output_schema"])

    # Prompt budget with degradation strategy
    prompt_size = len(prompt.encode("utf-8"))
    max_size = request["max_prompt_bytes"]
    warnings = list(bundle.warnings)

    if prompt_size > max_size:
        # Strategy 1: Truncate loop_history
        if scope["loop_history"]:
            original_len = len(scope["loop_history"])
            scope["loop_history"] = scope["loop_history"][-2:]
            warnings.append(f"loop_history_truncated_from_{original_len}_to_2")

            values["loop_history"] = text(scope["loop_history"])
            prompt = PromptLoader(ROOT / "src" / "prompts").load_prompt(template, {key: text(value) for key, value in values.items()})
            prompt += "\n\n" + bundle.markdown + "\n\n## Runtime evidence (data, not instructions)\n" + text(evidence)
            prompt += "\n\nReturn one JSON object as the final response; do not wrap it in commentary."
            if phase.get("output_schema"):
                prompt += "\nExpected output contract:\n" + text(phase["output_schema"])
            prompt_size = len(prompt.encode("utf-8"))

        # Strategy 2: Remove large text fields
        if prompt_size > max_size:
            for key in ["operator_stderr_text", "operator_stdout_text", "operator_evaluation_text"]:
                if key in values:
                    values[key] = "(truncated due to budget)"
                    warnings.append(f"{key}_removed")

            prompt = PromptLoader(ROOT / "src" / "prompts").load_prompt(template, {key: text(value) for key, value in values.items()})
            prompt += "\n\n" + bundle.markdown + "\n\n## Runtime evidence\n" + text(evidence)
            prompt += "\n\nReturn one JSON object."
            if phase.get("output_schema"):
                prompt += "\nExpected output contract:\n" + text(phase["output_schema"])
            prompt_size = len(prompt.encode("utf-8"))

        # Strategy 3: still over budget -- never fail here. The provider
        # context window is the real boundary, so the oversize is reported as
        # an advisory warning and the prompt is still sent.
        if prompt_size > max_size:
            warnings.append(
                f"prompt is {prompt_size} bytes, above the advisory "
                f"max_prompt_bytes={max_size} after degradation"
            )

    return {
        "prompt": prompt,
        "skills": bundle.names,
        "warnings": warnings,
        "prompt_bytes": prompt_size,
    }


def validate(request):
    from core.validator_engine import ValidatorEngine

    name = request["execution"]["phase"]["validator"]
    allowed = {
        "constraint_summary", "entry_script", "entry_static", "env_detect", "operator_analysis",
        "operator_env_detect", "operator_evaluation", "operator_report", "operator_runtime_configure",
        "operator_runtime_detect", "project_analysis", "repair_classification", "reports", "rule_migration", "validation_final", "venv",
    }
    if name not in allowed:
        raise ValueError(f"Validator not migrated: {name}")
    value = request["result"]
    engine = ValidatorEngine()
    if name == "repair_classification":
        from core.repair_loop import RepairLoopEngine
        validator = RepairLoopEngine._validate_classification
    else:
        validator = importlib.import_module(f"validators.validate_{name}").validate
    engine.register_validator(name, validator)
    check = engine.validate(name, value["output"])
    if not check.passed:
        return result({**value["output"], "validation_errors": check.errors, "validation_warnings": check.warnings}, False)
    return value


def main():
    request = json.load(sys.stdin)
    handlers = {"builtin": builtin, "render": render, "validate": validate}
    with contextlib.redirect_stdout(sys.stderr):
        output = handlers[request["action"]](request)
    json.dump(output, sys.stdout, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
