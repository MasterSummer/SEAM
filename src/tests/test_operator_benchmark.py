"""Tests for single-operator KernelBench helpers."""

import json
from pathlib import Path

from core.config import load_workflow
from core.operator_benchmark import (
    _classify_timeout,
    _evaluator_lock_path,
    _index_diagnostic_artifacts,
    _prepare_evaluation_runtime,
    _select_timeout_workload,
    build_candidate_manifest,
    build_failure_manifest,
    classify_operator_evaluation,
    configure_operator_runtime,
    format_diagnostic_artifacts_context,
    summarize_traces,
    validate_operator_artifact,
    validate_operator_task,
)
from validators.validate_operator_env_detect import validate as validate_env
from validators.validate_operator_evaluation import validate as validate_evaluation
from validators.validate_operator_error_analysis import validate as validate_error_analysis


def test_operator_error_analysis_contract_accepts_no_probe_and_rejects_bad_role() -> None:
    analysis = {
        "category": "unknown",
        "root_cause": "",
        "repair_role": "none",
        "evaluation_valid": False,
        "requires_clean_rerun": True,
        "operator_code_evidence": False,
        "suggested_fix": "",
        "failed_workloads": [],
        "evidence": [],
        "missing_evidence": [],
        "hypotheses": [],
        "next_verification": [],
        "handoff": "Need a focused diagnostic before repair.",
    }
    assert validate_error_analysis(analysis)["passed"] is True

    invalid = {**analysis, "repair_role": "timeout_analyzer"}
    result = validate_error_analysis(invalid)
    assert result["passed"] is False
    assert any("repair_role" in error for error in result["errors"])


def test_kernelbench_workflow_combines_generation_evaluation_and_repair_loop() -> None:
    workflow = load_workflow("src/workflows/npu_kernelbench_custom_op.yaml")

    assert [phase.id for phase in workflow.phases] == [
        "phase_0_operator_env_detect",
        "phase_1_operator_runtime_detect",
        "phase_1_5_operator_runtime_configure",
        "phase_2_operator_analysis",
        "phase_3_custom_op_prepare",
        "phase_4_operator_report",
    ]
    generation_loop = workflow.sub_workflows["operator_repair_loop"]
    phases = {phase["id"]: phase for phase in generation_loop.phases}
    order = list(phases)
    assert phases["initialize_operator_artifact"]["operation"] == "operator_artifact_initialize"
    assert phases["validate_operator_artifact"]["operation"] == "operator_artifact_validate"
    assert order.index("initialize_operator_artifact") < order.index("validate_operator_artifact")
    assert phases["pack_existing_operator"]["operation"] == "operator_artifact_pack"
    assert order.index("initialize_operator_artifact") < order.index("pack_existing_operator")
    assert order.index("pack_existing_operator") < order.index("validate_operator_artifact")
    assert "operator_artifact_state.rebuild_required" in phases["prepare_operator"]["condition"]
    assert "solution_ready" not in phases["prepare_operator"]["condition"]
    assert phases["prepare_operator"]["on_failure"] == "continue"
    assert phases["pack_prepared_operator"]["params"]["producer"] == "prepare_operator"
    assert order.index("prepare_operator") < order.index("pack_prepared_operator")
    assert order.index("pack_prepared_operator") < order.index("refresh_operator_artifact")
    assert phases["refresh_operator_artifact"]["output_as"] == "operator_artifact_state"
    assert not phases["refresh_operator_artifact"].get("params", {}).get("require_changed")
    assert order.index("refresh_operator_artifact") < order.index("evaluate_operator")
    assert "condition" not in phases["evaluate_operator"]
    assert order.index("evaluate_operator") < order.index("classify_operator_evaluation")
    assert phases["analyze_operator_failure"]["validator"] == "operator_error_analysis"
    assert phases["analyze_operator_failure"]["condition"] == (
        "${loop_state.operator_passed} != true"
    )
    assert phases["fix_operator_environment"]["condition"] == (
        "${loop_state.operator_passed} != true "
        "and ${loop_state.error_analysis.repair_role} == 'operator_environment_fixer'"
    )
    assert phases["fix_operator"]["condition"] == (
        "${loop_state.operator_passed} != true "
        "and ${loop_state.error_analysis.repair_role} == 'operator_fixer'"
    )
    assert phases["fix_operator_environment"]["on_failure"] == "continue"
    assert phases["fix_operator"]["on_failure"] == "continue"
    assert phases["evaluate_operator"]["on_failure"] == "continue"
    assert phases["pack_repaired_operator"]["params"]["producer"] == "fix_operator"
    assert phases["pack_repaired_operator"]["condition"] == phases["fix_operator"]["condition"]
    assert order.index("fix_operator") < order.index("pack_repaired_operator")
    assert "$.error_analysis.repair_role == 'none'" in generation_loop.stop_conditions[2]["condition"]


def test_kernelbench_agents_share_path_only_documentation() -> None:
    workflow = load_workflow("src/workflows/npu_kernelbench_custom_op.yaml")
    analysis = next(
        phase for phase in workflow.phases if phase.id == "phase_2_operator_analysis"
    )
    assert analysis.runtime_skills is not None
    assert analysis.runtime_skills.include == ["cuda-custom-op-to-npu-custom-op", "npu-kernelbench-numerics", "ascendc-event-sync"]
    assert analysis.runtime_skills.inject_full is False
    assert analysis.runtime_skills.missing == "error"

    phases = {
        phase["id"]: phase
        for phase in workflow.sub_workflows["operator_repair_loop"].phases
    }
    for phase_id in ("prepare_operator", "fix_operator"):
        config = phases[phase_id]["runtime_skills"]
        assert config["include"] == analysis.runtime_skills.include
        # The fixer gets the full numerical rules; prepare only the summary.
        assert config["inject_full"] is (phase_id == "fix_operator")
        assert config["missing"] == "error"


def test_operator_evaluation_classification_routes_reference_and_device_failures() -> None:
    reference = classify_operator_evaluation(
        {
            "failure_category": "evaluator_failed",
            "evaluation_scope": "reference",
            "evaluation_completeness": "reference_failed",
        }
    )
    assert reference["category"] == "reference_unavailable"
    assert reference["repair_role"] == "operator_environment_fixer"

    device = classify_operator_evaluation(
        {
            "failure_category": "device_memory_fault",
            "evaluation_scope": "candidate",
            "evaluation_completeness": "partial",
            "first_failure_workload": "case-014",
        }
    )
    assert device["repair_role"] == "operator_fixer"
    assert device["requires_single_workload_debug"] is True
    assert device["first_failure_workload"] == "case-014"

    timeout = classify_operator_evaluation(
        {
            "evaluation_status": "timeout",
            "statuses": {"TIMEOUT": 1},
            "executed_count": 1,
            "evaluation_completeness": "partial",
            "first_failure_workload": 0,
            "timeout_workload": {
                "index": 0,
                "status": "TIMEOUT",
                "is_inferred": False,
            },
        }
    )
    assert timeout["category"] == "timeout"
    assert timeout["timeout_kind"] == "first_workload"
    assert timeout["repair_role"] == "timeout_analyzer"
    assert timeout["requires_instrumented_probe"] is True

    ordinary_candidate_log = classify_operator_evaluation(
        {
            "failure_category": "evaluator_failed",
            "evaluation_scope": "unknown",
            "evaluation_completeness": "no_trace",
        }
    )
    assert ordinary_candidate_log["category"] == "evaluator_failed"
    assert ordinary_candidate_log["repair_role"] == "none"


def test_timeout_after_completed_workload_is_partial_trace(tmp_path: Path) -> None:
    classification = classify_operator_evaluation(
        {
            "evaluation_status": "timeout",
            "executed_count": 1,
            "evaluation_completeness": "partial",
            "timeout_workload": {
                "index": 1,
                "status": "NOT_RECORDED_AT_TIMEOUT",
                "is_inferred": True,
            },
        }
    )

    assert classification["timeout_kind"] == "partial_trace"
    assert classification["timeout_workload"]["index"] == 1


def test_timeout_workload_selection_distinguishes_observed_and_inferred(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        '\n'.join([
            json.dumps({"uuid": "case-001", "inputs": {"x": {"shape": [8]}}}),
            json.dumps({"uuid": "case-002", "inputs": {"x": {"shape": [16]}}}),
        ]) + "\n",
        encoding="utf-8",
    )
    inferred = _select_timeout_workload(
        workload_path,
        [{"uuid": "case-001", "evaluation": {"status": "PASSED"}}],
        timed_out=True,
    )
    observed = _select_timeout_workload(
        workload_path,
        [{"uuid": "case-001", "evaluation": {"status": "TIMEOUT"}}],
        timed_out=True,
    )

    assert inferred == {
        "index": 1,
        "uuid": "case-002",
        "inputs": {"x": {"shape": [16], "dtype": None}},
        "status": "NOT_RECORDED_AT_TIMEOUT",
        "selection_basis": "next_unrecorded_workload",
        "is_inferred": True,
        "is_ambiguous": True,
        "trace_status_is_synthetic": False,
    }
    assert observed["index"] == 0
    assert observed["selection_basis"] == "timeout_trace"
    assert observed["is_inferred"] is False


def test_synthetic_timeout_traces_identify_serial_runner_trigger_as_inferred(
    tmp_path: Path,
) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        "\n".join(
            json.dumps({"uuid": f"case-{index:03d}", "inputs": {}})
            for index in range(3)
        ) + "\n",
        encoding="utf-8",
    )
    traces = [
        {"workload": {"uuid": "case-000"}, "evaluation": {"status": "PASSED"}},
        {
            "workload": {"uuid": "case-001"},
            "evaluation": {
                "status": "TIMEOUT",
                "log": "Timeout after 450s (or worker crashed)",
            },
        },
        {
            "workload": {"uuid": "case-002"},
            "evaluation": {
                "status": "TIMEOUT",
                "log": "Timeout after 450s (or worker crashed)",
            },
        },
    ]

    selected = _select_timeout_workload(
        workload_path,
        traces,
        timed_out=False,
        runner_mode="persistent",
        runner_worker_count=1,
    )

    assert selected["index"] == 1
    assert selected["uuid"] == "case-001"
    assert selected["is_inferred"] is True
    assert selected["is_ambiguous"] is False
    assert selected["trace_status_is_synthetic"] is True
    assert selected["selection_basis"] == "first_unreturned_single_worker_batch"
    assert selected["synthetic_timeout_count"] == 2


def test_trace_only_timeout_is_classified_for_timeout_analysis() -> None:
    classification = classify_operator_evaluation(
        {
            "evaluation_status": "completed",
            "failure_category": "partial_trace_timeout",
            "timeout_classification": "runner_workload_timeout",
            "statuses": {"PASSED": 2, "TIMEOUT": 3},
            "executed_count": 5,
            "evaluation_completeness": "complete",
            "timeout_workload": {
                "index": 2,
                "uuid": "case-002",
                "status": "TIMEOUT",
                "is_inferred": True,
            },
        }
    )

    assert classification["category"] == "timeout"
    assert classification["timeout_classification"] == "runner_workload_timeout"
    assert classification["repair_role"] == "timeout_analyzer"
    assert classification["timeout_workload"]["is_inferred"] is True


def test_diagnostic_artifact_context_prioritizes_timeout_profiles_and_stays_valid_json(
    tmp_path: Path,
) -> None:
    evaluation_dir = tmp_path / "evaluation"
    profile_dir = evaluation_dir / "profile"
    work_dir = evaluation_dir / "work"
    (profile_dir / "case-002_ref").mkdir(parents=True)
    (profile_dir / "case-002_sol").mkdir(parents=True)
    work_dir.mkdir(parents=True)
    (evaluation_dir / "process.json").write_text('{"pid": 12}\n', encoding="utf-8")
    (profile_dir / "case-002_ref" / "profiler.json").write_text("{}", encoding="utf-8")
    (profile_dir / "case-002_sol" / "profiler.json").write_text("{}", encoding="utf-8")
    (profile_dir / "case-001_sol").mkdir()
    (profile_dir / "case-001_sol" / "profiler.json").write_text("{}", encoding="utf-8")
    nested_profile = work_dir / "GELU-solution-author" / "profile" / "case-002_sol" / "trace"
    nested_profile.mkdir(parents=True)
    (nested_profile / "kernel_details.csv").write_text("kernel,duration\n", encoding="utf-8")
    (work_dir / "eval.py").write_text("pass\n", encoding="utf-8")

    index = _index_diagnostic_artifacts(evaluation_dir)
    index_path = evaluation_dir / "diagnostic_artifacts.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    context = json.loads(
        format_diagnostic_artifacts_context(
            index_path,
            timeout_workload={"uuid": "case-002"},
            max_bytes=10000,
        )
    )

    target_profiles = [
        item for item in context["files"]
        if item.get("workload_uuid") == "case-002"
    ]
    assert {item["profile_role"] for item in target_profiles} == {
        "candidate",
        "reference",
    }
    assert any("work/GELU-solution-author/profile/case-002_sol" in item["relative_path"] for item in target_profiles)
    assert index["directories"]["profile"]["file_count"] == 4
    assert index["directories"]["work"]["file_count"] == 2
    assert context["files"][0]["relative_path"] == "process.json"
    assert context["index_path"] == str(index_path.resolve())
    assert "case-001_sol" not in json.dumps(target_profiles)


def test_operator_artifact_validation_rejects_flat_descriptor(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    solution = task_dir / "solution.json"
    solution.write_text(json.dumps({
        "operator": "Cat",
        "solution_type": "ascendc",
        "entry_point": "adapter.py::run",
        "sources": [{"path": "adapter.py", "content": "def run(value):\n    return value\n"}],
    }), encoding="utf-8")

    result = validate_operator_artifact(solution, task_dir)

    assert result["candidate_evaluable"] is False
    assert result["solution_ready"] is False
    assert any("requires non-empty name" in error for error in result["errors"])


def test_operator_artifact_validation_detects_materialized_source_mismatch(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    solution = task_dir / "solution.json"
    solution.write_text(
        json.dumps(
            {"sources": [{"path": "adapter.py", "content": "return 1\n"}]}
        ),
        encoding="utf-8",
    )
    (task_dir / "adapter.py").write_text("return 2\n", encoding="utf-8")
    result = validate_operator_artifact(solution, task_dir)
    assert result["solution_ready"] is False
    assert result["writeback_status"] == "mismatch"
    assert result["candidate_manifest"]["writeback_mismatches"] == ["adapter.py"]


def test_operator_artifact_validation_requires_rebuild(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    missing = validate_operator_artifact(task_dir / "solution.json", task_dir)
    assert missing["solution_ready"] is False
    assert missing["artifact_complete"] is False
    assert missing["rebuild_required"] is True

    solution = task_dir / "solution.json"
    solution.write_text(
        json.dumps(
            {"sources": [{"path": "adapter.py", "content": "return 1\n"}]}
        ),
        encoding="utf-8",
    )
    ready = validate_operator_artifact(solution, task_dir)
    assert ready["solution_ready"] is True
    assert ready["artifact_complete"] is True

    stale = validate_operator_artifact(
        solution,
        task_dir,
        baseline_sha256=ready["solution_sha256"],
        require_changed=True,
    )
    assert stale["candidate_evaluable"] is False
    assert stale["stale_candidate"] is True


def test_candidate_manifest_detects_source_writeback_mismatch(tmp_path: Path) -> None:
    solution_dir = tmp_path / "candidate"
    solution_dir.mkdir()
    solution = solution_dir / "solution.json"
    solution.write_text(
        json.dumps(
            {
                "sources": [
                    {"path": "adapter.py", "content": "return 1\n"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (solution_dir / "adapter.py").write_text("return 2\n", encoding="utf-8")

    manifest = build_candidate_manifest(solution)

    assert manifest["writeback_status"] == "mismatch"
    assert manifest["candidate_evaluable"] is False
    assert manifest["writeback_mismatches"] == ["adapter.py"]


def test_failing_workload_digest_keeps_every_failure() -> None:
    from core.operator_benchmark import _failing_workload_digest

    traces = [
        {"workload": {"uuid": f"w{i}", "inputs": {"x": {"shape": [2], "dtype": "float16"}}},
         "evaluation": {"status": "INCORRECT_NUMERICAL", "correctness": {"max_absolute_error": 1.0}}}
        for i in range(30)
    ]
    failing, count = _failing_workload_digest(traces)
    assert count == 30
    assert len(failing) == 30
    assert failing[0]["uuid"] == "w0"


def test_validate_real_gelu_task() -> None:
    task_dir = Path(
        "/Users/yiding/code/ascendc-kernelgen-data-master-npu-kernelbench-"
    ) / "npu-kernelbench/data/kernel_generator/level1/GELU"

    result = validate_operator_task(task_dir)

    assert result["validation_passed"] is True
    assert result["operator_name"] == "GELU"
    assert result["workload_count"] == 50
    assert validate_env(
        {"operator_task_dir": str(task_dir), "solution_type": "ascendc"}
    )["passed"]


def test_non_ascendc_solution_type_is_rejected() -> None:
    task_dir = Path(
        "/Users/yiding/code/ascendc-kernelgen-data-master-npu-kernelbench-"
    ) / "npu-kernelbench/data/kernel_generator/level1/GELU"

    assert validate_env(
        {"operator_task_dir": str(task_dir), "solution_type": "triton"}
    )["passed"] is False


def test_task_without_reference_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "definition.json").write_text(
        json.dumps({"name": "MissingReference"}), encoding="utf-8"
    )
    (tmp_path / "workload.jsonl").write_text(
        json.dumps({"uuid": "case-001"}) + "\n", encoding="utf-8"
    )

    result = validate_operator_task(tmp_path)

    assert result["validation_passed"] is False
    assert "definition.reference must be a non-empty string" in result["errors"]


def test_summarize_traces_preserves_error_metrics(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "evaluation": {
                            "status": "PASSED",
                            "correctness": {
                                "max_absolute_error": 0.001,
                                "max_relative_error": 0.002,
                                "matched_ratio": 1.0,
                            },
                        }
                    }
                ),
                json.dumps(
                    {
                        "evaluation": {
                            "status": "INCORRECT_NUMERICAL",
                            "correctness": {
                                "max_absolute_error": 0.004,
                                "max_relative_error": 0.01,
                                "matched_ratio": 0.98,
                            },
                        }
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_traces(traces_path, operator_name="GELU")

    assert summary["total_workloads"] == 2
    assert summary["passed_workloads"] == 1
    assert summary["workload_pass_rate"] == 0.5
    assert summary["operator_pass_rate"] == 0.0
    assert summary["max_absolute_error"] == 0.004
    assert summary["max_relative_error"] == 0.01
    assert summary["min_matched_ratio"] == 0.98
    assert validate_evaluation(summary)["passed"]


def test_runtime_configuration_is_run_scoped(tmp_path: Path) -> None:
    result = configure_operator_runtime(
        output_dir=tmp_path,
        runtime_report={"python_executable": "/opt/python", "opp_path": "/opt/opp"},
    )

    env_path = Path(result["environment_file"])
    assert result["configuration_status"] == "ready"
    assert env_path.is_file()
    env_text = env_path.read_text(encoding="utf-8")
    assert f"SEAM_OPERATOR_RUN_ROOT={tmp_path}" in env_text
    assert "TMPDIR=" in env_text
    assert "ASCEND_OPP_PATH=/opt/opp" in env_text
    assert (tmp_path / "runtime" / "configuration.json").is_file()


def test_failure_manifest_marks_incomplete_evaluation(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        json.dumps(
            {
                "uuid": "case-001",
                "evaluation": {"status": "RUNTIME_ERROR"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    solution = tmp_path / "solution.json"
    solution.write_text("{}", encoding="utf-8")
    manifest = build_failure_manifest(
        {
            "iteration": "2",
            "evaluation_id": "eval-test",
            "evaluation_status": "timeout",
            "evaluation_valid": False,
            "total_workloads": 1,
            "passed_workloads": 0,
            "statuses": {"RUNTIME_ERROR": 1},
            "trace_complete": False,
        },
        traces_path,
        solution,
    )
    assert manifest["evaluation_valid"] is False
    assert manifest["invalid_reason"] == "timeout"
    assert manifest["failure_groups"][0]["category"] == "runtime_error"
    assert manifest["failure_groups"][0]["representative_workloads"] == ["case-001"]


def test_timeout_evidence_is_classified_without_hiding_the_timeout(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"uuid": "case-001"}) + "\n", encoding="utf-8")
    assert _classify_timeout(traces, "", "", "cleaned") == "partial_trace_timeout"
    assert _classify_timeout(tmp_path / "missing.jsonl", "", "", "cleaned") == "no_trace_timeout"
    assert _classify_timeout(tmp_path / "missing.jsonl", "", "Segmentation fault", "cleaned") == "host_crash"
    assert _classify_timeout(tmp_path / "missing.jsonl", "", "devmm acl error 507035", "cleaned") == "device_memory_fault"
    assert _classify_timeout(tmp_path / "missing.jsonl", "", "No space left on device", "cleaned") == "resource_exhausted"


def test_evaluator_lock_is_device_scoped_when_one_device_is_visible(tmp_path: Path) -> None:
    lock = _evaluator_lock_path(
        tmp_path, {"ASCEND_RT_VISIBLE_DEVICES": "3"}
    )
    assert lock.name == ".seam-kernelbench-evaluator.3.lock"
    assert _evaluator_lock_path(tmp_path, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}).name == ".seam-kernelbench-evaluator.lock"


def test_evaluation_runtime_isolated_from_base_runtime(tmp_path: Path) -> None:
    base_env = tmp_path / "base.env"
    base_env.write_text("ASCEND_OPP_PATH=/opt/opp\nTMPDIR=/shared/tmp\n", encoding="utf-8")
    first = _prepare_evaluation_runtime(tmp_path / "eval-1", base_env)
    second = _prepare_evaluation_runtime(tmp_path / "eval-2", base_env)

    first_text = first.read_text(encoding="utf-8")
    second_text = second.read_text(encoding="utf-8")
    assert "ASCEND_OPP_PATH=/opt/opp" in first_text
    assert "TMPDIR=/shared/tmp" not in first_text
    assert "TMPDIR=" in first_text
    assert first_text != second_text
    assert (tmp_path / "eval-1" / "runtime" / "ownership.json").is_file()
    assert (tmp_path / "eval-2" / "runtime" / "ownership.json").is_file()


def test_kernelbench_repair_prompt_uses_injected_text_not_evidence_paths() -> None:
    prompt = Path("src/prompts/repair_operator_fixer_kernelbench_ascendc.md").read_text(
        encoding="utf-8"
    )

    assert "{operator_definition_text}" in prompt
    assert "{operator_workload_text}" in prompt
    assert "{operator_evaluation_text}" in prompt
    assert "{operator_error_analysis_text}" in prompt
    assert "{operator_diagnostic_artifacts_text}" in prompt
    assert "{operator_process_metadata_text}" in prompt
    assert "{build_artifact_path}" not in prompt
    assert "{latest_trace_path}" not in prompt
    assert "do not issue parallel tool calls" in prompt
    assert "Do not spend tool calls reopening" in prompt
