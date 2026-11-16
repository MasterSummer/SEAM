# Operator Benchmark Failure Analysis

Analyze the latest full-workload KernelBench result for one Ascend C operator.
The following operator API policy is binding for any proposed repair:

{operator_constraints}

Any `torch.*` or `torch_npu.*` API not explicitly listed is forbidden. If the
latest failure was caused by such an API, classify it as `api_policy_violation`
and recommend replacing the approach rather than adding the API to the source.
Base the diagnosis on the saved traces, summary, build output, source evidence,
and evaluator artifacts listed in `diagnostic_artifacts.json`. You may also run
a focused diagnostic probe when it can distinguish concrete hypotheses.

## Evidence Integrity and Causal Diagnosis

Keep each diagnosis tied to the current iteration and candidate. From the saved
artifacts record, when available: run/iteration identifier, candidate identity
or source hash, executed command, exit status, first failing stage, workload ID,
source location and log/trace path. Mark absent or truncated evidence explicitly;
never invent provenance or treat a log tail as a complete compiler diagnostic.
Read the relevant original artifact when the supplied excerpt is insufficient.
Do not combine an older candidate's success with the current candidate's failure.

Compiler errors require inspection of the failing expression, declarations and
active API signature, not keyword matching or predetermined source replacement.
For runtime failures distinguish an observed device fault from a suspected source
cause. A timeout, zero traces or failed synchronization alone does not identify
whether the candidate, runtime, evaluator or API transport caused the failure.
Separate confirmed observations, hypotheses and missing evidence; specify the
smallest next check that could discriminate between plausible causes.

Treat the reference as the comparison baseline, not as proof of semantics. Only
report a suspected reference defect with a concrete contract/source/workload
conflict. Correlated failures, parameter patterns or TODOs alone are insufficient.
Do not change benchmark inputs, the reference or acceptance criteria.

Runtime detection report: {runtime_report}
Runtime configuration result: {runtime_configuration}
Latest evaluator result: {evaluation_result}
Deterministic evaluation classification: {evaluation_classification}
Diagnostic artifact index path: {diagnostic_artifacts_path}

Process metadata (`process.json`):
```json
{operator_process_metadata_text}
```

Indexed evaluator/profile/work files (the complete index remains available at
`diagnostic_artifacts_path`):
```json
{operator_diagnostic_artifacts_text}
```

## Optional Smoke/Probe

You may use `bash` to inspect indexed artifacts and, when useful, design and run
a short smoke/probe yourself. This is optional: skip it when the saved evidence
already distinguishes the cause, the probe is unsafe or too expensive, or the
needed runtime/device is unavailable. Choose the smallest probe that can
separate specific hypotheses; do not use keyword searches as a substitute for
causal analysis.

Keep any new probe script or output in `$TMPDIR` (or another disposable
diagnostic location). Do not edit candidate sources, `solution.json`, the
reference, workload inputs, evaluator code/configuration, or acceptance rules.
Do not run the full workload evaluation in this analysis phase. A smoke result
is diagnostic evidence only, never acceptance; after a repair, the workflow's
full-workload evaluation remains the acceptance gate.

For every probe you run, record the exact command, relevant environment and
workload, exit status, and the useful stdout/stderr or artifact paths in
`evidence`. State what the result supports and what it does not establish. If
you do not run a probe, continue from the saved evidence and explain in
`next_verification` what focused probe would best resolve any remaining
uncertainty.

## Timeout Analysis and Repair Routing

If `evaluation_classification.repair_role` is `timeout_analyzer`, this is a
timeout-investigation turn before repair, not a terminal role. Use the supplied
index path and curated listing to query actual files under `profile/` and
`work/`; do not assume that a profile exists just because the directory exists.
Use `process.json`, traces, summary, stdout/stderr, the frozen evaluated
candidate, and available reference/candidate profiles together. The listing
prioritizes the timed-out workload; inspect the full index or its directory when
`omitted_from_prompt` is nonzero.

For a timeout:

- Read `timeout_workload.selection_basis`, `is_inferred`, and
  `trace_status_is_synthetic`. A `TIMEOUT` trace whose log says
  `(or worker crashed)` is a runner-generated placeholder, not a direct
  observation of the kernel that hung. With the recorded persistent runner and
  one worker, the first such trace is the inferred in-flight workload; later
  TIMEOUT traces are unreturned batch items. For parallel workers, treat the
  listed `candidate_workloads` as alternatives, not as confirmed causes.
- The profiler may be nested under `work/<solution>/profile/<uuid>_ref` or
  `.../<uuid>_sol`, not only under the top-level `profile/`. Use the indexed
  `profile_role` and `workload_uuid` to query both sides for the same case.
  Record the timeout workload, its selection basis, previous completed cases,
  trace status/log, relevant profile paths and the first stage supported by
  actual files. Do not relabel the first ordinary failed workload as the
  timed-out workload.
- Compare reference and candidate evidence for the same workload where both
  profiles exist. Inspect the relevant kernel/source location and build/runtime
  artifacts before assigning causality.
- Distinguish candidate code defects from environment/evaluator defects using
  concrete logs, profile events, resource snapshots, process metadata, and
  source evidence. A timeout, missing trace, or missing profile alone does not
  prove either cause.
- If evidence supports a specific candidate source/build defect, set
  `repair_role: "operator_fixer"` and give a causal, location-specific fix. If it
  supports an execution environment, dependency, staging, resource, or evaluator
  lifecycle defect, set `repair_role: "operator_environment_fixer"` and name the
  supporting artifacts. These roles trigger the corresponding repair phase and
  a full-workload rerun; do not stop after merely labeling a timeout.
- If evidence still does not distinguish the causes, set `repair_role: "none"`,
  list the missing artifact or observation, and request the smallest focused
  profiling/probe run in `next_verification`. Do not make an ungrounded repair.
- Do not assume a missing solution profile proves that the candidate did not
  return; verify how the active runner records profiles and correlate with
  traces and host-side logs.

Use `failure_category`, `timeout_classification`, `resource_before`, and
`resource_after` as observations from the framework, not infallible root causes.
A device fault or hang may originate in candidate code; source evidence is needed
before assigning a specific kernel defect. Likewise, a timeout is not automatically
an environment failure.

Classify `category` as `task_schema`, `build`, `link`, `install`, `adapter`,
`runtime`, `numerical`, `api_policy_violation`, `fallback`, `environment`, or
`unknown`, and choose the role from the evidence:

- `operator_fixer`: a candidate source, adapter, build configuration or API-policy
  defect is supported by a source location and causal explanation.
- `operator_environment_fixer`: execution paths, dependencies, staging, resource
  ownership or evaluator lifecycle are supported as the problem. Do not prescribe
  killing unrelated processes or changing global/vendor installations.
- `none`: evidence cannot yet establish a repair, or an external task/reference
  issue requires escalation. State the missing evidence and next action rather
  than requesting speculative source edits.

Incomplete traces, stale aggregates or a missing staging file make evaluation
incomplete; they do not by themselves identify who caused it. Keep
`evaluation_valid` distinct from `operator_code_evidence`: an incomplete run can
still contain direct compiler/source evidence. Use `requires_clean_rerun` when
current evidence cannot establish full-workload correctness. In `handoff`, explain
any disagreement with the deterministic classification and the supporting evidence.
Do not change `definition.json` or `workload.jsonl`.

Return exactly one JSON object:

```json
{
  "category": "unknown",
  "root_cause": "",
  "repair_role": "operator_fixer|operator_environment_fixer|none",
  "evaluation_valid": false,
  "requires_clean_rerun": false,
  "operator_code_evidence": false,
  "suggested_fix": "",
  "failed_workloads": [],
  "evidence": [],
  "missing_evidence": [],
  "hypotheses": [],
  "next_verification": [],
  "handoff": ""
}
```
