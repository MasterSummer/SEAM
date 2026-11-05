import type { Data } from './types.js'

/**
 * Bounded, semantic projection of one loop iteration for the loop history.
 *
 * The history is replayed into every later prompt as runtime evidence, so it
 * must stay small and must never carry candidate source bodies: sources, diffs
 * and traces live in the workspace and are referenced by path. Only the fields
 * that later repair rounds actually reason about are kept, and every truncation
 * is marked explicitly so evidence is never silently altered.
 */
export const HISTORY_TEXT_LIMIT = 400

export function historyText(value: unknown, limit = HISTORY_TEXT_LIMIT): unknown {
  if (typeof value !== 'string') return value
  return value.length <= limit ? value : `${value.slice(0, limit)}…[${value.length} chars omitted]`
}

function pick(source: unknown, keys: readonly string[]): Data {
  const record = source && typeof source === 'object' && !Array.isArray(source)
    ? source as Data
    : {}
  const out: Data = {}
  for (const key of keys) {
    const value = record[key]
    if (value !== undefined) out[key] = value
  }
  return out
}

const EVALUATION_FIELDS = [
  'operator_passed', 'evaluation_status', 'failure_category', 'timeout_classification',
  'operator_pass_rate', 'workload_pass_rate', 'total_workloads', 'passed_workloads',
  'first_failure_workload', 'cascade_failure_count', 'evaluation_completeness',
  'evaluation_scope', 'visible_device', 'cleanup_status',
  'traces_path', 'summary_path', 'failure_manifest_path', 'candidate_manifest_path',
] as const
const PREPARE_FIELDS = [
  'prepared', 'build_status', 'blocking_category', 'source_dir', 'solution_path',
  'manifest_path', 'build_config_path',
] as const

/** Summarize one iteration's step outputs into the record later rounds see. */
export function historyRecord(iteration: number, status: string, outputs: Data): Data {
  const evaluation = outputs.evaluate_operator ?? outputs.operator_kernelbench_eval ?? {}
  const classification = outputs.evaluation_classification ?? outputs.classify_operator_evaluation ?? {}
  const analysis = outputs.error_analysis ?? outputs.analyze_operator_failure ?? {}
  const fix = outputs.fix_operator ?? outputs.fix_operator_environment ?? {}
  const prepare = outputs.prepare_operator ?? outputs.operator_preparation ?? {}
  const record: Data = {
    iteration,
    status,
    evaluation: pick(evaluation, EVALUATION_FIELDS),
    classification: pick(classification, ['category', 'repair_role']),
  }
  const analysisRecord = pick(analysis, [
    'category', 'repair_role', 'evaluation_valid', 'operator_code_evidence',
    'requires_clean_rerun', 'requires_single_workload_debug',
  ])
  for (const key of ['root_cause', 'suggested_fix']) {
    const value = (analysis as Data)[key]
    if (value !== undefined) analysisRecord[key] = historyText(value)
  }
  if (Object.keys(analysisRecord).length) record.error_analysis = analysisRecord
  const fixRecord = pick(fix, ['modified_files', 'configuration_status', 'phase_id'])
  const fixSummary = (fix as Data).summary
  if (fixSummary !== undefined) fixRecord.summary = historyText(fixSummary)
  if (Object.keys(fixRecord).length) record.fix = fixRecord
  const prepareRecord = pick(prepare, PREPARE_FIELDS)
  if (Object.keys(prepareRecord).length) record.prepare = prepareRecord
  return record
}
