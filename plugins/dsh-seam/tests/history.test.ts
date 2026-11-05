import assert from 'node:assert/strict'
import { test } from 'node:test'
import { historyRecord, historyText } from '../src/history.js'

test('loop history keeps statuses, categories and paths but never candidate source', () => {
  const record = historyRecord(2, 'failure', {
    evaluate_operator: {
      operator_passed: false,
      evaluation_status: 'completed',
      failure_category: 'compile_failed',
      operator_pass_rate: 0,
      traces_path: '/runs/1/traces.jsonl',
      candidate_source_blob: 'x'.repeat(5000),
    },
    evaluation_classification: { category: 'compile_failed', repair_role: 'operator_fixer' },
    error_analysis: {
      category: 'compile',
      repair_role: 'operator_fixer',
      root_cause: 'main.py:6 uses a forbidden namespace. '.repeat(50),
      suggested_fix: 'rename the custom op namespace',
      evaluation_valid: true,
    },
    prepare_operator: {
      prepared: true,
      build_status: 'ready',
      source_dir: '/task/sources',
      api_evidence: 'y'.repeat(3000),
    },
    fix_operator: { modified_files: ['sources/main.py'], summary: 'changed namespace' },
  })
  const body = JSON.stringify(record)
  assert.equal((record.evaluation as Record<string, unknown>).operator_passed, false)
  assert.equal((record.evaluation as Record<string, unknown>).traces_path, '/runs/1/traces.jsonl')
  assert.equal((record.classification as Record<string, unknown>).category, 'compile_failed')
  assert.deepEqual((record.fix as Record<string, unknown>).modified_files, ['sources/main.py'])
  assert.ok(!body.includes('candidate_source_blob'))
  assert.ok(!body.includes('api_evidence'))
  assert.ok(!body.includes('x'.repeat(500)), 'source body must not be copied into history')
  assert.ok(body.includes('chars omitted'), 'truncation must be explicit')
  assert.ok(body.length < 2000, `history record must stay small, got ${body.length}: ${body}`)
})

test('historyText marks truncation with the original length', () => {
  const text = historyText('a'.repeat(1000), 100) as string
  assert.ok(text.startsWith('a'.repeat(100)))
  assert.ok(text.includes('1000 chars omitted'))
  assert.equal(historyText('short'), 'short')
})
