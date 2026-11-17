#!/usr/bin/env bash
# Thin kernel-agent loop: one persistent dsh session, all tools, task-level DDL.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash src/scripts/run_operator_agent.sh <TASK_DIR> <KERNELBENCH_ROOT> \
    --target-soc SOC --device N [options]

Options:
  --operator-output-dir DIR
  --run-id ID
  --provider NAME
  --model NAME
  --base-url URL
  --api-key-env NAME
  --task-budget SECONDS      Task-level deadline (default 7200)
  --profile NAME             dsh profile (default headless)
  --ensure-key               Export a key from /root/.local/share/opencode/auth.json
  -h, --help
USAGE
}

TASK_DIR="${1:-}"; KB_ROOT="${2:-}"; shift 2 2>/dev/null || true
[[ -n "$TASK_DIR" && -n "$KB_ROOT" ]] || { usage >&2; exit 2; }

TARGET_SOC=""; DEVICE=""; OUT=""; RUN_ID=""
PROVIDER="${SEAM_PROVIDER:-chuangzhi}"
MODEL="${SEAM_MODEL:-deepseek-v4-flash-0731}"
BASE_URL="${SEAM_BASE_URL:-https://apicz.boyuerichdata.com/v1}"
API_KEY_ENV="${SEAM_API_KEY_ENV:-CHUANGZHI_KEY}"
TASK_BUDGET="${SEAM_TASK_BUDGET:-7200}"
PROFILE="${SEAM_DSH_PROFILE:-headless}"
ENSURE_KEY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-soc) TARGET_SOC="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --operator-output-dir) OUT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --api-key-env) API_KEY_ENV="$2"; shift 2 ;;
    --task-budget) TASK_BUDGET="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --ensure-key) ENSURE_KEY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$TARGET_SOC" ]] || { echo "--target-soc is required" >&2; exit 2; }
[[ -n "$DEVICE" ]] || { echo "--device is required" >&2; exit 2; }
TASK_DIR="$(cd "$TASK_DIR" && pwd -P)"
KB_ROOT="$(cd "$KB_ROOT" && pwd -P)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TASK_NAME="$(basename "$TASK_DIR")"
RUN_ID="${RUN_ID:-${TASK_NAME,,}-$(date '+%Y%m%d-%H%M%S')}"
OUT="${OUT:-/opt/data/npu_kernelbench_runs/agent-loop/$RUN_ID}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd -P)"
WORKSPACE="$OUT/workspace"
rm -rf "$WORKSPACE"
cp -a "$TASK_DIR" "$WORKSPACE"
rm -f "$WORKSPACE/solution.json"
mkdir -p "$WORKSPACE/candidates" "$WORKSPACE/evaluations" "$WORKSPACE/subagents" "$OUT/tmp"
touch "$WORKSPACE/notes.md" "$WORKSPACE/experiments.md"
cat > "$WORKSPACE/research.md" <<'EOF'
# Research log

Record broad or multi-file reads here. Small targeted reads can stay in notes
and experiments; they do not need an entry.

| Question | Decision this read changes | Target path/symbol | Result | Still open? |
|---|---|---|---|---|
EOF


if [[ "$ENSURE_KEY" == "1" && -z "${!API_KEY_ENV:-}" ]]; then
  export "$API_KEY_ENV=$(python3 -c "import json;print(json.load(open('/root/.local/share/opencode/auth.json'))['deepseek']['key'])")"
fi
if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "API key env $API_KEY_ENV is empty" >&2
  exit 2
fi

PATCH="$OUT/$RUN_ID.patch.yml"
PROMPT="$OUT/$RUN_ID.prompt.txt"
cat > "$PATCH" <<EOF
- id: tool-bash
  disabled: false
- id: tool-fs
  disabled: false
- id: tool-fs-search
  disabled: false
- id: tool-subagent
  disabled: false
- id: llm-pi-ai
  config:
    providers:
      $PROVIDER:
        displayName: $PROVIDER
        api: openai-completions
        baseURL: $BASE_URL
        apiKeyEnv: $API_KEY_ENV
        models:
          - id: $MODEL
            contextWindow: 262144
- id: agent-default-model
  config:
    provider: $PROVIDER
    model: $MODEL
EOF

NOW=$(date +%s)
DEADLINE=$((NOW + TASK_BUDGET))
cat > "$PROMPT" <<EOF
You are a senior Ascend C / KernelBench engineer. Work directly in this workspace.
Do NOT call seam_workflow, do not use the old phase pipeline, and do not wait for a separate fixer.

Task: $TASK_NAME
Level dir: $(dirname "$TASK_DIR")
Target SoC: $TARGET_SOC
NPU: $DEVICE (set ASCEND_RT_VISIBLE_DEVICES=$DEVICE)
Workspace: $WORKSPACE
Official task data: $TASK_DIR
KernelBench root: $KB_ROOT
Task deadline: $DEADLINE (about $((TASK_BUDGET / 60)) minutes from launch)

Tools:
- read / write / edit / glob / grep / bash / subagent
- smoke/full evaluator via bash:
  python $REPO_ROOT/src/scripts/kernel_eval.py smoke \\
    --task $WORKSPACE --solution $WORKSPACE/solution.json \\
    --device $DEVICE --limit 2 --output-dir $WORKSPACE/evaluations \\
    --feedback $WORKSPACE/feedback.jsonl --kernelbench-root $KB_ROOT
  python $REPO_ROOT/src/scripts/kernel_eval.py full \\
    --task $WORKSPACE --solution $WORKSPACE/solution.json \\
    --device $DEVICE --timeout 2400 --output-dir $WORKSPACE/evaluations \\
    --feedback $WORKSPACE/feedback.jsonl --kernelbench-root $KB_ROOT

Rules:
0. Preflight first: run `ls -l /dev/davinci$DEVICE` and a tiny
   `python -c "import torch,torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"`.
   If the device is not visible, stop and report an environment blocker immediately;
   do not spend the budget building a candidate that cannot be evaluated.
1. Read definition.json, workload.jsonl and reference first.
2. Write a valid KernelBench solution.json plus main.py / op_host / op_kernel. Use the native Cat example only as a packaging reference:
   /opt/data/npu_kernelbench_runs/llm-cat/workspace
3. No torch.cat/reference bypass, no CPU fallback, no fake success. Follow docs/operator_constraints.example.json.
4. Evaluation: once the first solution.json exists, build it and run smoke.
   Smoke is diagnostic and may be run any number of times. Full is authoritative;
   run it as soon as smoke passes, and always by 90% of the budget.
5. Research discipline:
   - Small, targeted reads are fine.
   - Do not do large-scale SDK/repo reading: scanning many headers, searching the
     whole source tree, or building a full API catalog.
   - If a task seems to require broad reading, first write one line in research.md:
     `Question: ... | Scope: ... | Decision this should change: ...`, then read
     only the smallest set of files that can answer it.
   - If you notice 3+ consecutive research-only tool calls (read/grep/glob/sed/cat/find),
     treat that as a hint that it may be time to write/build/smoke; use judgment.
   - Prefer reading what the compiler or a failing trace points at.
6. Subagent design:
   - Use the `subagent` tool for narrow read-only questions, one at a time:
     API declaration lookup, shape/dtype audit, independent review, or failure
     reproduction.
   - Give the subagent one precise question and the exact paths to inspect.
     Require it to return <=10 lines: answer, evidence paths/symbols, confidence,
     and what decision it changes. No full logs, no source edits.
   - Save each report under subagents/ and cite it in research.md. The parent
     agent owns all source edits; do not let parallel subagents edit the same files.
   - Use a subagent when a question would otherwise require many header reads;
     do not use it to avoid building or smoke-testing.
7. Read the feedback JSON and the referenced traces/logs. Do not guess from a summary alone.
8. Keep notes.md, experiments.md and research.md current. Snapshot promising candidates under candidates/.
9. At 50% of the budget you should have a buildable candidate. At 75% you should have a smoke pass. At 90% run full or preserve the best candidate. At the deadline, stop exploring and report the best candidate and exact pass counts.
10. Do not modify definition.json or workload.jsonl.

Start now. Work autonomously until the task is solved or the deadline is reached. Do not ask questions.
EOF

echo "Runtime: thin agent loop"
echo "Task: $TASK_NAME"
echo "Workspace: $WORKSPACE"
echo "Output: $OUT"
echo "Model: $PROVIDER/$MODEL"
echo "Deadline: $(date -d "@$DEADLINE" '+%F %T')"
START=$(date +%s)
set +e
(
  cd "$WORKSPACE"
  export TMPDIR="$OUT/tmp" SEAM_AGENT_TMPDIR="$OUT/tmp"
  export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
  # bwrap's fresh /dev hides davinci nodes; this profile is trusted and needs NPU access.
  export DSH_PERMISSION_MODE="${SEAM_DSH_PERMISSION_MODE:-danger-full-access}"
  export NO_PROXY='127.0.0.1,localhost,::1' no_proxy='127.0.0.1,localhost,::1'
  timeout --signal=TERM --kill-after=120s "$TASK_BUDGET" \
    dsh --profile "$PROFILE" --patch "$PATCH" "$(cat "$PROMPT")"
) > "$OUT/agent.log" 2>&1
RC=$?
set -e
DUR=$(( $(date +%s) - START ))
cat > "$OUT/run.json" <<JSON
{"run_id":"$RUN_ID","task":"$TASK_NAME","status":"$([[ $RC -eq 0 ]] && echo completed || echo failed)","exit_code":$RC,"duration_s":$DUR,"workspace":"$WORKSPACE"}
JSON
echo "agent exit=$RC duration=${DUR}s log=$OUT/agent.log"
exit "$RC"
