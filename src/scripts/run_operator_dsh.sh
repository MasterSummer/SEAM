#!/usr/bin/env bash
# Default SEAM operator entry: run one KernelBench operator through the dsh
# native workflow engine (plugins/dsh-seam), not through OpenCode.
#
# The workflow state machine runs inside dsh's workflow worker; dsh role
# sessions do the LLM work; SEAM's Python bridge keeps the domain builtins.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/.." && pwd)"
PLUGIN_DIR="$REPO_ROOT/plugins/dsh-seam"
DEFAULT_POLICY="$REPO_ROOT/docs/operator_constraints.example.json"

TARGET_SOC=""
OUTPUT_DIR=""
POLICY_PATH="$DEFAULT_POLICY"
MAX_ITER=""
RUN_ID=""
PROVIDER="${SEAM_PROVIDER:-local-qwen}"
MODEL="${SEAM_MODEL:-qwen3.8-27b}"
BASE_URL="${SEAM_BASE_URL:-http://127.0.0.1:8002/v1}"
API_KEY_ENV="${SEAM_API_KEY_ENV:-LOCAL_QWEN_KEY}"
PYTHON_BIN="${SEAM_PYTHON:-/home/yiding/npu-kernelbench/.venv/bin/python}"
PHASE_TIMEOUT="${SEAM_PHASE_TIMEOUT:-7200}"
PROFILE="${SEAM_DSH_PROFILE:-headless}"
KEEP_WORKSPACE=0
DEVICE="${SEAM_ASCEND_DEVICE:-}"

usage() {
    cat <<'USAGE'
Usage:
  bash src/scripts/run_operator_dsh.sh <OPERATOR_TASK_DIR> <KERNELBENCH_ROOT> \
    --target-soc SOC [options]

Runs the operator workflow on the dsh native workflow engine (default runtime).

Options:
  --operator-output-dir DIR    Trace/artifact output directory
  --operator-constraints PATH  JSON API policy (default: docs/operator_constraints.example.json)
  --target-soc SOC             Actual NPU SoC, e.g. Ascend910B3 (required)
  --run-id ID                  Stable run id (default: <operator>-<timestamp>)
  --provider NAME              dsh provider route (default: local-qwen)
  --model NAME                 model id (default: qwen3.8-27b)
  --base-url URL               OpenAI-compatible endpoint (default: http://127.0.0.1:8002/v1)
  --python PATH                Python with SEAM + NPU deps
  --phase-timeout SECONDS      Per-phase budget (default: 7200)
  --max-iter N                 Recorded only: repair cap lives in the workflow YAML globals
  --profile NAME               dsh profile (default: headless)
  --device N                   Pin ASCEND_RT_VISIBLE_DEVICES (also SEAM_ASCEND_DEVICE)
  --keep-workspace             Keep the copied workspace on failure
  -h, --help                   Show this help

Environment:
  SEAM_RUNTIME=opencode        Route to the legacy OpenCode launcher instead.
USAGE
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then usage; exit 0; fi
if [[ $# -lt 2 ]]; then usage >&2; exit 2; fi
TASK_DIR="$1"; KERNELBENCH_ROOT="$2"; shift 2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-soc) [[ $# -ge 2 ]] || { echo "--target-soc requires a value" >&2; exit 2; }; TARGET_SOC="$2"; shift 2 ;;
        --operator-output-dir) [[ $# -ge 2 ]] || { echo "--operator-output-dir requires a path" >&2; exit 2; }; OUTPUT_DIR="$2"; shift 2 ;;
        --operator-constraints) [[ $# -ge 2 ]] || { echo "--operator-constraints requires a path" >&2; exit 2; }; POLICY_PATH="$2"; shift 2 ;;
        --run-id) [[ $# -ge 2 ]] || { echo "--run-id requires a value" >&2; exit 2; }; RUN_ID="$2"; shift 2 ;;
        --provider) [[ $# -ge 2 ]] || { echo "--provider requires a value" >&2; exit 2; }; PROVIDER="$2"; shift 2 ;;
        --model) [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 2; }; MODEL="$2"; shift 2 ;;
        --base-url) [[ $# -ge 2 ]] || { echo "--base-url requires a value" >&2; exit 2; }; BASE_URL="$2"; shift 2 ;;
        --python) [[ $# -ge 2 ]] || { echo "--python requires a path" >&2; exit 2; }; PYTHON_BIN="$2"; shift 2 ;;
        --phase-timeout) [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo "--phase-timeout requires a positive integer" >&2; exit 2; }; PHASE_TIMEOUT="$2"; shift 2 ;;
        --max-iter) [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo "--max-iter requires a positive integer" >&2; exit 2; }; MAX_ITER="$2"; shift 2 ;;
        --profile) [[ $# -ge 2 ]] || { echo "--profile requires a value" >&2; exit 2; }; PROFILE="$2"; shift 2 ;;
        --device) [[ $# -ge 2 ]] || { echo "--device requires a value" >&2; exit 2; }; DEVICE="$2"; shift 2 ;;
        --keep-workspace) KEEP_WORKSPACE=1; shift ;;
        --server-url|--agent|--message-timeout|--tool-barrier-timeout)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            echo "note: $1 is an OpenCode-only option and is ignored by the dsh runtime" >&2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$TARGET_SOC" ]] || { echo "--target-soc is required" >&2; exit 2; }
TASK_DIR="$(cd "$TASK_DIR" 2>/dev/null && pwd -P)" || { echo "Task directory not found" >&2; exit 2; }
KERNELBENCH_ROOT="$(cd "$KERNELBENCH_ROOT" 2>/dev/null && pwd -P)" || { echo "KernelBench root not found" >&2; exit 2; }
[[ -f "$TASK_DIR/definition.json" ]] || { echo "Missing $TASK_DIR/definition.json" >&2; exit 2; }
[[ -f "$TASK_DIR/workload.jsonl" ]] || { echo "Missing $TASK_DIR/workload.jsonl" >&2; exit 2; }
[[ -f "$KERNELBENCH_ROOT/pyproject.toml" ]] || { echo "KernelBench root must contain pyproject.toml" >&2; exit 2; }
[[ -f "$POLICY_PATH" ]] || { echo "API policy not found: $POLICY_PATH" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
command -v dsh >/dev/null || { echo "dsh not found in PATH" >&2; exit 2; }
[[ -f "$PLUGIN_DIR/dist/plugin.js" ]] || { echo "dsh plugin not built: $PLUGIN_DIR/dist/plugin.js (run npm ci && npm run link:dsh -- <dsh-source> && npm run build)" >&2; exit 2; }

TASK_NAME="$(basename "$TASK_DIR")"
LEVEL="unknown"
case "$TASK_DIR" in
    */level1/*) LEVEL="level1" ;; */level2/*) LEVEL="level2" ;;
    */level3/*) LEVEL="level3" ;; */level4/*) LEVEL="level4" ;;
esac
[[ -n "$RUN_ID" ]] || RUN_ID="${TASK_NAME,,}-$(date '+%Y%m%d-%H%M%S')"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_ROOT="${SEAM_OPERATOR_OUTPUT_ROOT:-/opt/data/npu_kernelbench_runs/dsh}"
    OUTPUT_DIR="$OUTPUT_ROOT/$TASK_NAME/$(date '+%Y%m%d_%H%M%S')"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"
RUN_ROOT="$OUTPUT_DIR/run-root"
WORKSPACE="$OUTPUT_DIR/workspace"
mkdir -p "$RUN_ROOT"
rm -rf "$WORKSPACE"
cp -a "$TASK_DIR" "$WORKSPACE"
# The launcher owns pack/publish; drop any benchmark-bundled solution so a stale
# candidate cannot be mistaken for this run's output.
rm -f "$WORKSPACE/solution.json"

PATCH="$OUTPUT_DIR/$RUN_ID.patch.yml"
PROMPT_FILE="$OUTPUT_DIR/$RUN_ID.prompt.txt"
cat > "$PATCH" <<EOF
- id: workflow-worker-thread
  disabled: false
- id: tool-bash
  disabled: false
- id: tool-fs
  disabled: false
- id: tool-fs-search
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
- insert:
    - id: seam-workflow
      name: $PLUGIN_DIR/dist/plugin.js
      config:
        repository: $REPO_ROOT
        runRoot: $RUN_ROOT
        python: $PYTHON_BIN
        workflows:
          kernelbench: src/workflows/npu_kernelbench_custom_op.yaml
        defaultTimeoutSeconds: $PHASE_TIMEOUT
        maxPromptBytes: 262144
        roleTools:
          main_engineer: [read, write, edit, glob, grep, bash]
          error_analyzer: [read, glob, grep]
          operator_fixer: [read, write, edit, glob, grep, bash]
          operator_environment_fixer: [read, write, edit, glob, grep, bash]
        roleOptions:
          main_engineer:
            provider: $PROVIDER
            model: $MODEL
          error_analyzer:
            provider: $PROVIDER
            model: $MODEL
          operator_fixer:
            provider: $PROVIDER
            model: $MODEL
          operator_environment_fixer:
            provider: $PROVIDER
            model: $MODEL
EOF

cat > "$PROMPT_FILE" <<EOF
You are driving a SEAM operator workflow through the seam_workflow tool.

Step 1. Call seam_workflow exactly once with:
{"action":"run","runId":"$RUN_ID","workflow":"kernelbench","workspace":"$WORKSPACE","context":{"OPERATOR_TASK_DIR":"$WORKSPACE","OPERATOR_SOLUTION_PATH":"$WORKSPACE/solution.json","OPERATOR_OUTPUT_DIR":"$OUTPUT_DIR","KERNELBENCH_ROOT":"$KERNELBENCH_ROOT","SOLUTION_TYPE":"ascendc","TARGET_SOC":"$TARGET_SOC","OPERATOR_LEVEL":"$LEVEL","OPERATOR_CONSTRAINTS":"$POLICY_PATH"}}

Step 2. Call seam_workflow with {"action":"wait","runId":"$RUN_ID","timeoutSeconds":43200}. It blocks until the workflow reaches a terminal state. If it returns status "running", call wait again. Never start a second run.

Step 3. Print the final status JSON.
EOF

printf 'Runtime: dsh (%s)\nOperator: %s (%s)\nKernelBench: %s\nTarget SoC: %s\nModel: %s/%s\nWorkspace: %s\nOutput: %s\nRun id: %s\n' \
    "$PROFILE" "$TASK_NAME" "$LEVEL" "$KERNELBENCH_ROOT" "$TARGET_SOC" "$PROVIDER" "$MODEL" "$WORKSPACE" "$OUTPUT_DIR" "$RUN_ID"

export SEAM_REPOSITORY="$REPO_ROOT"
export SEAM_DSH_PLUGIN="$PLUGIN_DIR/dist/plugin.js"
export SEAM_RUN_ROOT="$RUN_ROOT"
export SEAM_PYTHON="$PYTHON_BIN"
export SEAM_PROVIDER="$PROVIDER"
export SEAM_MODEL="$MODEL"
export DSH_PERMISSION_MODE="${DSH_PERMISSION_MODE:-workspace-write}"
if [[ -n "$DEVICE" ]]; then export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"; fi
# Keep dsh spill/subprocess dirs and evaluator sockets out of the shared /tmp.
# The sandbox grants os.tmpdir() as a temp area, so pointing TMPDIR here also
# makes role agents write their scratch files under the user workspace.
AGENT_TMP_ROOT="${SEAM_AGENT_TMPDIR:-/home/yiding/tmp}"
AGENT_TMP="$AGENT_TMP_ROOT/$RUN_ID"
mkdir -p "$AGENT_TMP"
export TMPDIR="$AGENT_TMP" TMP="$AGENT_TMP" TEMP="$AGENT_TMP"
export SEAM_OPERATOR_SHORT_TMP_ROOT="$AGENT_TMP/seam-kb"
# Do not let HTTP proxy settings intercept the local OpenAI-compatible endpoint.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
if [[ -z "${!API_KEY_ENV:-}" ]]; then
    export "$API_KEY_ENV=EMPTY"
fi

cd "$REPO_ROOT"
set +e
dsh --profile "$PROFILE" --patch "$PATCH" "$(cat "$PROMPT_FILE")"
DSH_EXIT=$?
set -e

STATE="$RUN_ROOT/$RUN_ID/state.json"
STATUS="unknown"
if [[ -f "$STATE" ]]; then
    STATUS="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status","unknown"))' "$STATE" 2>/dev/null || echo unknown)"
fi
printf 'dsh exit=%s workflow status=%s state=%s\n' "$DSH_EXIT" "$STATUS" "$STATE"
if [[ "$STATUS" == "completed" ]]; then
    rm -rf -- "$AGENT_TMP"
else
    printf 'temp kept for diagnosis: %s\n' "$AGENT_TMP" >&2
fi

if [[ "$STATUS" != "completed" && "$KEEP_WORKSPACE" -ne 1 ]]; then
    echo "workspace kept for diagnosis: $WORKSPACE" >&2
fi
[[ "$STATUS" == "completed" ]] || exit 1
