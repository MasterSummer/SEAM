#!/usr/bin/env bash
# SEAM operator entry: dsh is the only runtime. The legacy OpenCode launcher is
# intentionally not available; this wrapper forwards every argument to the dsh
# entry point so existing drivers keep working unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_operator_dsh.sh" "$@"
