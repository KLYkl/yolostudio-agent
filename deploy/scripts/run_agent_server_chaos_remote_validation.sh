#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "bash" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi

detect_app_root() {
  local candidates=(
    "$SCRIPT_DIR/../.."
    "$HOME/yolostudio_agent_proto"
    "/opt/yolostudio-agent"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate/agent_plan/agent/tests" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "$HOME/yolostudio_agent_proto"
}

detect_conda_root() {
  local candidates=(
    "$HOME/miniconda3"
    "/opt/conda"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "/opt/conda"
}

APP_ROOT="${APP_ROOT:-$(detect_app_root)}"
CONDA_ROOT="${CONDA_ROOT:-$(detect_conda_root)}"
REQUESTED_ENV="${1:-auto}"
OUTPUT_ROOT="${2:-/tmp/agent_server_chaos_output}"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "missing conda.sh under: $CONDA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$APP_ROOT" ]]; then
  echo "missing app root: $APP_ROOT" >&2
  exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"

resolve_env() {
  local requested="$1"
  local env_names
  env_names="$(conda env list | awk 'NF && $1 !~ /^#/ {print $1}')"

  if [[ -n "$requested" && "$requested" != "auto" ]] && grep -qx "$requested" <<<"$env_names"; then
    printf '%s\n' "$requested"
    return 0
  fi

  for candidate in yolostudio-agent-server agent-server yolodo yolo; do
    if grep -qx "$candidate" <<<"$env_names"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -n "$requested" && "$requested" != "auto" ]]; then
    echo "requested conda env not found: $requested" >&2
  fi
  echo "no usable conda env found (tried: yolostudio-agent-server, agent-server, yolodo, yolo)" >&2
  return 1
}

ENV_NAME="$(resolve_env "$REQUESTED_ENV")"
echo "using conda env: $ENV_NAME"
conda activate "$ENV_NAME"

mkdir -p "$OUTPUT_ROOT"
cd "$APP_ROOT"
export PYTHONPATH="$APP_ROOT:$APP_ROOT/agent_plan${PYTHONPATH:+:$PYTHONPATH}"

summary_json="$OUTPUT_ROOT/agent_server_chaos_summary.json"
fail_log="$OUTPUT_ROOT/agent_server_chaos_failures.log"
: > "$fail_log"

test_names=(
  test_agent_server_chaos_p0.py
  test_agent_server_chaos_p1_compare_resilience.py
  test_agent_server_chaos_p1_confirmation_matrix.py
  test_agent_server_chaos_p1_crossmainline.py
  test_agent_server_chaos_p1_followup.py
  test_agent_server_chaos_p1_input_matrix.py
  test_agent_server_chaos_p1_recovery.py
  test_agent_server_chaos_p1_repeat_tolerance.py
  test_agent_server_chaos_p1_replanning.py
  test_agent_server_chaos_p1_revision_matrix.py
  test_agent_server_chaos_p1_running_matrix.py
  test_agent_server_chaos_p2_context.py
  test_agent_server_chaos_p2_crossmainline_extra.py
  test_agent_server_chaos_p2_guardrail_misc.py
  test_agent_server_chaos_p2_integrity.py
  test_agent_server_chaos_p2_mismatch_matrix.py
)

test_files=()
for test_name in "${test_names[@]}"; do
  test_path="$APP_ROOT/agent_plan/agent/tests/$test_name"
  if [[ ! -f "$test_path" ]]; then
    echo "missing chaos test: $test_path" >&2
    exit 1
  fi
  test_files+=("$test_path")
done

passed=0
failed=0
printf '{\n  "ok": true,\n  "tests": [\n' > "$summary_json"
first=1
for test_file in "${test_files[@]}"; do
  test_name="$(basename "$test_file")"
  log_path="$OUTPUT_ROOT/${test_name%.py}.log"
  echo "=== running $test_name ==="
  if python "$test_file" >"$log_path" 2>&1; then
    status="passed"
    passed=$((passed + 1))
  else
    status="failed"
    failed=$((failed + 1))
    {
      echo "### $test_name"
      cat "$log_path"
      echo
    } >> "$fail_log"
  fi
  if [[ $first -eq 0 ]]; then
    printf ',\n' >> "$summary_json"
  fi
  first=0
  python - <<PY >> "$summary_json"
import json
print(json.dumps({
    "name": "$test_name",
    "status": "$status",
    "log_path": "$log_path",
}, ensure_ascii=False), end='')
PY
done
printf '\n  ],\n  "passed": %d,\n  "failed": %d,\n  "failure_log": "%s",\n  "all_passed": %s\n}\n' "$passed" "$failed" "$fail_log" "$([[ $failed -eq 0 ]] && echo true || echo false)" >> "$summary_json"
cat "$summary_json"

if [[ $failed -ne 0 ]]; then
  exit 1
fi
