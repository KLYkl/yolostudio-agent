#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-list}"

ALLOWED_PREFIXES=(
  "/data/test_dataset"
  "/data/agent_cap_tests"
  "/data/test_dataset_split_for_yaml"
)

TARGETS=(
  "/data/agent_cap_tests/nonstandard_dataset/pics_split"
  "/data/agent_cap_tests/nonstandard_dataset_split"
  "/data/agent_cap_tests/nonstandard_dataset_split(1)"
  "/data/agent_cap_tests/nonstandard_dataset_split(2)"
  "/data/agent_cap_tests/zyb/images_split"
  "/data/agent_cap_tests/zyb/images_split(1)"
  "/data/agent_cap_tests/zyb/images_split(2)"
  "/data/agent_cap_tests/zyb/images_split(3)"
  "/data/agent_cap_tests/zyb/images_split(4)"
  "/data/agent_cap_tests/zyb/images_split(5)"
  "/data/agent_cap_tests/zyb/images_split(6)"
  "/data/agent_cap_tests/zyb/images_split(7)"
  "/data/agent_cap_tests/zyb/images_split(8)"
  "/data/agent_cap_tests/zyb/images_split(9)"
  "/data/test_dataset/images_split"
  "/data/test_dataset/images_split(1)"
  "/data/test_dataset/images_split(2)"
  "/data/test_dataset/images_split(3)"
  "/data/test_dataset/images_split(4)"
  "/data/test_dataset/images_split(5)"
  "/data/test_dataset/images_split(6)"
  "/data/test_dataset/images_split(7)"
  "/data/test_dataset/images_split(8)"
  "/data/test_dataset/images_split(9)"
  "/data/test_dataset_split_for_yaml"
)

is_allowed_prefix() {
  local resolved="$1"
  local prefix
  for prefix in "${ALLOWED_PREFIXES[@]}"; do
    if [[ "$resolved" == "$prefix" || "$resolved" == "$prefix/"* ]]; then
      return 0
    fi
  done
  return 1
}

validate_target() {
  local target="$1"
  local resolved
  resolved="$(realpath -m -- "$target")"

  if ! is_allowed_prefix "$resolved"; then
    echo "[ABORT] target is outside allowed roots: $target -> $resolved" >&2
    exit 1
  fi

  case "$resolved" in
    *split*|*/images_split|*/pics_split|"/data/test_dataset_split_for_yaml") ;;
    *)
      echo "[ABORT] target does not look like a split artifact: $target -> $resolved" >&2
      exit 1
      ;;
  esac
}

list_targets() {
  local count=0
  local target
  for target in "${TARGETS[@]}"; do
    validate_target "$target"
    if [[ -e "$target" ]]; then
      echo "[EXISTS] $target"
      count=$((count + 1))
    else
      echo "[MISSING] $target"
    fi
  done
  echo "---"
  echo "existing_targets=$count"
}

clean_targets() {
  local count=0
  local target
  for target in "${TARGETS[@]}"; do
    validate_target "$target"
    if [[ -e "$target" ]]; then
      rm -rf -- "$target"
      echo "[REMOVED] $target"
      count=$((count + 1))
    else
      echo "[SKIP] $target"
    fi
  done
  echo "---"
  echo "removed_targets=$count"
}

case "$ACTION" in
  list)
    list_targets
    ;;
  clean)
    clean_targets
    ;;
  *)
    echo "Usage: $0 [list|clean]" >&2
    exit 2
    ;;
esac
