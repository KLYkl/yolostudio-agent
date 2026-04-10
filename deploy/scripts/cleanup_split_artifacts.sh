#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-list}"

ALLOWED_PREFIXES=(
  "/home/kly/test_dataset"
  "/home/kly/agent_cap_tests"
  "/home/kly/test_dataset_split_for_yaml"
)

TARGETS=(
  "/home/kly/agent_cap_tests/nonstandard_dataset/pics_split"
  "/home/kly/agent_cap_tests/nonstandard_dataset_split"
  "/home/kly/agent_cap_tests/nonstandard_dataset_split(1)"
  "/home/kly/agent_cap_tests/nonstandard_dataset_split(2)"
  "/home/kly/agent_cap_tests/zyb/images_split"
  "/home/kly/agent_cap_tests/zyb/images_split(1)"
  "/home/kly/agent_cap_tests/zyb/images_split(2)"
  "/home/kly/agent_cap_tests/zyb/images_split(3)"
  "/home/kly/agent_cap_tests/zyb/images_split(4)"
  "/home/kly/agent_cap_tests/zyb/images_split(5)"
  "/home/kly/agent_cap_tests/zyb/images_split(6)"
  "/home/kly/agent_cap_tests/zyb/images_split(7)"
  "/home/kly/agent_cap_tests/zyb/images_split(8)"
  "/home/kly/agent_cap_tests/zyb/images_split(9)"
  "/home/kly/test_dataset/images_split"
  "/home/kly/test_dataset/images_split(1)"
  "/home/kly/test_dataset/images_split(2)"
  "/home/kly/test_dataset/images_split(3)"
  "/home/kly/test_dataset/images_split(4)"
  "/home/kly/test_dataset/images_split(5)"
  "/home/kly/test_dataset/images_split(6)"
  "/home/kly/test_dataset/images_split(7)"
  "/home/kly/test_dataset/images_split(8)"
  "/home/kly/test_dataset/images_split(9)"
  "/home/kly/test_dataset_split_for_yaml"
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
    *split*|*/images_split|*/pics_split|"/home/kly/test_dataset_split_for_yaml") ;;
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
