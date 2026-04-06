#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_api.sh [cpu|cuda] [port]

Environment overrides:
  CS336_TRAIN_DATA_META_PATH   Path to train.meta.json
  CS336_VOCAB_SIZE             Tokenizer vocabulary size
  CS336_API_DB_PATH            SQLite database path
  CS336_CONTEXT_LENGTH         Training context length
  CS336_DEVICE                 Training device (cpu or cuda)
  CS336_MIXED_PRECISION        off, bf16, or fp16
  CS336_ACTIVATION_CHECKPOINTING  1 to enable, 0 to disable
  CS336_PRELOAD_DATASET        1 to load the tokenized corpus into RAM on startup, 0 to lazy-load per request
  CS336_MAX_STEPS_CAP          Optional step cap for smoke tests
  CS336_API_ACCEPT_ALL_KEYS    Accept any non-empty API key (default: 1)
  CS336_API_KEYS               Comma-separated allowlist when accept-all is disabled

Examples:
  ./scripts/start_api.sh
  ./scripts/start_api.sh cpu
  ./scripts/start_api.sh cuda
  CS336_MAX_STEPS_CAP=2 ./scripts/start_api.sh cuda 8000
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DEVICE_ARG="${1:-}"
PORT="${2:-${PORT:-8000}}"

if [[ -n "${DEVICE_ARG}" ]]; then
  case "${DEVICE_ARG}" in
    cpu|cuda)
      export CS336_DEVICE="${DEVICE_ARG}"
      ;;
    *)
      echo "Invalid device: ${DEVICE_ARG}" >&2
      usage >&2
      exit 1
      ;;
  esac
fi

export CS336_TRAIN_DATA_META_PATH="${CS336_TRAIN_DATA_META_PATH:-/root/autodl-tmp/tokids/fineweb_edu_full/train.meta.json}"
export CS336_VOCAB_SIZE="${CS336_VOCAB_SIZE:-32000}"
export CS336_API_DB_PATH="${CS336_API_DB_PATH:-/root/autodl-tmp/api/api.db}"
export CS336_CONTEXT_LENGTH="${CS336_CONTEXT_LENGTH:-512}"
if [[ -z "${CS336_DEVICE:-}" ]]; then
  export CS336_DEVICE="$(
    uv run python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo cpu
  )"
fi
if [[ -z "${CS336_MIXED_PRECISION:-}" ]]; then
  if [[ "${CS336_DEVICE}" == cuda* ]]; then
    export CS336_MIXED_PRECISION="bf16"
  else
    export CS336_MIXED_PRECISION="off"
  fi
fi
if [[ -z "${CS336_ACTIVATION_CHECKPOINTING:-}" ]]; then
  if [[ "${CS336_DEVICE}" == cuda* ]]; then
    export CS336_ACTIVATION_CHECKPOINTING="1"
  else
    export CS336_ACTIVATION_CHECKPOINTING="0"
  fi
fi
export CS336_PRELOAD_DATASET="${CS336_PRELOAD_DATASET:-1}"
export CS336_API_ACCEPT_ALL_KEYS="${CS336_API_ACCEPT_ALL_KEYS:-1}"

if [[ ! -f "${CS336_TRAIN_DATA_META_PATH}" ]]; then
  echo "Missing tokenized training metadata: ${CS336_TRAIN_DATA_META_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${CS336_API_DB_PATH}")"

echo "[start_api] train_data_meta_path=${CS336_TRAIN_DATA_META_PATH}"
echo "[start_api] vocab_size=${CS336_VOCAB_SIZE}"
echo "[start_api] api_db_path=${CS336_API_DB_PATH}"
echo "[start_api] context_length=${CS336_CONTEXT_LENGTH}"
echo "[start_api] device=${CS336_DEVICE}"
echo "[start_api] mixed_precision=${CS336_MIXED_PRECISION}"
echo "[start_api] activation_checkpointing=${CS336_ACTIVATION_CHECKPOINTING}"
echo "[start_api] preload_dataset=${CS336_PRELOAD_DATASET}"
if [[ -n "${CS336_MAX_STEPS_CAP:-}" ]]; then
  echo "[start_api] max_steps_cap=${CS336_MAX_STEPS_CAP}"
fi
echo "[start_api] host=0.0.0.0 port=${PORT}"

exec uv run python -m cs336_scaling.api_server --host 0.0.0.0 --port "${PORT}"
