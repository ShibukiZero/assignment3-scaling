#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_managed_api_experiment.sh [--device cpu|cuda] [--port PORT] <grid_json> [<grid_json> ...]

Legacy single-grid form (still supported):
  ./scripts/run_managed_api_experiment.sh <grid_json> [cpu|cuda] [port]

Environment overrides:
  API_STARTUP_TIMEOUT_S   Seconds to wait for the API to become ready (default: 300)
  API_READY_POLL_S        Seconds between readiness checks (default: 2)
  MANAGED_POWEROFF_ON_SUCCESS  1 to power off the host after a successful run (default: 0)
  CS336_LOG_DIR           Directory for daemon logs and pid files

Examples:
  ./scripts/run_managed_api_experiment.sh artifacts/experiments/ch3/3_1_1_shape_lr_sweep/grid.json
  ./scripts/run_managed_api_experiment.sh artifacts/experiments/ch3/3_1_1_shape_lr_sweep/grid.json cuda
  ./scripts/run_managed_api_experiment.sh artifacts/experiments/ch3/3_1_1_shape_lr_sweep/grid.json cuda 8000
  ./scripts/run_managed_api_experiment.sh --device cuda --port 8000 \
    artifacts/experiments/ch3/3_3_1_isoflops_3e15_full/grid.json \
    artifacts/experiments/ch3/3_3_2_isoflops_6e15_full/grid.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DEVICE_ARG="cuda"
PORT="8000"
GRID_JSONS=()

if [[ $# -ge 1 && "${1:-}" != --* ]]; then
  if [[ $# -eq 1 ]]; then
    GRID_JSONS=("$1")
  elif [[ $# -eq 2 && ( "${2:-}" == "cpu" || "${2:-}" == "cuda" ) ]]; then
    GRID_JSONS=("$1")
    DEVICE_ARG="$2"
  elif [[ $# -eq 3 && ( "${2:-}" == "cpu" || "${2:-}" == "cuda" ) ]]; then
    GRID_JSONS=("$1")
    DEVICE_ARG="$2"
    PORT="$3"
  else
    GRID_JSONS=("$@")
  fi
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --device)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --device" >&2
          exit 1
        fi
        DEVICE_ARG="$2"
        shift 2
        ;;
      --port)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --port" >&2
          exit 1
        fi
        PORT="$2"
        shift 2
        ;;
      --*)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
      *)
        GRID_JSONS+=("$1")
        shift
        ;;
    esac
  done
fi

if [[ ${#GRID_JSONS[@]} -lt 1 ]]; then
  usage >&2
  exit 1
fi

STARTUP_TIMEOUT_S="${API_STARTUP_TIMEOUT_S:-300}"
READY_POLL_S="${API_READY_POLL_S:-2}"
POWEROFF_ON_SUCCESS="${MANAGED_POWEROFF_ON_SUCCESS:-0}"
BASE_URL="http://127.0.0.1:${PORT}"

for GRID_JSON in "${GRID_JSONS[@]}"; do
  if [[ ! -f "${GRID_JSON}" ]]; then
    echo "[managed_run] missing grid file: ${GRID_JSON}" >&2
    exit 1
  fi
done

FIRST_EXPERIMENT_DIR="$(cd "$(dirname "${GRID_JSONS[0]}")" && pwd)"
MANAGED_LOG_PATH="${FIRST_EXPERIMENT_DIR}/managed_run.log"

export CS336_LOG_DIR="${CS336_LOG_DIR:-${FIRST_EXPERIMENT_DIR}/service_logs}"
mkdir -p "${CS336_LOG_DIR}"

started_server=0
cleanup_done=0

stop_api_if_needed() {
  if [[ "${started_server}" == "1" && "${cleanup_done}" == "0" ]]; then
    echo "[managed_run] stopping local API on port ${PORT}" | tee -a "${MANAGED_LOG_PATH}"
    ./scripts/stop_api.sh "${PORT}" | tee -a "${MANAGED_LOG_PATH}"
    cleanup_done=1
    started_server=0
  fi
}

poweroff_host() {
  echo "[managed_run] powering off host after successful managed run" | tee -a "${MANAGED_LOG_PATH}"
  sync || true
  if command -v shutdown >/dev/null 2>&1; then
    nohup sh -c 'sleep 2; shutdown -h now' >/dev/null 2>&1 &
    return
  fi
  if command -v poweroff >/dev/null 2>&1; then
    nohup sh -c 'sleep 2; poweroff' >/dev/null 2>&1 &
    return
  fi
  echo "[managed_run] no shutdown command available; host poweroff skipped" | tee -a "${MANAGED_LOG_PATH}" >&2
}

cleanup() {
  stop_api_if_needed
}

trap cleanup EXIT

echo "[managed_run] num_grids=${#GRID_JSONS[@]}" | tee "${MANAGED_LOG_PATH}"
for GRID_JSON in "${GRID_JSONS[@]}"; do
  echo "[managed_run] grid_json=${GRID_JSON}" | tee -a "${MANAGED_LOG_PATH}"
done
echo "[managed_run] base_url=${BASE_URL}" | tee -a "${MANAGED_LOG_PATH}"
echo "[managed_run] device=${DEVICE_ARG}" | tee -a "${MANAGED_LOG_PATH}"
echo "[managed_run] startup_timeout_s=${STARTUP_TIMEOUT_S}" | tee -a "${MANAGED_LOG_PATH}"
echo "[managed_run] ready_poll_s=${READY_POLL_S}" | tee -a "${MANAGED_LOG_PATH}"
echo "[managed_run] poweroff_on_success=${POWEROFF_ON_SUCCESS}" | tee -a "${MANAGED_LOG_PATH}"
echo "[managed_run] service_log_dir=${CS336_LOG_DIR}" | tee -a "${MANAGED_LOG_PATH}"

echo "[managed_run] starting local API" | tee -a "${MANAGED_LOG_PATH}"
./scripts/start_api.sh --daemon "${DEVICE_ARG}" "${PORT}" | tee -a "${MANAGED_LOG_PATH}"
started_server=1

echo "[managed_run] waiting for API readiness at ${BASE_URL}" | tee -a "${MANAGED_LOG_PATH}"
ready=0
deadline=$((SECONDS + STARTUP_TIMEOUT_S))
while (( SECONDS < deadline )); do
  if curl -fsS "${BASE_URL}/openapi.json" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep "${READY_POLL_S}"
done

if [[ "${ready}" != "1" ]]; then
  echo "[managed_run] API did not become ready before timeout" | tee -a "${MANAGED_LOG_PATH}" >&2
  exit 1
fi

echo "[managed_run] API is ready, launching experiment grid batch" | tee -a "${MANAGED_LOG_PATH}"
for GRID_JSON in "${GRID_JSONS[@]}"; do
  EXPERIMENT_DIR="$(cd "$(dirname "${GRID_JSON}")" && pwd)"
  EXPERIMENT_ID="$(basename "${EXPERIMENT_DIR}")"
  RESULTS_PATH="${EXPERIMENT_DIR}/results.json"

  echo "[managed_run] starting experiment_id=${EXPERIMENT_ID}" | tee -a "${MANAGED_LOG_PATH}"
  rm -f "${RESULTS_PATH}"
  uv run python scripts/run_api_experiment_grid.py \
    "${GRID_JSON}" \
    --base-url "${BASE_URL}" \
    --output-dir "${EXPERIMENT_DIR}" | tee -a "${MANAGED_LOG_PATH}"

  if [[ ! -s "${RESULTS_PATH}" ]]; then
    echo "[managed_run] missing or empty results file: ${RESULTS_PATH}" | tee -a "${MANAGED_LOG_PATH}" >&2
    exit 1
  fi

  if ! grep -q "\"experiment_id\": \"${EXPERIMENT_ID}\"" "${RESULTS_PATH}"; then
    echo "[managed_run] results file does not contain the expected experiment_id" | tee -a "${MANAGED_LOG_PATH}" >&2
    exit 1
  fi

  if ! grep -q "\"responses\"" "${RESULTS_PATH}"; then
    echo "[managed_run] results file does not contain responses" | tee -a "${MANAGED_LOG_PATH}" >&2
    exit 1
  fi

  echo "[managed_run] results verified at ${RESULTS_PATH}" | tee -a "${MANAGED_LOG_PATH}"
done

echo "[managed_run] all experiments completed successfully" | tee -a "${MANAGED_LOG_PATH}"

stop_api_if_needed

if [[ "${POWEROFF_ON_SUCCESS}" == "1" ]]; then
  poweroff_host
fi
