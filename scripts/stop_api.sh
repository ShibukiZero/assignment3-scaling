#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/stop_api.sh [port]

Environment overrides:
  CS336_LOG_DIR   Directory containing API pid files (default: /root/autodl-tmp/api-logs)

Examples:
  ./scripts/stop_api.sh
  ./scripts/stop_api.sh 8001
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PORT="${1:-${PORT:-8000}}"
export CS336_LOG_DIR="${CS336_LOG_DIR:-/root/autodl-tmp/api-logs}"

pid_path="${CS336_LOG_DIR}/api-${PORT}.pid"
shutdown_url="http://127.0.0.1:${PORT}/__admin__/shutdown"

if [[ ! -f "${pid_path}" ]]; then
  echo "[stop_api] missing pid file: ${pid_path}" >&2
  exit 1
fi

server_pid="$(cat "${pid_path}")"

if [[ -z "${server_pid}" ]]; then
  echo "[stop_api] empty pid file: ${pid_path}" >&2
  exit 1
fi

if ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "[stop_api] process ${server_pid} is not running"
  rm -f "${pid_path}"
  exit 0
fi

echo "[stop_api] stopping pid=${server_pid} port=${PORT}"

if command -v curl >/dev/null 2>&1; then
  if curl -fsS -X POST "${shutdown_url}" >/dev/null 2>&1; then
    echo "[stop_api] graceful shutdown requested via ${shutdown_url}"
  else
    echo "[stop_api] graceful shutdown endpoint unavailable; falling back to SIGTERM" >&2
    kill "${server_pid}"
  fi
else
  echo "[stop_api] curl is unavailable; falling back to SIGTERM" >&2
  kill "${server_pid}"
fi

for _ in $(seq 1 30); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    rm -f "${pid_path}"
    echo "[stop_api] stopped"
    exit 0
  fi
  sleep 1
done

echo "[stop_api] process ${server_pid} did not exit after 30s; sending SIGKILL" >&2
kill -9 "${server_pid}"
rm -f "${pid_path}"
echo "[stop_api] stopped"
