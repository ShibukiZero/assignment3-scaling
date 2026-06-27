#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${2:-runs/api_smoke_${STAMP}}"
INCLUDE_CAP_TESTS="${INCLUDE_CAP_TESTS:-0}"

mkdir -p "${OUTDIR}"

RUN_ID="${STAMP}"
SMALL_KEY="suite-small-${RUN_ID}"
MAX128_KEY="suite-max128-${RUN_ID}"
MAX256_KEY="suite-max256-${RUN_ID}"
NEVER_USED_KEY="suite-never-used-${RUN_ID}"

SUMMARY_TSV="${OUTDIR}/summary.tsv"
METADATA_TXT="${OUTDIR}/metadata.txt"

cat > "${METADATA_TXT}" <<EOF
base_url=${BASE_URL}
run_id=${RUN_ID}
include_cap_tests=${INCLUDE_CAP_TESTS}
small_key=${SMALL_KEY}
max128_key=${MAX128_KEY}
max256_key=${MAX256_KEY}
never_used_key=${NEVER_USED_KEY}
started_at=$(date -Is)
EOF

printf "name\tstatus\tbody_file\turl_file\n" > "${SUMMARY_TSV}"

run_case() {
  local name="$1"
  local url="$2"
  local body_file="${OUTDIR}/${name}.body.json"
  local status_file="${OUTDIR}/${name}.status.txt"
  local url_file="${OUTDIR}/${name}.url.txt"

  printf "%s\n" "${url}" > "${url_file}"
  local status
  status="$(curl -sS -o "${body_file}" -w "%{http_code}" "${url}")"
  printf "%s\n" "${status}" > "${status_file}"
  printf "%s\t%s\t%s\t%s\n" "${name}" "${status}" "$(basename "${body_file}")" "$(basename "${url_file}")" >> "${SUMMARY_TSV}"
}

run_case \
  "01_loss_small_first" \
  "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=${SMALL_KEY}"

run_case \
  "02_loss_small_cached" \
  "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=${SMALL_KEY}"

run_case \
  "03_total_small_after_cache_check" \
  "${BASE_URL}/total_flops_used?api_key=${SMALL_KEY}"

run_case \
  "04_history_small_after_cache_check" \
  "${BASE_URL}/previous_runs?api_key=${SMALL_KEY}"

run_case \
  "05_loss_small_second_budget" \
  "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=30000000000000&api_key=${SMALL_KEY}"

run_case \
  "06_total_small_after_second_query" \
  "${BASE_URL}/total_flops_used?api_key=${SMALL_KEY}"

run_case \
  "07_history_small_after_second_query" \
  "${BASE_URL}/previous_runs?api_key=${SMALL_KEY}"

run_case \
  "08_invalid_d_model" \
  "${BASE_URL}/loss?d_model=63&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=${SMALL_KEY}"

run_case \
  "09_invalid_num_layers" \
  "${BASE_URL}/loss?d_model=64&num_layers=1&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=${SMALL_KEY}"

run_case \
  "10_invalid_num_heads" \
  "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=32&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=${SMALL_KEY}"

run_case \
  "11_invalid_batch_size" \
  "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=2&batch_size=64&learning_rate=0.0003&train_flops=10000000000000&api_key=${SMALL_KEY}"

run_case \
  "12_invalid_learning_rate" \
  "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.01&train_flops=10000000000000&api_key=${SMALL_KEY}"

run_case \
  "13_invalid_train_flops" \
  "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=20000000000000&api_key=${SMALL_KEY}"

run_case \
  "14_incompatible_d_model_num_heads" \
  "${BASE_URL}/loss?d_model=510&num_layers=2&num_heads=8&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=${SMALL_KEY}"

run_case \
  "15_never_used_total" \
  "${BASE_URL}/total_flops_used?api_key=${NEVER_USED_KEY}"

run_case \
  "16_never_used_history" \
  "${BASE_URL}/previous_runs?api_key=${NEVER_USED_KEY}"

run_case \
  "17_loss_max_128" \
  "${BASE_URL}/loss?d_model=1024&num_layers=24&num_heads=16&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=${MAX128_KEY}"

run_case \
  "18_loss_max_128_cached" \
  "${BASE_URL}/loss?d_model=1024&num_layers=24&num_heads=16&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=${MAX128_KEY}"

run_case \
  "19_total_max_128" \
  "${BASE_URL}/total_flops_used?api_key=${MAX128_KEY}"

run_case \
  "20_history_max_128" \
  "${BASE_URL}/previous_runs?api_key=${MAX128_KEY}"

run_case \
  "21_loss_max_256" \
  "${BASE_URL}/loss?d_model=1024&num_layers=24&num_heads=16&batch_size=256&learning_rate=0.0003&train_flops=10000000000000&api_key=${MAX256_KEY}"

run_case \
  "22_loss_max_256_cached" \
  "${BASE_URL}/loss?d_model=1024&num_layers=24&num_heads=16&batch_size=256&learning_rate=0.0003&train_flops=10000000000000&api_key=${MAX256_KEY}"

run_case \
  "23_total_max_256" \
  "${BASE_URL}/total_flops_used?api_key=${MAX256_KEY}"

run_case \
  "24_history_max_256" \
  "${BASE_URL}/previous_runs?api_key=${MAX256_KEY}"

if [[ "${INCLUDE_CAP_TESTS}" == "1" ]]; then
  CAP_KEY="suite-cap-${RUN_ID}"
  printf "cap_key=%s\n" "${CAP_KEY}" >> "${METADATA_TXT}"

  run_case \
    "25_cap_first_1e18" \
    "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.001&train_flops=1000000000000000000&api_key=${CAP_KEY}"

  run_case \
    "26_cap_second_1e18" \
    "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0009&train_flops=1000000000000000000&api_key=${CAP_KEY}"

  run_case \
    "27_cap_rejected_after_limit" \
    "${BASE_URL}/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0008&train_flops=10000000000000&api_key=${CAP_KEY}"

  run_case \
    "28_cap_total_after_rejection" \
    "${BASE_URL}/total_flops_used?api_key=${CAP_KEY}"

  run_case \
    "29_cap_history_after_rejection" \
    "${BASE_URL}/previous_runs?api_key=${CAP_KEY}"
fi

printf "finished_at=%s\n" "$(date -Is)" >> "${METADATA_TXT}"

echo "Wrote smoke-suite logs to ${OUTDIR}"
