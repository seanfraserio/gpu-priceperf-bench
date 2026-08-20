#!/usr/bin/env bash
# runner-vllm/entrypoint.sh
set -euo pipefail

: "${MODEL:?MODEL is required}"
: "${HOURLY_RATE_USD:?HOURLY_RATE_USD is required}"
TP_SIZE="${TP_SIZE:-1}"
PRECISION="${PRECISION:-fp8}"
TTL_MINUTES="${TTL_MINUTES:-45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# 1. Arm the TTL before anything else can hang. This is the primary budget guard.
( sleep $((TTL_MINUTES * 60)); echo "TTL reached, powering off"; poweroff -f ) &

shutdown_now() { echo "run finished, powering off"; poweroff -f; }
trap shutdown_now EXIT

# 2. Download weights, timed separately so it never pollutes benchmark numbers.
DL_START=$(date +%s)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}')
"
export DOWNLOAD_SECONDS=$(( $(date +%s) - DL_START ))
echo "weights downloaded in ${DOWNLOAD_SECONDS}s"

# 3. Serve. No speculative decoding, no prefix caching, fixed KV budget.
# VLLM_START_EPOCH lets the harness measure boot from the server's real start,
# not from when Python happened to attach.
export VLLM_START_EPOCH=$(date +%s)
vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --quantization "${PRECISION}" \
  --no-enable-prefix-caching \
  --port 8000 &

# 4. Benchmark. run_vllm waits for /health and records boot_seconds itself.
python -m gppb.run_vllm
