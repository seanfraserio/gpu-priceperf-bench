#!/usr/bin/env bash
# runner-vllm/onstart.sh — runs inside the stock vllm/vllm-openai image.
#
# There is no custom image. The vLLM version pin lives in the upstream tag
# (launch.vast.IMAGE); everything this adds is ~200 lines of harness, cloned
# at boot into a container that already spends minutes downloading weights.
set -euo pipefail

# Vast writes instance env into /etc/environment; a non-login onstart shell
# does not pick it up on its own.
if [ -f /etc/environment ]; then
  set -a
  . /etc/environment
  set +a
fi

: "${MODEL:?MODEL is required}"
: "${HOURLY_RATE_USD:?HOURLY_RATE_USD is required}"
TP_SIZE="${TP_SIZE:-1}"
PRECISION="${PRECISION:-fp8}"
TTL_MINUTES="${TTL_MINUTES:-45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPPB_REPO="${GPPB_REPO:-https://github.com/seanfraserio/gpu-priceperf-bench.git}"
GPPB_REF="${GPPB_REF:-main}"

# 1. Arm the TTL before anything else can hang — clone, pip, weight download
# and vLLM boot all sit behind this. Primary budget guard.
( sleep $((TTL_MINUTES * 60)); echo "TTL reached, powering off"; poweroff -f ) &

shutdown_now() { echo "run finished, powering off"; poweroff -f; }
trap shutdown_now EXIT

# 2. Fetch the harness at a pinned revision. A rented GPU never runs whatever
# happens to be on the branch tip.
git clone --filter=blob:none "${GPPB_REPO}" /opt/gppb-src
git -C /opt/gppb-src checkout "${GPPB_REF}"
export PYTHONPATH=/opt/gppb-src/src
pip install --no-cache-dir hf_transfer "pydantic>=2.7" "httpx>=0.27"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

# 3. Download weights, timed separately so it never pollutes benchmark numbers.
DL_START=$(date +%s)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}')
"
export DOWNLOAD_SECONDS=$(( $(date +%s) - DL_START ))
echo "weights downloaded in ${DOWNLOAD_SECONDS}s"

# precision-flag-begin
# --quantization takes a scheme (fp8, awq, gptq); a plain dtype belongs to
# --dtype. Passing a dtype as a quantisation scheme makes vLLM refuse to start,
# which on a rented box means paying for a boot that never serves.
case "${PRECISION}" in
  auto|half|float16|bfloat16|float|float32)
    PRECISION_FLAG="--dtype ${PRECISION}" ;;
  *)
    PRECISION_FLAG="--quantization ${PRECISION}" ;;
esac
# precision-flag-end

# 4. Serve. No speculative decoding, no prefix caching, fixed KV budget.
# VLLM_START_EPOCH lets the harness measure boot from the server's real start,
# not from when Python happened to attach.
export VLLM_START_EPOCH=$(date +%s)
vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  ${PRECISION_FLAG} \
  --no-enable-prefix-caching \
  --port 8000 &

# 5. Benchmark. run_vllm waits for /health and records boot_seconds itself.
python -m gppb.run_vllm
