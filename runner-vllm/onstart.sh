#!/usr/bin/env bash
# runner-vllm/onstart.sh — runs inside the stock vllm/vllm-openai image.
#
# There is no custom image. The vLLM version pin lives in the upstream tag
# (launch.vast.IMAGE); everything this adds is ~200 lines of harness, cloned
# at boot into a container that already spends minutes downloading weights.
set -euo pipefail

# Everything this script prints is the only account of what happened on a
# machine that is about to be destroyed. Vast already records it, so read that
# file rather than restructuring stdout: routing it through `tee` made stdout a
# pipe, and one dead tee then kills bash with an untrapped SIGPIPE — no EXIT
# trap, no failure report, and no self-destruct.
GPPB_LOG_PATH="${GPPB_LOG_PATH:-/var/log/onstart.log}"

# self-destruct-begin
# `poweroff` is denied in an unprivileged container — on a real instance it
# failed with "Operation not permitted" while the meter kept running. Vast
# gives every container a scoped key and its own label, which together can
# destroy exactly this instance and nothing else on the account.
self_destruct() {
  local id="${VAST_CONTAINERLABEL#C.}"
  if [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${id:-}" ]; then
    echo "self-destruct: destroying instance ${id} via API"
    curl -sS --max-time 30 --retry 3 --retry-delay 5 -X DELETE \
      "https://console.vast.ai/api/v0/instances/${id}/" \
      -H "Authorization: Bearer ${CONTAINER_API_KEY}" \
      -H "Content-Type: application/json" \
      -d '{}' || true
  else
    echo "self-destruct: no container API key; falling back to poweroff"
  fi
  # Best effort, and harmless when the API call already worked.
  poweroff -f 2>/dev/null || true
}
# self-destruct-end

# 0. Arm the self-destruct FIRST, before any command that can fail or hang.
# This script runs under `set -e`; anything above this point that exits
# non-zero leaves an instance billing with nothing to stop it.
# The instance environment is not readable yet, so this can only use a fixed
# backstop. It is deliberately longer than any real run: its job is to catch a
# script that dies before the configured TTL below ever arms.
TTL_BACKSTOP_MINUTES="${TTL_BACKSTOP_MINUTES:-90}"
( sleep $((TTL_BACKSTOP_MINUTES * 60)); echo "backstop TTL reached"; self_destruct ) &

# Vast writes instance env into /etc/environment; a non-login onstart shell
# does not pick it up on its own. A malformed line in that file must not kill
# the run, so failures here are swallowed deliberately.
if [ -f /etc/environment ]; then
  set +e
  set -a
  . /etc/environment 2>/dev/null || true
  set +a
  set -e
fi
# Now that the instance environment is loaded, arm the real TTL. Whichever
# timer fires first wins; the backstop above only matters if we never got here.
TTL_MINUTES="${TTL_MINUTES:-45}"
( sleep $((TTL_MINUTES * 60)); echo "TTL reached"; self_destruct ) &


: "${MODEL:?MODEL is required}"

# rate-fetch-begin
# The network call, kept apart from the logic that decides what to do with it
# so that logic can be exercised against a stand-in.
gppb_fetch_instance_json() {
  # The key is read from the environment inside the program. argv is
  # world-readable through ps, and this key controls the instance that is
  # spending money right now.
  python3 -c '
import os, urllib.request
request = urllib.request.Request(
    "https://console.vast.ai/api/v0/instances/%s/" % os.environ["CONTAINER_ID"],
    headers={"Authorization": "Bearer %s" % os.environ["CONTAINER_API_KEY"],
             "User-Agent": "gppb/1"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    print(response.read().decode())
'
}
# rate-fetch-end

# rate-discovery-begin
# HOURLY_RATE_USD is the denominator of every published $/1M, and it arrives
# from outside. The orchestrator passes the accepted offer's rate, which is
# correct and needs no help. A template launched from the web UI has no such
# plumbing, and a hand-typed rate that is stale or mistyped corrupts the
# headline in silence — the run succeeds, the result validates, and the number
# is fiction. So when nobody supplied one, the instance asks the marketplace
# what it is being charged.
#
# dph_total, not dph_base: the all-in figure includes storage, and every tier
# here rents 120GB. Verified live 2026-08-23 on a probe instance —
# dph_total 0.05 against dph_base 0.0467 on the same container.
if [ -z "${HOURLY_RATE_USD:-}" ]; then
  if [ -z "${CONTAINER_ID:-}" ] || [ -z "${CONTAINER_API_KEY:-}" ]; then
    echo "HOURLY_RATE_USD is required: none supplied, and this container has" \
         "no CONTAINER_ID/CONTAINER_API_KEY to look it up with" >&2
    exit 1
  fi
  HOURLY_RATE_USD="$(gppb_fetch_instance_json | python3 -c '
import json, sys
payload = json.load(sys.stdin)
instance = payload.get("instances", payload)
if isinstance(instance, list):
    instance = instance[0] if instance else {}
rate = instance.get("dph_total")
# An error page must never become a price, and a zero would divide into
# infinity and publish it.
if not isinstance(rate, (int, float)) or rate <= 0:
    sys.exit("no usable dph_total in the marketplace reply")
print(repr(float(rate)))
')" || {
    echo "HOURLY_RATE_USD could not be discovered — refusing to publish a" \
         "cost derived from a rate nobody knows" >&2
    exit 1
  }
  echo "discovered hourly rate: ${HOURLY_RATE_USD}"
fi
export HOURLY_RATE_USD
# rate-discovery-end

: "${HOURLY_RATE_USD:?HOURLY_RATE_USD is required}"
TP_SIZE="${TP_SIZE:-1}"
PRECISION="${PRECISION:-fp8}"
TTL_MINUTES="${TTL_MINUTES:-45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPPB_REPO="${GPPB_REPO:-https://github.com/seanfraserio/gpu-priceperf-bench.git}"
GPPB_REF="${GPPB_REF:-main}"


# failure-report-begin
# A run that dies before it uploads a result looks exactly like one that
# finished: the container disappears either way and the orchestrator sees
# "completed". Two runs were burned that way with nothing left to diagnose,
# because the log dies with the container. A non-zero exit publishes it.
report_failure() {
  export GPPB_EXIT_CODE="$1"
  export GPPB_LOG_PATH
  export GPPB_FAIL_KEY="fail-$(date +%s)-$$.json"
  if [ -z "${SINK_URL:-}" ] || [ -z "${SINK_TOKEN:-}" ]; then
    echo "failure report: no sink configured"
    return 0
  fi
  # The token is read from the environment inside the program: argv is
  # world-readable through ps.
  python3 -c '
import json, os, urllib.request
# The path vast writes is a convention, not a promise, so try the ones it
# is known to use before giving up. A report that says only "no such file"
# is worth no more than no report.
candidates = [os.environ.get("GPPB_LOG_PATH", ""),
              "/var/log/onstart.log", "/var/log/onstart_bash.log",
              "/var/log/portal/onstart.log"]
tail = "log unavailable; looked in %s" % ", ".join(c for c in candidates if c)
for path in candidates:
    if not path:
        continue
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 64000))
            tail = "%s:\n%s" % (path, fh.read().decode("utf-8", "replace"))
        break
    except OSError:
        continue
body = json.dumps({
    "kind": "failure",
    "exit_code": int(os.environ["GPPB_EXIT_CODE"]),
    "model": os.environ.get("MODEL", ""),
    "run_index": os.environ.get("RUN_INDEX", ""),
    "log": tail,
}).encode()
request = urllib.request.Request(
    os.environ["SINK_URL"].rstrip("/") + "/" + os.environ["GPPB_FAIL_KEY"],
    data=body, method="PUT",
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + os.environ["SINK_TOKEN"],
             # Cloudflare answers urllib default agent with 403, so without
             # this the report is rejected before the token is even read.
             "User-Agent": "gppb-runner/1.0"},
)
urllib.request.urlopen(request, timeout=60).read()
' && echo "failure report: uploaded ${GPPB_FAIL_KEY}" \
  || echo "failure report: upload failed"
}

# Diagnosis is worth less than the meter: whatever happens above, the instance
# stops billing.
on_exit() {
  local code=$?
  trap - EXIT
  if [ "${code}" -ne 0 ]; then
    echo "onstart exited ${code}"
    report_failure "${code}"
  else
    echo "run finished"
  fi
  self_destruct
}
# failure-report-end
trap on_exit EXIT TERM HUP PIPE

# 2. Fetch the harness at a pinned revision. A rented GPU never runs whatever
# happens to be on the branch tip.
git clone --filter=blob:none "${GPPB_REPO}" /opt/gppb-src
git -C /opt/gppb-src checkout "${GPPB_REF}"
export PYTHONPATH=/opt/gppb-src/src
pip3 install --no-cache-dir hf_transfer "pydantic>=2.7" "httpx>=0.27"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

# 3. Download weights, timed separately so it never pollutes benchmark numbers.
DL_START=$(date +%s)
python3 -c "
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

# batch-cap-begin
# vLLM defaults max_num_seqs to 1024. The 27B is a hybrid: every decode
# sequence holds one Mamba cache block, and an 80GB card is left with ~823 of
# them, so CUDA graph capture refuses to start and the whole rented boot is
# lost before a single token is served. The sweep never has more than its top
# level in flight, so capping there removes the failure without capping any
# batch the benchmark actually forms.
_levels="${SWEEP:-1,2,4,8,16,32,64,128,256}"
MAX_NUM_SEQS="$(printf '%s\n' ${_levels//,/ } | sort -n | tail -1)"
# batch-cap-end

# 4. Serve. No speculative decoding, no prefix caching, fixed KV budget.
# VLLM_START_EPOCH lets the harness measure boot from the server's real start,
# not from when Python happened to attach.
export VLLM_START_EPOCH=$(date +%s)
vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  ${PRECISION_FLAG} \
  --no-enable-prefix-caching \
  --port 8000 &
# The harness watches this pid: a boot that has already failed should not cost
# another thirty minutes of polling on a rented GPU.
VLLM_PID=$!
export VLLM_PID

# 5. Benchmark. run_vllm waits for /health and records boot_seconds itself.
python3 -m gppb.run_vllm
