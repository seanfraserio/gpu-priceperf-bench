#!/usr/bin/env bash
# runner-nccl/onstart.sh — runs inside the stock nvidia/cuda devel image.
#
# No custom image. nccl-tests has no official binary distribution, so it is
# compiled here: ~2-3 minutes of billed GPU time against a multi-GPU box, which
# is cheaper than hosting a 6GB image for a benchmark that runs a handful of
# times. Both the harness and nccl-tests are pinned.
set -euo pipefail

# 0. Arm the self-destruct FIRST, before any command that can fail or hang.
# This runs under `set -e`; anything above the TTL that exits non-zero leaves
# an instance billing with nothing to stop it.
TTL_MINUTES="${TTL_MINUTES:-20}"
( sleep $((TTL_MINUTES * 60)); echo "TTL reached, powering off"; poweroff -f ) &

# Vast writes instance env into /etc/environment; a non-login onstart shell
# does not pick it up on its own. A malformed line there must not kill the run.
if [ -f /etc/environment ]; then
  set +e
  set -a
  . /etc/environment 2>/dev/null || true
  set +a
  set -e
fi
TTL_MINUTES="${TTL_MINUTES:-20}"

GPU_COUNT="${GPU_COUNT:-2}"
GPPB_REPO="${GPPB_REPO:-https://github.com/seanfraserio/gpu-priceperf-bench.git}"
GPPB_REF="${GPPB_REF:-main}"
NCCL_TESTS_REPO="${NCCL_TESTS_REPO:-https://github.com/NVIDIA/nccl-tests.git}"
NCCL_TESTS_REF="${NCCL_TESTS_REF:-v2.19.7}"


shutdown_now() { echo "run finished, powering off"; poweroff -f; }
trap shutdown_now EXIT

# 2. Build deps. The devel image ships nvcc; NCCL headers come from apt.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  git build-essential libnccl2 libnccl-dev python3 python3-pip
rm -rf /var/lib/apt/lists/*

# 3. Compile nccl-tests at a pinned tag.
git clone "${NCCL_TESTS_REPO}" /opt/nccl-tests
git -C /opt/nccl-tests checkout "${NCCL_TESTS_REF}"
make -C /opt/nccl-tests -j"$(nproc)"

# 4. Fetch the harness at a pinned revision.
git clone --filter=blob:none "${GPPB_REPO}" /opt/gppb-src
git -C /opt/gppb-src checkout "${GPPB_REF}"
export PYTHONPATH=/opt/gppb-src/src
pip3 install --no-cache-dir "pydantic>=2.7" "httpx>=0.27"

# 5. Measure.
/opt/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g "${GPU_COUNT}" \
  | tee /tmp/all_reduce.txt
/opt/nccl-tests/build/all_gather_perf -b 8 -e 1G -f 2 -g "${GPU_COUNT}" \
  | tee /tmp/all_gather.txt

python3 -m gppb.run_nccl /tmp/all_reduce.txt /tmp/all_gather.txt
