#!/usr/bin/env bash
# runner-nccl/entrypoint.sh
set -euo pipefail

GPU_COUNT="${GPU_COUNT:-2}"
TTL_MINUTES="${TTL_MINUTES:-20}"

( sleep $((TTL_MINUTES * 60)); poweroff -f ) &

/opt/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g "${GPU_COUNT}" \
  | tee /tmp/all_reduce.txt
/opt/nccl-tests/build/all_gather_perf -b 8 -e 1G -f 2 -g "${GPU_COUNT}" \
  | tee /tmp/all_gather.txt

python3 -m gppb.run_nccl /tmp/all_reduce.txt /tmp/all_gather.txt
poweroff -f
