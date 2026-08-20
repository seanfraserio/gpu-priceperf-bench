#!/usr/bin/env bash
# scripts/dryrun.sh — full path proof, zero spend, any machine.
set -euo pipefail

echo "==> unit tests"
pytest -q

echo "==> starting mock OpenAI-compatible server"
python tests/mock_server.py 8000 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "==> waiting for health"
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 1; done

echo "==> short sweep into local sink"
MODEL=mock-model \
HOURLY_RATE_USD=0.00 \
SWEEP=1,2,4 \
TP_SIZE=1 \
SKIP_VLLM_IMPORT=1 \
python -m gppb.run_vllm

echo "==> schema validation"
pytest tests/test_schema_validation.py -q

echo "==> report generation"
python -m report.generate

echo "==> DRY RUN PASSED — safe to spend money"
