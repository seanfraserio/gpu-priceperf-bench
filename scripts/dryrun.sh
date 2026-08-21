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

echo "==> short sweep into a throwaway sink (never the published results/)"
export DRYRUN_DIR="$(mktemp -d)"
trap 'kill $SERVER_PID 2>/dev/null || true; rm -rf "$DRYRUN_DIR"' EXIT
RESULTS_DIR="$DRYRUN_DIR" \
MODEL=mock-model \
HOURLY_RATE_USD=0.00 \
SWEEP=1,2,4 \
TP_SIZE=1 \
SKIP_VLLM_IMPORT=1 \
python -m gppb.run_vllm

echo "==> dry-run result validates against the schema"
python - <<PY
import json, pathlib, jsonschema, os
schema = json.loads(pathlib.Path("schema/result.schema.json").read_text())
files = list(pathlib.Path(os.environ["DRYRUN_DIR"]).glob("*.json"))
assert files, "the sweep produced no result"
for f in files:
    jsonschema.validate(json.loads(f.read_text()), schema)
print(f"  {len(files)} result(s) valid")
PY

echo "==> report generation"
python -m report.generate

echo "==> DRY RUN PASSED — safe to spend money"
