import subprocess, sys, time
import httpx
import pytest
from gppb.sweep import run_step
from gppb.models import Workload


@pytest.fixture(scope="module")
def server():
    proc = subprocess.Popen([sys.executable, "tests/mock_server.py", "8123"])
    for _ in range(50):
        try:
            if httpx.get("http://127.0.0.1:8123/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("mock server never became healthy")
    yield "http://127.0.0.1:8123"
    proc.kill()


async def test_sweep_runs_against_mock_server(server):
    step = await run_step(server, "mock", concurrency=2, workload=Workload(), requests_per_step=4)
    assert step.requests_failed == 0
    assert step.requests_completed == 4
    assert step.output_tokens_total == 4 * 256
    assert step.output_tokens_per_sec > 0
