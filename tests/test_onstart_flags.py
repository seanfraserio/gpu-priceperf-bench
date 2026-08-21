"""The serve flags are shell, but a wrong flag costs a rented boot — so the
selection logic is exercised directly."""
import subprocess

SCRIPT = "runner-vllm/onstart.sh"


def _precision_flag(precision: str) -> str:
    """Run the script's flag-selection block in isolation."""
    body = open(SCRIPT).read()
    start = body.index("# precision-flag-begin")
    end = body.index("# precision-flag-end")
    snippet = body[start:end]
    out = subprocess.run(
        ["bash", "-c", f'PRECISION="{precision}"\n{snippet}\necho "$PRECISION_FLAG"'],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_a_dtype_becomes_dtype_not_quantization():
    """--quantization bfloat16 is not a valid vLLM argument; the server would
    refuse to start and the rented boot would be wasted."""
    assert _precision_flag("bfloat16") == "--dtype bfloat16"


def test_float16_is_also_a_dtype():
    assert _precision_flag("float16") == "--dtype float16"


def test_auto_is_a_dtype():
    assert _precision_flag("auto") == "--dtype auto"


def test_a_quantisation_scheme_stays_quantization():
    assert _precision_flag("fp8") == "--quantization fp8"


def test_awq_stays_quantization():
    assert _precision_flag("awq") == "--quantization awq"


SOURCE_CMD = ". /etc/environment"


def test_ttl_arms_before_the_environment_is_sourced():
    """The script runs under `set -e`. Sourcing /etc/environment can fail on a
    malformed line, and if the TTL is not already armed the script dies with no
    self-destruct — the instance then bills until someone notices."""
    body = open(SCRIPT).read()
    assert body.index("poweroff -f") < body.index(SOURCE_CMD)


def test_sourcing_the_environment_cannot_kill_the_script():
    """A bad line in /etc/environment must not abort the run."""
    body = open(SCRIPT).read()
    start = body.index(SOURCE_CMD)
    window = body[start - 200:start + 200]
    assert "|| true" in window or "set +e" in window, window


def test_nccl_runner_arms_ttl_before_sourcing_too():
    body = open("runner-nccl/onstart.sh").read()
    assert body.index("poweroff -f") < body.index(SOURCE_CMD)


def _run_snippet(script: str, marker: str, prelude: str, call: str) -> str:
    """Execute one marked block from an onstart script in isolation."""
    body = open(script).read()
    snippet = body[body.index(f"# {marker}-begin"):body.index(f"# {marker}-end")]
    out = subprocess.run(["bash", "-c", f"{prelude}\n{snippet}\n{call}"],
                         capture_output=True, text=True)
    return out.stdout + out.stderr


def test_self_destruct_calls_the_vast_api_with_the_container_key():
    """poweroff is denied in an unprivileged container — it failed with
    'Operation not permitted' on a real instance while the meter kept running.
    The instance must destroy itself through the API instead."""
    out = _run_snippet(
        SCRIPT, "self-destruct",
        prelude=(
            'export CONTAINER_API_KEY=tok123\n'
            'export VAST_CONTAINERLABEL=C.4242\n'
            'curl() { echo "CURL $*"; }\n'
            'poweroff() { echo "POWEROFF $*"; }\n'
        ),
        call="self_destruct",
    )
    assert "4242" in out, out
    assert "tok123" in out, out
    assert "instances" in out, out


def test_self_destruct_still_tries_poweroff_as_a_fallback():
    out = _run_snippet(
        SCRIPT, "self-destruct",
        prelude=(
            'unset CONTAINER_API_KEY VAST_CONTAINERLABEL\n'
            'curl() { echo "CURL $*"; }\n'
            'poweroff() { echo "POWEROFF $*"; }\n'
        ),
        call="self_destruct",
    )
    assert "POWEROFF" in out, out


def test_scripts_use_python3_not_python():
    """The vLLM image ships python3; a bare `python` is not on PATH and the
    run died at the weight download with 'python: command not found'."""
    for script in (SCRIPT, "runner-nccl/onstart.sh"):
        body = open(script).read()
        for line in body.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("python "), f"{script}: {stripped}"
            assert not stripped.startswith("python -"), f"{script}: {stripped}"


def test_two_ttls_a_backstop_and_the_configured_one():
    """The first TTL arms before the environment is readable, so it can only
    use a default. Once the real TTL_MINUTES is known a second timer arms with
    it — otherwise a run asking for 20 minutes would burn 45."""
    body = open(SCRIPT).read()
    armings = [i for i in range(len(body)) if body.startswith("( sleep $((TTL", i)]
    assert len(armings) == 2, f"expected a backstop and a configured TTL, got {len(armings)}"
    assert armings[0] < body.index(SOURCE_CMD), "the backstop must arm first"
    assert armings[1] > body.index(SOURCE_CMD), "the real TTL arms once env is known"


def test_backstop_ttl_is_not_shorter_than_a_normal_run():
    """A backstop that fires mid-run would kill good work."""
    body = open(SCRIPT).read()
    assert "TTL_BACKSTOP_MINUTES:-90" in body
