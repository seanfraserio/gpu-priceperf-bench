"""A rented instance that dies before it uploads anything is indistinguishable
from one that finished: the container disappears either way, and the
orchestrator reports 'completed'. Two runs were burned that way with no
diagnosis available, because the only witness — the onstart log — died with the
container. A non-zero exit must publish that log to the sink first."""
import subprocess

SCRIPT = "runner-vllm/onstart.sh"


def _run_block(prelude: str, call: str, script: str = SCRIPT) -> str:
    body = open(script).read()
    start = body.index("# failure-report-begin")
    end = body.index("# failure-report-end")
    snippet = body[start:end]
    out = subprocess.run(["bash", "-c", f"{prelude}\n{snippet}\n{call}"],
                         capture_output=True, text=True)
    return out.stdout + out.stderr


_STUBS = (
    'export SINK_URL=https://sink.example\n'
    'export SINK_TOKEN=tok123\n'
    'export GPPB_LOG_PATH=/dev/null\n'
    'self_destruct() { echo "SELF_DESTRUCT"; }\n'
    'python3() { echo "UPLOAD key=${GPPB_FAIL_KEY} code=${GPPB_EXIT_CODE}"; cat >/dev/null; }\n'
)


def test_a_non_zero_exit_uploads_the_log_before_self_destruct():
    out = _run_block(_STUBS, '( exit 7 ); on_exit')
    assert "UPLOAD" in out, out
    assert "code=7" in out, out
    assert out.index("UPLOAD") < out.index("SELF_DESTRUCT"), out


def test_the_failure_object_is_keyed_so_it_cannot_be_read_as_a_result():
    """The sink Worker only accepts .json keys, so the log rides inside a JSON
    document — which means the report reader must be able to tell it apart from
    a benchmark result by key alone."""
    out = _run_block(_STUBS, '( exit 1 ); on_exit')
    key = [w for w in out.split() if w.startswith("key=")][0][4:]
    assert key.startswith("fail-"), key
    assert key.endswith(".json"), key


def test_a_clean_exit_uploads_nothing():
    """A successful run already published its result; a second object would be
    noise in the published record."""
    out = _run_block(_STUBS, 'true; on_exit')
    assert "UPLOAD" not in out, out
    assert "SELF_DESTRUCT" in out, out


def test_an_upload_that_fails_still_destroys_the_instance():
    """Diagnosis is worth less than the meter. If the sink is unreachable the
    instance must still stop billing."""
    stubs = _STUBS.replace(
        'python3() { echo "UPLOAD key=${GPPB_FAIL_KEY} code=${GPPB_EXIT_CODE}"; cat >/dev/null; }',
        'python3() { cat >/dev/null; return 1; }',
    )
    out = _run_block(stubs, '( exit 3 ); on_exit')
    assert "SELF_DESTRUCT" in out, out


def test_no_sink_configured_is_not_an_error():
    """A failure before /etc/environment is sourced has nowhere to report to."""
    stubs = _STUBS.replace('export SINK_URL=https://sink.example\n', 'unset SINK_URL\n')
    out = _run_block(stubs, '( exit 2 ); on_exit')
    assert "SELF_DESTRUCT" in out, out


def test_the_token_never_reaches_the_upload_command_line():
    """argv is world-readable through ps; the token travels in the environment."""
    body = open(SCRIPT).read()
    block = body[body.index("# failure-report-begin"):body.index("# failure-report-end")]
    for line in block.splitlines():
        assert "$SINK_TOKEN" not in line or "environ" in line, line
        assert "${SINK_TOKEN}" not in line or "-z" in line, line


def test_the_trap_is_armed_before_anything_that_can_fail():
    """git clone, pip install and the weight download are all failure modes
    that produced no report; the trap must already cover them."""
    body = open(SCRIPT).read()
    assert body.index("trap on_exit EXIT") < body.index("git clone")


def test_the_log_is_captured_from_the_top_of_the_script():
    """A tail that starts after the failing command reports nothing useful."""
    body = open(SCRIPT).read()
    assert "tee -a" in body
    assert body.index("tee -a") < body.index("git clone")
