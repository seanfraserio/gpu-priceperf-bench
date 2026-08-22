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


def test_the_script_does_not_redirect_its_own_stdout():
    """`exec > >(tee -a log)` made stdout a pipe. If tee ever goes away, the
    next write kills bash with SIGPIPE — and an untrapped SIGPIPE runs no EXIT
    trap, so the instance neither reports the failure nor destroys itself. One
    L40S billed for fifty minutes past the end of its own sweep."""
    body = open(SCRIPT).read()
    assert "exec >" not in body, "vast already records onstart output"
    assert "tee -a" not in body


def test_the_log_read_back_is_the_one_vast_already_writes():
    body = open(SCRIPT).read()
    assert "/var/log/onstart.log" in body


def test_a_signalled_death_still_stops_the_meter():
    """SIGTERM from the host, or a SIGPIPE, must not leave an instance whose
    only remaining stop-clock is a background sleeper."""
    body = open(SCRIPT).read()
    trap_line = [l for l in body.splitlines() if l.startswith("trap on_exit")][0]
    for signal in ("TERM", "HUP", "PIPE"):
        assert signal in trap_line, trap_line


def test_self_destruct_cannot_hang_on_a_stalled_api_call():
    """An L40S finished its sweep and then billed for another fifty minutes.
    curl with no timeout is one way that happens: it blocks forever and the
    poweroff fallback below it never runs."""
    for script in (SCRIPT, "runner-nccl/onstart.sh"):
        body = open(script).read()
        block = body[body.index("# self-destruct-begin"):body.index("# self-destruct-end")]
        assert "--max-time" in block, script


def test_the_upload_identifies_itself_with_a_real_user_agent():
    """Cloudflare answers Python-urllib's default User-Agent with 403, so every
    failure report was silently rejected while the stubbed tests above passed.
    The results themselves upload through httpx, whose agent is allowed, which
    is why only the diagnostic path was dark."""
    body = open(SCRIPT).read()
    block = body[body.index("# failure-report-begin"):body.index("# failure-report-end")]
    assert "User-Agent" in block, "an unset agent is rejected before it is read"
    assert "urllib" not in block.split("User-Agent")[1].split("\n")[0]


def test_the_log_path_reaches_the_child_process():
    """GPPB_LOG_PATH was a plain shell variable, so the uploader — a separate
    python3 — read an empty string and reported 'log unavailable'. The report
    arrived carrying nothing."""
    body = open(SCRIPT).read()
    block = body[body.index("# failure-report-begin"):body.index("# failure-report-end")]
    assert "export GPPB_LOG_PATH" in block


def test_the_uploader_tries_more_than_one_log_location():
    """The path vast writes is a convention, not a promise. A report that says
    only 'no such file' is the same as no report."""
    body = open(SCRIPT).read()
    block = body[body.index("# failure-report-begin"):body.index("# failure-report-end")]
    assert block.count("/var/log/") >= 1
    assert "candidates" in block
