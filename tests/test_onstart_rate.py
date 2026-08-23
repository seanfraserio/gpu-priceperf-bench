"""An instance that does not know what it costs cannot produce a cost number.

HOURLY_RATE_USD is the denominator of every published $/1M, and it is supplied
from outside. The orchestrator passes the accepted offer's rate, which is
correct and automatic. A template launched from the web UI has no such
plumbing, and a hand-typed rate that is stale or simply wrong corrupts the
headline silently — the run succeeds, the result validates, and the number is
fiction.

So the instance asks the marketplace what it is being charged. Verified live on
2026-08-23: CONTAINER_ID and CONTAINER_API_KEY are injected into every
container, and GET /api/v0/instances/<id>/ returns dph_total for that key."""
import subprocess

SCRIPT = "runner-vllm/onstart.sh"


def _discover(env: str) -> subprocess.CompletedProcess:
    """Run the rate-discovery block in isolation, under a given environment."""
    body = open(SCRIPT).read()
    start = body.index("# rate-discovery-begin")
    end = body.index("# rate-discovery-end")
    snippet = body[start:end]
    return subprocess.run(
        ["bash", "-c", f'{env}\n{snippet}\necho "RATE=$HOURLY_RATE_USD"'],
        capture_output=True, text=True,
    )


def test_an_explicit_rate_is_never_overridden():
    """The orchestrator already knows the accepted offer's rate. Asking the
    marketplace again would add a network call and a way to disagree with the
    price the run was actually budgeted against."""
    out = _discover('HOURLY_RATE_USD=2.5894; CONTAINER_ID=1; CONTAINER_API_KEY=k')
    assert "RATE=2.5894" in out.stdout


def test_the_rate_is_discovered_when_it_was_not_supplied():
    """A template launched from the UI supplies no rate. dph_total is the
    all-in figure — compute plus storage — which is what the run is billed."""
    out = _discover(
        'unset HOURLY_RATE_USD\n'
        'CONTAINER_ID=42\n'
        'CONTAINER_API_KEY=secret\n'
        # Stand in for the marketplace.
        'gppb_fetch_instance_json() { echo \'{"instances": '
        '{"dph_total": 1.3775, "dph_base": 1.3352}}\'; }\n'
    )
    assert "RATE=1.3775" in out.stdout, out.stdout + out.stderr


def test_the_all_in_rate_is_preferred_over_the_compute_only_one():
    """dph_base excludes storage. Every tier here rents 120GB, so quoting
    dph_base would understate the true bill on every row at once."""
    out = _discover(
        'unset HOURLY_RATE_USD\n'
        'CONTAINER_ID=42\n'
        'CONTAINER_API_KEY=secret\n'
        'gppb_fetch_instance_json() { echo \'{"instances": '
        '{"dph_total": 1.3775, "dph_base": 1.3352}}\'; }\n'
    )
    assert "RATE=1.3352" not in out.stdout


def test_a_run_that_cannot_learn_its_rate_refuses_to_start():
    """The alternative is a result carrying an invented denominator. Failing at
    boot costs one minute; publishing a fictional $/1M costs the comparison."""
    out = _discover(
        'unset HOURLY_RATE_USD\n'
        'unset CONTAINER_ID\n'
        'unset CONTAINER_API_KEY\n'
    )
    assert out.returncode != 0
    assert "HOURLY_RATE_USD" in (out.stdout + out.stderr)


def test_an_unparseable_reply_is_not_treated_as_a_rate():
    """A marketplace error page must not become a price. Zero or empty would
    divide into infinity and publish it."""
    out = _discover(
        'unset HOURLY_RATE_USD\n'
        'CONTAINER_ID=42\n'
        'CONTAINER_API_KEY=secret\n'
        'gppb_fetch_instance_json() { echo "<html>502 Bad Gateway</html>"; }\n'
    )
    assert out.returncode != 0
    assert "RATE=0" not in out.stdout


def test_the_container_key_never_reaches_the_command_line():
    """Same rule as SINK_TOKEN: argv is world-readable through ps, and this key
    controls the instance that is currently spending money. It is read from the
    environment inside the program, never interpolated into an argument."""
    body = open(SCRIPT).read()
    start = body.index("# rate-discovery-begin")
    end = body.index("# rate-discovery-end")
    snippet = body[start:end]
    fetch = body[body.index("# rate-fetch-begin"):body.index("# rate-fetch-end")]
    assert "$CONTAINER_API_KEY" not in fetch
    assert "${CONTAINER_API_KEY}" not in fetch
    assert 'os.environ["CONTAINER_API_KEY"]' in fetch
