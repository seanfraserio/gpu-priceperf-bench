import pytest
from gppb.cost import selfhost_usd_per_mtok, api_usd_per_mtok, coldstart_usd, throughput_knee
from gppb.models import StepResult, Stats


def _step(concurrency: int, tps: float, failed: int = 0) -> StepResult:
    s = Stats(p50=1.0, p90=1.0, min=1.0, max=1.0)
    return StepResult(
        concurrency=concurrency, requests_completed=1, requests_failed=failed,
        wall_seconds=1.0, output_tokens_total=256, output_tokens_per_sec=tps,
        ttft_ms=s, tpot_ms=s,
    )


def test_selfhost_cost_matches_hand_computed_value():
    # $1.90/hr at 1000 tok/s -> 3.6M tokens/hr -> 1.90/3.6 = $0.5278 per 1M
    assert selfhost_usd_per_mtok(1.90, 1000.0) == pytest.approx(0.527777, rel=1e-5)


def test_selfhost_cost_halves_when_throughput_doubles():
    assert selfhost_usd_per_mtok(2.0, 2000.0) == pytest.approx(
        selfhost_usd_per_mtok(2.0, 1000.0) / 2
    )


def test_selfhost_cost_rejects_zero_throughput():
    with pytest.raises(ValueError):
        selfhost_usd_per_mtok(1.90, 0.0)


def test_api_cost_blends_input_and_output_at_the_fixed_shape():
    # Chutes: $0.40/M in, $3.00/M out, at 1024 in / 256 out.
    # Per request: 1024/1e6*0.40 + 256/1e6*3.00 = 0.0004096 + 0.000768 = 0.0011776
    # Normalised per 1M OUTPUT tokens: 0.0011776 / 256 * 1e6 = 4.6
    assert api_usd_per_mtok(0.40, 3.00, 1024, 256) == pytest.approx(4.6, rel=1e-6)


def test_api_cost_ignoring_input_would_understate_by_a_third():
    """Guard against the classic error of quoting only the output rate."""
    blended = api_usd_per_mtok(0.40, 3.00, 1024, 256)
    assert blended > 3.00


def test_coldstart_cost():
    # 78.1s at $1.90/hr
    assert coldstart_usd(78.1, 1.90) == pytest.approx(78.1 / 3600 * 1.90, rel=1e-9)


def test_throughput_knee_picks_highest_throughput_step():
    steps = [_step(1, 100.0), _step(8, 900.0), _step(64, 1200.0), _step(256, 1150.0)]
    assert throughput_knee(steps).concurrency == 64


def test_throughput_knee_rejects_empty_sweep():
    with pytest.raises(ValueError):
        throughput_knee([])


def test_throughput_knee_skips_failed_steps():
    """Rate-limited failures inflate throughput; skip them to avoid chart lies."""
    steps = [
        _step(1, 100.0),      # clean
        _step(8, 900.0),      # clean
        _step(64, 1500.0, failed=5),  # high throughput but failed — skip
        _step(256, 1150.0),   # clean
    ]
    assert throughput_knee(steps).concurrency == 256


def test_throughput_knee_raises_when_all_steps_failed():
    """No clean step means no reliable throughput number."""
    steps = [
        _step(1, 100.0, failed=1),
        _step(8, 900.0, failed=2),
        _step(64, 1200.0, failed=1),
    ]
    with pytest.raises(ValueError):
        throughput_knee(steps)


def test_throughput_knee_breaks_ties_at_lower_concurrency():
    """On exact throughput tie, pick lower concurrency (cheaper to run)."""
    steps = [
        _step(1, 1000.0),
        _step(8, 1000.0),     # exact tie with concurrency=1
        _step(64, 900.0),
    ]
    assert throughput_knee(steps).concurrency == 1
