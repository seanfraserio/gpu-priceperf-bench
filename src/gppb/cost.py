"""Cost arithmetic. Pure, offline, unit-tested — a pricing correction never
costs a GPU rental."""
from __future__ import annotations

from gppb.models import StepResult

SECONDS_PER_HOUR = 3600
TOKENS_PER_MTOK = 1_000_000


def selfhost_usd_per_mtok(hourly_rate_usd: float, output_tokens_per_sec: float) -> float:
    """USD per 1M output tokens when renting a GPU by the hour.

    $/1M = hourly_rate / (tokens_per_sec * 3600) * 1e6
    """
    if output_tokens_per_sec <= 0:
        raise ValueError("output_tokens_per_sec must be positive")
    tokens_per_hour = output_tokens_per_sec * SECONDS_PER_HOUR
    return hourly_rate_usd / tokens_per_hour * TOKENS_PER_MTOK


def api_usd_per_mtok(
    input_per_mtok_usd: float,
    output_per_mtok_usd: float,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """USD per 1M output tokens for a metered API, at the fixed workload shape.

    Normalised to output tokens so it sits on the same axis as the self-host
    number. Input cost is folded in — quoting the bare output rate understates
    the real bill at a 1024/256 shape.
    """
    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    per_request = (
        input_tokens / TOKENS_PER_MTOK * input_per_mtok_usd
        + output_tokens / TOKENS_PER_MTOK * output_per_mtok_usd
    )
    return per_request / output_tokens * TOKENS_PER_MTOK


def coldstart_usd(boot_seconds: float, hourly_rate_usd: float) -> float:
    """What you pay for the GPU sitting there loading weights and warming up."""
    return boot_seconds / SECONDS_PER_HOUR * hourly_rate_usd


def throughput_knee(steps: list[StepResult]) -> StepResult:
    """The sweep step with peak aggregate throughput — where the headline
    $/1M is quoted.

    Considers only steps with no request failures: failed requests can inflate
    throughput (quick 503s), understating cost. On throughput ties, returns the
    step with lower concurrency (cheaper to run at the same speed).
    """
    if not steps:
        raise ValueError("cannot find a knee in an empty sweep")
    clean_steps = [s for s in steps if s.requests_failed == 0]
    if not clean_steps:
        raise ValueError("no clean steps (all had request failures)")
    return max(clean_steps, key=lambda s: s.output_tokens_per_sec)


def saturated(steps: list[StepResult]) -> bool:
    """Whether the sweep actually found the hardware's throughput ceiling.

    If peak throughput lands on the highest concurrency tested, the curve was
    still climbing when the sweep stopped: the peak is a floor on what the GPU
    can do, so the $/1M derived from it is an upper bound, not a measurement.
    Saying so is the difference between a benchmark and a guess."""
    clean = [s for s in steps if s.requests_failed == 0]
    if len(clean) < 2:
        return False
    return throughput_knee(steps).concurrency < max(s.concurrency for s in clean)
