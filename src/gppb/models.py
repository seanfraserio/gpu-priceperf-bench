"""The result contract. Every run in results/ is one serialized BenchResult."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


class Stats(BaseModel):
    p50: float
    p90: float
    min: float
    max: float


class StepResult(BaseModel):
    """One concurrency level of the sweep."""
    concurrency: int
    requests_completed: int
    requests_failed: int
    wall_seconds: float
    output_tokens_total: int
    output_tokens_per_sec: float
    ttft_ms: Stats
    tpot_ms: Stats


class Target(BaseModel):
    kind: Literal["vllm", "openrouter", "nccl"]
    model: str
    precision: str | None = None
    tp_size: int | None = None
    provider: str | None = None


class Hardware(BaseModel):
    gpu_name: str
    gpu_count: int = 1
    driver_version: str | None = None
    vllm_version: str | None = None
    peak_vram_bytes: int | None = None


class Pricing(BaseModel):
    """Rates as they actually were at run time. Never edited after the fact."""
    hourly_rate_usd: float | None = None
    input_per_mtok_usd: float | None = None
    output_per_mtok_usd: float | None = None


class Timings(BaseModel):
    download_seconds: float | None = None
    boot_seconds: float | None = None


class Workload(BaseModel):
    """Defaults are the global constraints. Overriding these breaks comparability."""
    input_tokens: int = 1024
    output_tokens: int = 256
    temperature: float = 0.0
    ignore_eos: bool = True
    max_model_len: int = 32768


class NcclRow(BaseModel):
    size_bytes: int
    busbw_gbps: float
    algbw_gbps: float


class BenchResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    valid: bool = True
    invalid_reason: str | None = None
    partial: bool = False
    target: Target
    hardware: Hardware
    pricing: Pricing
    timings: Timings
    workload: Workload
    run_index: int = 1
    steps: list[StepResult] = Field(default_factory=list)
    nccl_rows: list[NcclRow] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
