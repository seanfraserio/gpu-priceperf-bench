# Superseded: client connection-pool cap

These results were measured with `httpx.AsyncClient()` left at its default
`max_connections=100`. Concurrency levels above 100 therefore measured the
benchmark client's connection pool rather than the server: requests queued on
the laptop waiting for a socket.

The symptom looked exactly like a hardware saturation knee. On the RTX 5090,
TTFT at concurrency 128 rose from 671ms to 11829ms and aggregate throughput
fell — which is also what a GPU running out of KV cache looks like, and is why
this was reported as a clean result before the cause was found.

Levels at or below 64 are unaffected and remain valid measurements. Everything
above is an artefact of the harness. They are kept here as the record of what
was actually run, outside the directory the report reads, so they cannot
re-enter the published comparison.

Fixed by sizing the pool to the concurrency under test (`src/gppb/sweep.py`).
