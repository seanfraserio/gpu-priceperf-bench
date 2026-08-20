"""Parse nccl-tests tabular output into the shared result envelope."""
from __future__ import annotations

from gppb.models import NcclRow


def parse_nccl_output(text: str) -> list[NcclRow]:
    """nccl-tests prints comment lines starting with '#' and fixed-width data
    rows: size, count, type, redop, root, time, algbw, busbw."""
    rows: list[NcclRow] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8:
            continue
        try:
            rows.append(
                NcclRow(
                    size_bytes=int(fields[0]),
                    algbw_gbps=float(fields[-2]),
                    busbw_gbps=float(fields[-1]),
                )
            )
        except ValueError:
            continue
    return rows
