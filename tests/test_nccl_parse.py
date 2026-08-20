import pytest
from gppb.nccl_parse import parse_nccl_output

SAMPLE = """
# nThread 1 nGpus 1 minBytes 8 maxBytes 1073741824
#
#       size         count      type   redop    root     time   algbw   busbw
#        (B)    (elements)                             (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    32.11    0.00    0.00
     1048576        262144     float     sum      -1    45.20   23.20   43.50
  1073741824     268435456     float     sum      -1  4521.10  237.50  445.31
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 162.937
"""


def test_parses_every_data_row():
    rows = parse_nccl_output(SAMPLE)
    assert len(rows) == 3


def test_parses_sizes_and_bandwidths():
    rows = parse_nccl_output(SAMPLE)
    assert rows[-1].size_bytes == 1073741824
    assert rows[-1].algbw_gbps == 237.50
    assert rows[-1].busbw_gbps == 445.31


def test_skips_comment_and_header_lines():
    rows = parse_nccl_output(SAMPLE)
    assert all(r.size_bytes > 0 for r in rows)


def test_empty_output_yields_no_rows():
    assert parse_nccl_output("# nothing here\n") == []
