"""Nothing under test may write to the operator's home directory, and
nothing under test may depend on a tool the operator happens to have."""
import os

import pytest


@pytest.fixture(autouse=True)
def _blocklist_in_tmp(tmp_path, monkeypatch):
    from launch import orchestrate
    from launch.blocklist import Blocklist

    monkeypatch.setattr(orchestrate, "BLOCKLIST",
                        Blocklist(tmp_path / "blocked.json"))


@pytest.fixture(autouse=True)
def _vastai_on_path(tmp_path, monkeypatch):
    """A stand-in `vastai` executable, so no test needs the operator's own.

    Every test that reaches vast.py stubs `subprocess.run` and asserts on the
    argv that would have been passed — none of them execute the CLI. But
    `vastai_bin()` runs first and resolves a real path, so the whole launch and
    orchestration suite silently depended on the binary happening to sit in the
    developer's venv. It does here; it does not on a CI runner, where these
    fourteen tests had been failing on every push since the workflow was added.

    A dummy on PATH rather than a stubbed `vastai_bin` keeps the resolution
    logic itself under test — it is the reaper's way of finding its own tool
    while a GPU bills by the second, and stubbing it out would leave the money
    guard's lookup untested."""
    fake = tmp_path / "fakebin"
    fake.mkdir()
    stub = fake / "vastai"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake), prepend=os.pathsep)
