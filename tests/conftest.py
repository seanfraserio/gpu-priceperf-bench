"""Nothing under test may write to the operator's home directory."""
import pytest


@pytest.fixture(autouse=True)
def _blocklist_in_tmp(tmp_path, monkeypatch):
    from launch import orchestrate
    from launch.blocklist import Blocklist

    monkeypatch.setattr(orchestrate, "BLOCKLIST",
                        Blocklist(tmp_path / "blocked.json"))
