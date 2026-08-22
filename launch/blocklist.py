"""Machines that have already wasted money, remembered across restarts.

select_offer takes the cheapest qualifying offer, so a bad host with good
advertised numbers is not chosen once — it is chosen every time. One RTX 5090
listing (1508 Mbps, 0.996 reliability, and the lowest price on the board) took
three of four runs and returned nothing."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PATH = Path.home() / ".gppb-blocked-machines.json"


class Blocklist:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)

    def _entries(self) -> dict[str, str]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            # A missing or unreadable blocklist blocks nothing: it must never
            # be the reason a sweep cannot rent anything.
            return {}

    def machines(self) -> set[int]:
        return {int(key) for key in self._entries()}

    def record(self, machine_id: int, reason: str) -> None:
        entries = self._entries()
        entries[str(machine_id)] = reason
        try:
            self.path.write_text(json.dumps(entries, indent=2, sort_keys=True))
        except OSError:
            # Bookkeeping must never be the thing that aborts a paid run.
            pass
