"""Bounded reading of generated reports for display, separate from job inputs."""

import json
from pathlib import Path

from .jobs import object_pairs

_MAX_REPORT_BYTES = 64 * 1024 * 1024


def read_report(path):
    """Read a report without relaxing configuration or artifact admission limits.

    The limit bounds file bytes, not the memory used by decoded JSON objects.
    """
    path = Path(path)
    error = f"JSON report exceeds 64 MiB: {path.name}"
    if path.stat().st_size > _MAX_REPORT_BYTES:
        raise ValueError(error)
    with path.open("rb") as stream:
        # Enforce the bound even if the file grows after the stat check.
        data = stream.read(_MAX_REPORT_BYTES + 1)
    if len(data) > _MAX_REPORT_BYTES:
        raise ValueError(error)
    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value: {x}")),
    )
