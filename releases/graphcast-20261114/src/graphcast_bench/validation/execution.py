"""A custom run's execution evidence is distinct from reference acceptance."""

import json
from pathlib import Path


def write_summary(output, options, details):
    """Call only after the adapter has validated every requested stage."""
    result = dict(
        schema="graphcast.execution_summary.v1", options=options,
        execution=dict(status="passed", details=details),
        accuracy=dict(status="not_evaluated", reason="Custom execution has no frozen-reference acceptance"),
        reproduction=dict(status="not_evaluated", reason="Use a fixed preset for frozen reproduction checks"),
    )
    target = Path(output) / "execution-summary.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    return result
