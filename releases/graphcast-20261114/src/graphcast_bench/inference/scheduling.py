"""Keep warmup outside measured rounds, with explicit one-based round IDs."""

from graphcast_bench.delivery.options import positive_integer


def run_rounds(repeats, warmup_scope, warmup, measure):
    positive_integer("repeats", repeats)
    if warmup_scope not in ("once", "per-repeat"):
        raise ValueError("Unknown warmup scope")
    for round_id in range(1, repeats + 1):
        if round_id == 1 or warmup_scope == "per-repeat":
            warmup(round_id)
        measure(round_id)
