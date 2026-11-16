"""Require successful operand verification in this executable's warmup.

The C router aborts on any non-rounded finite GEMM input in verify mode.
The frozen trainer synchronizes warmup before selecting its first update.
No numerical operation or optimizer state is changed by this gate.
"""


class WarmupGate:
    def __init__(self):
        self.before = None
        self.verified_calls = 0

    def select(self, label, counts):
        if self.verified_calls:
            return "skip"
        if label == "initialization":
            return "verify"  # C router keeps autotuning baseline RNE.
        if label == "warmup":
            self.before = dict(counts)
            return "verify"
        if not label.startswith("update-") or self.before is None:
            raise ValueError("Cannot skip RNE without synchronized verified warmup")
        calls = counts["calls"] - self.before["calls"]
        rounded = counts["rounded"] - self.before["rounded"]
        if calls <= 0 or rounded != calls:
            raise ValueError("Warmup RNE coverage incomplete")
        self.verified_calls = calls
        return "skip"
