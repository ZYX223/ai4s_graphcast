import unittest

from graphcast_bench.optimizations.rne.warmup import WarmupGate


class GateTests(unittest.TestCase):
    def test_reject_missing_warmup(self):
        g = WarmupGate()
        with self.assertRaises(ValueError):
            g.select("update-01", {"calls": 5, "rounded": 5})

    def test_reject_incomplete_coverage(self):
        for after in ({"calls": 1, "rounded": 0}, {"calls": 0, "rounded": 0}):
            g = WarmupGate()
            g.select("warmup", {"calls": 0, "rounded": 0})
            with self.assertRaises(ValueError):
                g.select("update-01", after)

    def test_per_stage_warmup_then_skip(self):
        g = WarmupGate()
        self.assertEqual(g.select("initialization", {"calls": 10, "rounded": 10}), "verify")
        self.assertEqual(g.select("warmup", {"calls": 10, "rounded": 10}), "verify")
        self.assertEqual(g.select("update-25", {"calls": 400, "rounded": 400}), "skip")
        self.assertEqual(g.verified_calls, 390)
        self.assertEqual(g.select("microbatch0", {"calls": 400, "rounded": 400}), "skip")
        self.assertEqual(g.select("update-26", {"calls": 790, "rounded": 790}), "skip")

    def test_reject_label_before_switch(self):
        g = WarmupGate()
        with self.assertRaises(ValueError):
            g.select("microbatch0", {"calls": 0, "rounded": 0})


if __name__ == "__main__":
    unittest.main()
