"""Prevent CPU/GPU double counting and misleading overlapping-time totals."""

import unittest

from graphcast_bench.profile_summary import summarize_events


class ProfileSummaryTests(unittest.TestCase):
    def test_separates_cpu_api_and_gpu_and_uses_interval_union(self):
        events = [
            dict(ph="X", cat="HIP", name="hipStreamSynchronize", ts=0, dur=100),
            dict(ph="X", cat="HIPOPS", name="round_rne", ts=10, dur=10),
            dict(ph="X", cat="HIPOPS", name="gemm", ts=15, dur=20),
            dict(ph="M", cat="HIPOPS", name="process_name"),
        ]
        result = summarize_events(events)
        self.assertEqual(result["gpu_kernel_count"], 2)
        self.assertEqual(result["gpu_kernel_sum_ms"], 0.03)
        self.assertEqual(result["gpu_kernel_busy_union_ms"], 0.025)
        self.assertEqual(result["host_api_sum_ms"], 0.1)
        self.assertEqual(result["categories"]["rne"]["calls"], 1)

    def test_rejects_cpu_only_trace(self):
        with self.assertRaises(ValueError):
            summarize_events([dict(ph="X", cat="HIP", name="hipLaunchKernel", ts=0, dur=10)])

    def test_rejects_invalid_duration(self):
        with self.assertRaises(ValueError):
            summarize_events([dict(ph="X", cat="HIPOPS", name="gemm", ts=0, dur=-1)])


if __name__ == "__main__":
    unittest.main()
