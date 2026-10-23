"""Catch missing calls, wrong shape accounting and unrecognized checkpoint weights."""
import unittest

from graphcast_bench.operator_catalog import build_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.weights = {"layers": [
            {"module": "a", "shape": [4, 8], "sha256": {"row_major": "aa", "transpose": "at"}},
            {"module": "b", "shape": [4, 8], "sha256": {"row_major": "bb", "transpose": "bt"}},
        ]}
        self.trace = [
            dict(label="fp32:1", layer="a", m=8, n=12, k=4, status=0, weight_sha256="aa"),
            dict(label="fp32:1", layer="b", m=8, n=12, k=4, status=0, weight_sha256="bt"),
        ]
        self.hlo = '\n'.join(
            '%%gemm%d = (f32[12,8]{1,0}, s8[0]{0}) custom-call(%%a, %%b), '
            'custom_call_target="__cublas$gemm", backend_config={"gemm_backend_config":'
            '{"lhs_stride":"48", "rhs_stride":"32"}}' % i for i in range(2)
        )

    def test_counts_calls_not_rounds_and_calculates_traffic(self):
        result = build_catalog(self.trace + [dict(self.trace[0], label="fp32:2")],
                               self.hlo, self.weights, "fp32:1")
        row = result["gemm_shapes"][0]
        self.assertEqual(row["calls_per_step"], 2)
        self.assertEqual(row["flops_per_call"], 768)
        self.assertEqual(row["rne_read_write_bytes_per_call"], 640)
        self.assertEqual(row["activation_shape"], [12, 4])

    def test_rejects_incomplete_step(self):
        with self.assertRaises(ValueError):
            build_catalog(self.trace[:1], self.hlo, self.weights, "fp32:1")

    def test_rejects_unknown_weight(self):
        bad = [dict(self.trace[0], weight_sha256="other"), self.trace[1]]
        with self.assertRaises(ValueError):
            build_catalog(bad, self.hlo, self.weights, "fp32:1")

    def test_rejects_hlo_runtime_shape_mismatch(self):
        with self.assertRaises(ValueError):
            build_catalog(self.trace, self.hlo.replace('"48"', '"60"'), self.weights, "fp32:1")

    def test_rejects_duplicate_layer_even_if_call_count_matches(self):
        with self.assertRaises(ValueError):
            build_catalog([self.trace[0], self.trace[0]], self.hlo, self.weights, "fp32:1")


if __name__ == "__main__":
    unittest.main()
