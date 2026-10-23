"""Pin repeated-index aggregation and affine LayerNorm numerical contracts."""

import unittest

import numpy as np

from graphcast_bench.graph_operator_bench import make_operation


class GraphOperatorTests(unittest.TestCase):
    def test_scatter_keeps_duplicate_contributions_and_empty_nodes(self):
        fn = make_operation("scatter", node_count=4)
        result = np.asarray(
            fn(
                np.array([[1.0, 2.0], [3.0, 4.0], [-2.0, 5.0]], np.float32),
                np.array([2, 2, 0], np.int32),
            )
        )
        np.testing.assert_array_equal(result, [[-2, 5], [0, 0], [4, 6], [0, 0]])

    def test_gather_preserves_index_order_and_duplicates(self):
        fn = make_operation("gather")
        result = np.asarray(
            fn(np.array([[1, 2], [3, 4]], np.float32), np.array([1, 0, 1], np.int32))
        )
        np.testing.assert_array_equal(result, [[3, 4], [1, 2], [3, 4]])

    def test_layer_norm_includes_epsilon_scale_offset_and_constant_rows(self):
        fn = make_operation("layer_norm")
        x = np.array([[1, 2, 3, 4], [2, 2, 2, 2]], np.float32)
        scale = np.array([1, 2, 3, 4], np.float32)
        offset = np.array([-1, 2, -3, 4], np.float32)
        expected = np.array([[-1.5, -0.5, 0.5, 1.5], [0, 0, 0, 0]])
        expected[0] /= np.sqrt(1.25 + 1e-5)
        expected = expected * scale + offset
        np.testing.assert_allclose(np.asarray(fn(x, scale, offset)), expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
