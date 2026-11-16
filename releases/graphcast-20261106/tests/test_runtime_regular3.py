"""Verify complete static adjacency, axis handling and unchanged fallback."""

import unittest

import numpy as np

from graphcast_bench.optimizations.training.regular3 import regular3_sum, validate_regular3


class RegularThreeChecks(unittest.TestCase):
    def test_full_ordered_coverage_required(self):
        good = np.repeat(np.arange(7), 3)
        validate_regular3(good, 7)
        for bad in [
            good[::-1],
            good[:-1],
            np.zeros_like(good),
            good.astype(float),
            good.reshape(7, 3),
        ]:
            with self.assertRaises(ValueError):
                validate_regular3(bad, 7)
        with self.assertRaises(ValueError):
            validate_regular3(good, 8)

    def test_all_three_messages_and_batch_axis_preserved(self):
        ids = np.repeat(np.arange(7), 3)
        data = np.arange(21 * 2 * 5, dtype=np.float32).reshape(21, 2, 5)
        expected = np.zeros((7, 2, 5), dtype=np.float32)
        np.add.at(expected, ids, data)
        actual = regular3_sum(data, 7, np)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (7, 2, 5))

    def test_rounding_error_is_bounded_for_noninteger_messages(self):
        rng = np.random.RandomState(13)
        data = rng.normal(size=(111, 1, 19)).astype(np.float32)
        expected = np.zeros((37, 1, 19), dtype=np.float32)
        np.add.at(expected, np.repeat(np.arange(37), 3), data)
        np.testing.assert_allclose(regular3_sum(data, 37, np), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
