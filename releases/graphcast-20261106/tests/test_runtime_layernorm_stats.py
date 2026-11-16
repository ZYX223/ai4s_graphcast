"""CPU validation guards for the fixed centered LayerNorm marker candidate."""

import unittest
from types import SimpleNamespace

import numpy as np

import graphcast_bench.optimizations.training.layernorm_stats as candidate


def module(**changes):
    values = dict(
        axis=(-1,),
        param_axis=(-1,),
        create_scale=True,
        create_offset=True,
        use_fast_variance=False,
        eps=1e-5,
    )
    values.update(changes)
    return SimpleNamespace(**values)


class FixedLayerNormGuards(unittest.TestCase):
    def test_original_fixed_configuration_allowed(self):
        candidate.validate_spec(module(), np.zeros((3, 1, 512), np.float32), None, None)

    def test_arithmetic_precision_or_affine_changes_rejected(self):
        for changes in [
            dict(axis=(1,)),
            dict(param_axis=(0,)),
            dict(create_scale=False),
            dict(create_offset=False),
            dict(use_fast_variance=True),
            dict(eps=1e-3),
        ]:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    candidate.validate_spec(
                        module(**changes), np.zeros((3, 1, 512), np.float32), None, None
                    )
        for x in [
            np.zeros((3, 512), np.float32),
            np.zeros((3, 2, 512), np.float32),
            np.zeros((3, 1, 256), np.float32),
            np.zeros((3, 1, 512), np.float64),
        ]:
            with self.assertRaises(ValueError):
                candidate.validate_spec(module(), x, None, None)
        x = np.zeros((3, 1, 512), np.float32)
        with self.assertRaises(ValueError):
            candidate.validate_spec(module(), x, np.ones(512), None)
        with self.assertRaises(ValueError):
            candidate.validate_spec(module(), x, None, np.zeros(512))


if __name__ == "__main__":
    unittest.main()
