import json
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from graphcast_bench.metrics import LEVELS, VARIABLES, compare


def field(value=100.0):
    return xr.Dataset(
        {
            name: (
                ("batch", "time", "level", "lat", "lon"),
                np.full((1, 1, 13, 2, 3), value, dtype=np.float32),
                {"units": name},
            )
            for name in VARIABLES
        },
        coords={
            "batch": [0],
            "time": [np.timedelta64(240, "h")],
            "level": LEVELS,
            "lat": [1.0, 0.0],
            "lon": [0.0, 1.0, 2.0],
            "datetime": (("batch", "time"), [[np.datetime64("2021-01-11T06:00")]]),
        },
    )


class MetricTests(unittest.TestCase):
    def test_fixed_case_rejects_matching_cropped_forecasts(self):
        case = json.loads((Path(__file__).parents[1] / "configs/case.json").read_text())
        with self.assertRaisesRegex(ValueError, "fixed case grid"):
            compare(field(), field(), case=case)

    def test_fixed_grid_and_units_contract(self):
        from graphcast_bench.metrics import validate_case_grid_units

        case = json.loads((Path(__file__).parents[1] / "configs/case.json").read_text())
        # Metadata-only input keeps this contract test small; final_frame separately
        # enforces dimensions, dtype, finiteness, initialization and forecast lead.
        metadata = xr.Dataset(
            {name: ((), 0, {"units": case["units"][name]}) for name in VARIABLES},
            coords={"lat": np.linspace(90, -90, 721), "lon": np.arange(1440) * 0.25},
        )
        validate_case_grid_units(metadata, case)
        with self.assertRaisesRegex(ValueError, "fixed case grid"):
            validate_case_grid_units(metadata.assign_coords(lon=metadata.lon + 0.25), case)
        with self.assertRaisesRegex(ValueError, "fixed case grid"):
            validate_case_grid_units(metadata.isel(lat=slice(None, None, -1)), case)
        metadata["geopotential"].attrs["units"] = "m"
        with self.assertRaisesRegex(ValueError, "fixed case units"):
            validate_case_grid_units(metadata, case)

    def test_equal_weight_78_layers_and_z500_rmse(self):
        ref, candidate = field(), field(102)
        candidate["geopotential"].loc[dict(level=500)] = 101
        result = compare(candidate, ref)
        self.assertAlmostEqual(result["overall_mean_percent"], (77 * 2 + 1) / 78)
        self.assertAlmostEqual(result["z500_percent"], 1)
        self.assertAlmostEqual(result["z500_rmse_m"], 1 / 9.80665)
        self.assertTrue(result["metrics_within_limits"])
        self.assertEqual(len(result["layers"]), 78)

    def test_threshold_is_strict(self):
        self.assertFalse(compare(field(105), field())["metrics_within_limits"])

    def test_denominator_is_reference(self):
        self.assertAlmostEqual(compare(field(100), field(200))["overall_mean_percent"], 50)

    def test_alignment_and_invalid_values_rejected(self):
        mutations = [
            lambda x: x.assign_coords(time=[np.timedelta64(234, "h")]),
            lambda x: x.assign_coords(lat=[0.0, 1.0]),
            lambda x: x.assign_coords(level=list(reversed(LEVELS))),
            lambda x: x.assign_coords(
                datetime=(("batch", "time"), [[np.datetime64("2021-01-12T06:00")]])
            ),
            lambda x: x.drop_vars("temperature"),
            lambda x: x.astype(np.float64),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                compare(mutate(field()), field())
        for value in [np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare(field(value), field())
        with self.assertRaises(ValueError):
            compare(field(), field(0))
        wrong = field()
        wrong["temperature"].attrs["units"] = "wrong"
        with self.assertRaises(ValueError):
            compare(wrong, field())


if __name__ == "__main__":
    unittest.main()
