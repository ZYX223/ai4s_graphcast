import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from graphcast_bench.data.inference import load_input_dataset


class InputTests(unittest.TestCase):
    def test_timedelta_coordinates_survive_netcdf_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.nc"
            source = xr.Dataset(
                {"x": ("time", np.array([1, 2], np.float32))},
                coords={"time": np.array([-6, 0], dtype="timedelta64[h]")},
            )
            source.to_netcdf(path)
            loaded = load_input_dataset(path)
            np.testing.assert_array_equal(source.time.values, loaded.time.values)
            np.testing.assert_array_equal(source.x.values, loaded.x.values)
            self.assertEqual(loaded.x.dtype, np.float32)
