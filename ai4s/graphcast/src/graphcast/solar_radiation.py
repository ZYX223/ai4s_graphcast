"""Torch-runtime adapter for WeatherNext's JAX solar-radiation dependency.

WeatherNext ``graphcast.data_utils`` imports ``graphcast.solar_radiation`` at
module import time.  H100, MetaX, and MThreads training images are PyTorch-only,
so this adapter exposes the two upstream entry points through the existing
NumPy port of the same pinned equations.  No other WeatherNext source file is
modified.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xa

from graphcast_compat.official_solar import toa_incident_solar_radiation


def get_toa_incident_solar_radiation(
    timestamps: Sequence[Any],
    latitude: np.ndarray,
    longitude: np.ndarray,
    tsi_data: xa.DataArray | None = None,
    integration_period: Any = pd.Timedelta(hours=1),
    num_integration_bins: int = 360,
    use_jit: bool = False,
) -> np.ndarray:
    """Match the upstream API using the pinned NumPy equation port."""

    del use_jit
    if tsi_data is not None:
        raise NotImplementedError(
            "custom TSI data is unavailable in the PyTorch solar adapter"
        )
    period = pd.Timedelta(integration_period)
    period_seconds = int(period / pd.Timedelta(seconds=1))
    return toa_incident_solar_radiation(
        np.asarray(timestamps).astype("datetime64[ns]"),
        latitudes=np.asarray(latitude),
        longitudes=np.asarray(longitude),
        integration_period_seconds=period_seconds,
        num_integration_bins=num_integration_bins,
    )


def get_toa_incident_solar_radiation_for_xarray(
    data_array_like: xa.DataArray | xa.Dataset,
    tsi_data: xa.DataArray | None = None,
    integration_period: Any = pd.Timedelta(hours=1),
    num_integration_bins: int = 360,
    use_jit: bool = False,
) -> xa.DataArray:
    """Match WeatherNext's Xarray wrapper and preserve compatible coordinates."""

    missing_dims = {"lat", "lon"} - set(data_array_like.dims)
    if missing_dims:
        raise ValueError(
            f"'{missing_dims}' dimensions are missing in `data_array_like`."
        )
    missing_coords = {"datetime", "lat", "lon"} - set(
        data_array_like.coords
    )
    if missing_coords:
        raise ValueError(
            f"'{missing_coords}' coordinates are missing in `data_array_like`."
        )

    if "time" in data_array_like.dims:
        timestamps = data_array_like.coords["datetime"].data
    else:
        timestamps = [data_array_like.coords["datetime"].data.item()]
    radiation = get_toa_incident_solar_radiation(
        timestamps=timestamps,
        latitude=data_array_like.coords["lat"].data,
        longitude=data_array_like.coords["lon"].data,
        tsi_data=tsi_data,
        integration_period=integration_period,
        num_integration_bins=num_integration_bins,
        use_jit=use_jit,
    )
    if "time" in data_array_like.dims:
        output = xa.DataArray(radiation, dims=("time", "lat", "lon"))
    else:
        output = xa.DataArray(radiation[0], dims=("lat", "lon"))
    for name, coordinate in data_array_like.coords.items():
        if set(coordinate.dims).issubset(set(output.dims)):
            output.coords[name] = coordinate
    return output
