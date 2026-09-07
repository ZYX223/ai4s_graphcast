# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NumPy port of GraphCast's ERA5-compatible TOA solar radiation.

The equations and TSI table are derived from
``google-deepmind/weathernext/graphcast/solar_radiation.py`` at commit
``08cf73625c9d12bd9aaa038868bcb2fe488f2a22``.  This port removes the JAX and
Chex runtime dependencies and evaluates latitude chunks to bound peak memory.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


_JULIAN_YEAR_LENGTH_IN_DAYS = np.float32(365.25)
_J2000_EPOCH = 2451545.0
_SECONDS_PER_DAY = np.float32(60 * 60 * 24)
_ERA5_TSI_TIME = np.arange(1951.5, 2035.5, 1.0)
_ERA5_TSI = 0.9965 * np.asarray(
    [
        1365.7765,
        1365.7676,
        1365.6284,
        1365.6564,
        1365.7773,
        1366.3109,
        1366.6681,
        1366.6328,
        1366.3828,
        1366.2767,
        1365.9199,
        1365.7484,
        1365.6963,
        1365.6976,
        1365.7341,
        1365.9178,
        1366.1143,
        1366.1644,
        1366.2476,
        1366.2426,
        1365.9580,
        1366.0525,
        1365.7991,
        1365.7271,
        1365.5345,
        1365.6453,
        1365.8331,
        1366.2747,
        1366.6348,
        1366.6482,
        1366.6951,
        1366.2859,
        1366.1992,
        1365.8103,
        1365.6416,
        1365.6379,
        1365.7899,
        1366.0826,
        1366.6479,
        1366.5533,
        1366.4457,
        1366.3021,
        1366.0286,
        1365.7971,
        1365.6996,
        1365.6121,
        1365.7399,
        1366.1021,
        1366.3851,
        1366.6836,
        1366.6022,
        1366.6807,
        1366.2300,
        1366.0480,
        1365.8545,
        1365.8107,
        1365.7240,
        1365.6918,
        1365.6121,
        1365.7399,
        1366.1021,
        1366.3851,
        1366.6836,
        1366.6022,
        1366.6807,
        1366.2300,
        1366.0480,
        1365.8545,
        1365.8107,
        1365.7240,
        1365.6918,
        1365.6121,
        1365.7399,
        1366.1021,
        1366.3851,
        1366.6836,
        1366.6022,
        1366.6807,
        1366.2300,
        1366.0480,
        1365.8545,
        1365.8107,
        1365.7240,
        1365.6918,
    ],
    dtype=np.float64,
)


def era5_tsi_for_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """Interpolate GraphCast's ERA5-compatible total solar irradiance table."""

    index = pd.DatetimeIndex(np.asarray(timestamps).astype("datetime64[ns]"))
    dates = pd.DatetimeIndex(index.date)
    day_fraction = (index - dates) / pd.Timedelta(days=1)
    year_length = 365 + index.is_leap_year.astype(np.int64)
    year_fraction = (index.dayofyear - 1 + day_fraction) / year_length
    fractional_year = index.year + year_fraction
    return np.interp(
        np.asarray(fractional_year, dtype=np.float64),
        _ERA5_TSI_TIME,
        _ERA5_TSI,
    )


def _j2000_days(timestamp: np.datetime64) -> np.float32:
    return np.float32(
        pd.Timestamp(timestamp.astype("datetime64[ns]")).to_julian_date()
        - _J2000_EPOCH
    )


def _orbital_parameters(
    j2000_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = j2000_days / _JULIAN_YEAR_LENGTH_IN_DAYS
    rotational_phase = np.mod(j2000_days, np.float32(1.0))
    rel = np.float32(1.7535) + np.float32(6.283076) * theta
    rem = np.float32(6.240041) + np.float32(6.283020) * theta
    rlls = np.float32(4.8951) + np.float32(6.283076) * theta

    one = np.ones_like(theta)
    sin_rel = np.sin(rel)
    cos_rel = np.cos(rel)
    sin_two_rel = np.sin(np.float32(2.0) * rel)
    cos_two_rel = np.cos(np.float32(2.0) * rel)
    sin_two_rlls = np.sin(np.float32(2.0) * rlls)
    cos_two_rlls = np.cos(np.float32(2.0) * rlls)
    sin_four_rlls = np.sin(np.float32(4.0) * rlls)
    sin_rem = np.sin(rem)
    sin_two_rem = np.sin(np.float32(2.0) * rem)

    rllls = np.stack(
        [one, theta, sin_rel, cos_rel, sin_two_rel, cos_two_rel],
        axis=-1,
    ) @ np.asarray(
        [4.8952, 6.283320, -0.0075, -0.0326, -0.0003, 0.0002],
        dtype=np.float32,
    )
    sin_declination = np.sin(np.float32(0.409093)) * np.sin(rllls)
    cos_declination = np.sqrt(
        np.float32(1.0) - sin_declination**2
    )
    equation_of_time_seconds = np.stack(
        [
            sin_two_rlls,
            sin_rem,
            sin_rem * cos_two_rlls,
            sin_four_rlls,
            sin_two_rem,
        ],
        axis=-1,
    ) @ np.asarray([591.8, -459.4, 39.5, -12.7, -4.8], dtype=np.float32)
    solar_distance_au = np.stack([one, sin_rel, cos_rel], axis=-1) @ np.asarray(
        [1.0001, -0.0163, 0.0037],
        dtype=np.float32,
    )
    return (
        rotational_phase,
        sin_declination,
        cos_declination,
        equation_of_time_seconds,
        solar_distance_au,
    )


def _integrated_one_timestamp(
    timestamp: np.datetime64,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    tsi: float,
    *,
    integration_period_seconds: int,
    num_integration_bins: int,
    latitude_chunk_size: int,
) -> np.ndarray:
    offsets_days = np.linspace(
        -integration_period_seconds / float(24 * 60 * 60),
        0.0,
        num_integration_bins + 1,
        dtype=np.float32,
    )
    j2000_days = _j2000_days(timestamp) + offsets_days
    (
        rotational_phase,
        sin_declination,
        cos_declination,
        equation_of_time_seconds,
        solar_distance_au,
    ) = _orbital_parameters(j2000_days)
    solar_time = (
        rotational_phase
        + equation_of_time_seconds / _SECONDS_PER_DAY
    )
    longitudes_rad = np.deg2rad(longitudes).astype(np.float32)
    hour_angle = (
        np.float32(2.0 * np.pi) * solar_time[:, np.newaxis]
        + longitudes_rad[np.newaxis, :]
    )
    longitude_cosine = np.cos(hour_angle)
    solar_factor = (
        np.float32(1.0) / solar_distance_au
    ) ** np.float32(2.0)
    output = np.empty(
        (len(latitudes), len(longitudes)),
        dtype=np.float32,
    )
    dx_seconds = np.float32(
        integration_period_seconds / num_integration_bins
    )
    for start in range(0, len(latitudes), latitude_chunk_size):
        stop = min(start + latitude_chunk_size, len(latitudes))
        latitude_rad = np.deg2rad(latitudes[start:stop]).astype(np.float32)
        sin_latitude = np.sin(latitude_rad)
        cos_latitude = np.cos(latitude_rad)
        sin_altitude = (
            cos_latitude[np.newaxis, :, np.newaxis]
            * cos_declination[:, np.newaxis, np.newaxis]
            * longitude_cosine[:, np.newaxis, :]
            + sin_latitude[np.newaxis, :, np.newaxis]
            * sin_declination[:, np.newaxis, np.newaxis]
        )
        flux = (
            np.float32(tsi)
            * solar_factor[:, np.newaxis, np.newaxis]
            * np.maximum(sin_altitude, np.float32(0.0))
        )
        output[start:stop] = (
            np.sum(flux[:-1] + flux[1:], axis=0)
            * np.float32(0.5)
            * dx_seconds
        ).astype(np.float32)
    return output


def _integrated_timestamp_job(
    payload: tuple[
        np.datetime64,
        np.ndarray,
        np.ndarray,
        float,
        int,
        int,
        int,
    ],
) -> np.ndarray:
    (
        timestamp,
        latitudes,
        longitudes,
        tsi,
        integration_period_seconds,
        num_integration_bins,
        latitude_chunk_size,
    ) = payload
    return _integrated_one_timestamp(
        timestamp,
        latitudes,
        longitudes,
        tsi,
        integration_period_seconds=integration_period_seconds,
        num_integration_bins=num_integration_bins,
        latitude_chunk_size=latitude_chunk_size,
    )


def toa_incident_solar_radiation(
    timestamps: np.ndarray,
    *,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    integration_period_seconds: int = 60 * 60,
    num_integration_bins: int = 360,
    latitude_chunk_size: int = 8,
    workers: int = 1,
) -> np.ndarray:
    """Compute one-hour integrated TOA radiation in J m^-2.

    The timestamp is the end of the integration interval, matching ERA5 and
    GraphCast.  Output dimensions are ``(time, lat, lon)``.
    """

    timestamps = np.asarray(timestamps).astype("datetime64[ns]")
    latitudes = np.asarray(latitudes, dtype=np.float32)
    longitudes = np.asarray(longitudes, dtype=np.float32)
    if timestamps.ndim != 1 or latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError("timestamps, latitudes, and longitudes must be 1D")
    if (
        not len(timestamps)
        or not len(latitudes)
        or not len(longitudes)
        or np.isnat(timestamps).any()
    ):
        raise ValueError("solar-radiation coordinates must be non-empty and valid")
    if integration_period_seconds <= 0:
        raise ValueError("integration_period_seconds must be positive")
    if num_integration_bins <= 0:
        raise ValueError("num_integration_bins must be positive")
    if latitude_chunk_size <= 0:
        raise ValueError("latitude_chunk_size must be positive")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be positive")

    tsi = era5_tsi_for_timestamps(timestamps)
    result = np.empty(
        (len(timestamps), len(latitudes), len(longitudes)),
        dtype=np.float32,
    )
    payloads = [
        (
            timestamp,
            latitudes,
            longitudes,
            float(tsi[index]),
            integration_period_seconds,
            num_integration_bins,
            latitude_chunk_size,
        )
        for index, timestamp in enumerate(timestamps)
    ]
    worker_count = min(workers, len(payloads))
    if worker_count == 1:
        values = map(_integrated_timestamp_job, payloads)
        for index, value in enumerate(values):
            result[index] = value
        return result

    # The heavy work is in NumPy ufuncs, which release the GIL. Threads avoid
    # copying a full global field through multiprocessing pipes and remain safe
    # when the caller has already initialized xarray/Dask worker threads.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for index, value in enumerate(
            executor.map(
                _integrated_timestamp_job,
                payloads,
                chunksize=1,
            )
        ):
            result[index] = value
    return result
