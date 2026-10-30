"""The weekly report's final +240h output-consistency metric, in float64."""

import numpy as np
import xarray as xr

VARIABLES = (
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
    "vertical_velocity",
)
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def final_frame(ds, initialization="2021-01-01T06:00:00", lead_hours=240):
    if ds.sizes.get("batch") != 1 or ds.sizes.get("time") != 1:
        raise ValueError("Expected one final forecast, batch=1 and time=1")
    expected_time = np.array([lead_hours], dtype="timedelta64[h]").astype("timedelta64[ns]")
    expected_datetime = (np.datetime64(initialization, "ns") + expected_time)[None]
    if not np.array_equal(ds.time.values, expected_time):
        raise ValueError("Prediction is not the requested final lead")
    if "datetime" not in ds.coords or not np.array_equal(ds.datetime.values, expected_datetime):
        raise ValueError("Prediction initialization differs from the fixed case")
    if "level" not in ds.coords or not np.array_equal(ds.level.values, LEVELS):
        raise ValueError("Pressure levels differ from the 13-level contract")
    for coordinate in ("lat", "lon"):
        values = np.asarray(ds[coordinate])
        if not np.isfinite(values).all() or np.unique(values).size != values.size:
            raise ValueError("Invalid grid coordinates: " + coordinate)
    for name in VARIABLES:
        if name not in ds or set(ds[name].dims) != {"batch", "time", "level", "lat", "lon"}:
            raise ValueError("Missing/malformed atmospheric variable: " + name)
        if not ds[name].attrs.get("units"):
            raise ValueError("Missing physical units: " + name)
    for name, variable in ds.data_vars.items():
        if variable.dtype != np.float32 or not np.isfinite(variable.values).all():
            raise ValueError("Expected finite FP32 output: " + name)
    return ds.isel(batch=0, time=0, drop=True)


def validate_case_grid_units(ds, case):
    """Require the pinned global grid and units, even if two wrong files agree."""
    expected = {
        "lat": np.linspace(90.0, -90.0, case["grid"]["lat"]),
        "lon": np.arange(case["grid"]["lon"]) * (360.0 / case["grid"]["lon"]),
    }
    for coordinate, values in expected.items():
        if coordinate not in ds.coords or not np.array_equal(ds[coordinate].values, values):
            raise ValueError("Prediction differs from fixed case grid: " + coordinate)
    for name in VARIABLES:
        if name not in ds or ds[name].attrs.get("units") != case["units"][name]:
            raise ValueError("Prediction differs from fixed case units: " + name)


def compare(
    candidate, reference, initialization="2021-01-01T06:00:00", lead_hours=240, *, case=None
):
    if case is not None:
        initialization, lead_hours = case["initialization_time"], case["lead_hours"]
        for ds in (candidate, reference):
            validate_case_grid_units(ds, case)
    a, b = [final_frame(x, initialization, lead_hours) for x in (candidate, reference)]
    for coord in ("lat", "lon", "level"):
        if not np.array_equal(a[coord].values, b[coord].values):
            raise ValueError("Candidate/reference coordinate mismatch: " + coord)
    rows = []
    for name in VARIABLES:
        if a[name].attrs["units"] != b[name].attrs["units"]:
            raise ValueError("Candidate/reference units mismatch: " + name)
        for level in LEVELS:
            av = np.asarray(a[name].sel(level=level).transpose("lat", "lon"), dtype=np.float64)
            bv = np.asarray(b[name].sel(level=level).transpose("lat", "lon"), dtype=np.float64)
            diff = av - bv
            denominator = float(np.mean(np.abs(bv)))
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError("Invalid reference denominator")
            mae = float(np.mean(np.abs(diff)))
            rows.append(
                {
                    "variable": name,
                    "level_hpa": level,
                    "mae": mae,
                    "relative_error_percent": mae / denominator * 100,
                    "rmse": float(np.sqrt(np.mean(diff * diff))),
                    "max_abs": float(np.max(np.abs(diff))),
                    "bias": float(np.mean(diff)),
                    "reference_mean_abs": denominator,
                }
            )
    overall = float(np.mean([x["relative_error_percent"] for x in rows]))
    z = next(x for x in rows if x["variable"] == "geopotential" and x["level_hpa"] == 500)
    return {
        "reference": "official_jax_fp32",
        "overall_mean_percent": overall,
        "z500_percent": z["relative_error_percent"],
        "z500_mae_m": z["mae"] / 9.80665,
        "z500_rmse_m": z["rmse"] / 9.80665,
        "metrics_within_limits": overall < 5 and z["relative_error_percent"] < 5,
        "layers": rows,
    }


def load_reference(path, case):
    with np.load(path, allow_pickle=False) as source:
        fields = {
            name: (
                ("batch", "time", "level", "lat", "lon"),
                source[name][None, None],
                {"units": case["units"][name]},
            )
            for name in VARIABLES
        }
        lead = np.array([case["lead_hours"]], dtype="timedelta64[h]").astype("timedelta64[ns]")
        return xr.Dataset(
            fields,
            coords={
                "batch": [0],
                "time": lead,
                "level": source["level"],
                "lat": source["lat"],
                "lon": source["lon"],
                "datetime": (
                    ("batch", "time"),
                    (np.datetime64(case["initialization_time"], "ns") + lead)[None],
                ),
            },
        )
