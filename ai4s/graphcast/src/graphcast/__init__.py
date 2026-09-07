"""Runtime bridge to the pinned Google DeepMind WeatherNext source tree.

The actual ``graphcast`` modules are loaded from
``src/third_party/weathernext/graphcast``.  This small package exists
only so Torch-only accelerator images can replace the JAX solar-radiation
dependency while importing every other compatible upstream module unchanged.
"""

from __future__ import annotations

from pathlib import Path


UPSTREAM_COMMIT = "08cf73625c9d12bd9aaa038868bcb2fe488f2a22"
_UPSTREAM_PACKAGE = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "weathernext"
    / "graphcast"
)
if not _UPSTREAM_PACKAGE.is_dir():
    raise ImportError(
        "pinned WeatherNext GraphCast source is missing: "
        f"{_UPSTREAM_PACKAGE}"
    )

# Keep this bridge first so graphcast.solar_radiation resolves to the
# Torch-compatible adapter next to this file. All remaining modules resolve to
# the byte-identical pinned WeatherNext snapshot.
__path__.append(str(_UPSTREAM_PACKAGE))
