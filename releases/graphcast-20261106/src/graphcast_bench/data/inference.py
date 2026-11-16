"""Load portable inference inputs with explicit timedelta decoding."""

import numpy as np
import xarray as xr


def load_input_dataset(path):
    # Older pinned xarray decodes timedeltas from units only. Newer xarray also
    # interprets dtype metadata; explicitly retain the original units behavior.
    import inspect

    from xarray.coding.times import CFTimedeltaCoder

    decoder = True
    if "decode_via_units" in inspect.signature(CFTimedeltaCoder).parameters:
        decoder = CFTimedeltaCoder(decode_via_units=True, decode_via_dtype=False)
    with xr.open_dataset(path, decode_timedelta=decoder) as source:
        data = source.load()
    for variable in data.variables.values():
        marker = variable.attrs.get("dtype")
        if variable.dtype.kind == "m" and isinstance(marker, str) and np.dtype(marker).kind == "m":
            variable.attrs.pop("dtype")
    return data
