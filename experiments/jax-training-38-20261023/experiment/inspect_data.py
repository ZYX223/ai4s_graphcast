import json
from pathlib import Path
import xarray as xr,numpy as np
p=Path('/training-data/hres-20220101-025-13-12.nc')
d=xr.open_dataset(p,decode_timedelta=True)
print(dict(d.sizes));print(d)
r=dict(sizes=dict(d.sizes),variables={k:dict(dims=list(v.dims),shape=list(v.shape),dtype=str(v.dtype)) for k,v in d.data_vars.items()})
Path('/output/data-inspection.json').write_text(json.dumps(r,indent=2))
