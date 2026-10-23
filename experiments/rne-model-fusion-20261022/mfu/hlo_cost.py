"""Backend FLOP estimate from the exact, already-profiled optimized HLO."""
import json
from pathlib import Path
import jax
from jaxlib import xla_client
assert len(jax.devices('gpu'))==1
client=jax.devices('gpu')[0].client
result={}
for variant in ['baseline','fused']:
 text=(Path('/evidence')/('profile-'+variant+'-r1/run/optimized-hlo.txt')).read_text()
 module=xla_client._xla.hlo_module_from_text(text)
 result[variant]=xla_client._xla.hlo_module_cost_analysis(client,module)
Path('/output/hlo-cost.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
