import json
import math
import sys
from pathlib import Path

out, platform = Path(sys.argv[1]), sys.argv[2]
r = json.loads((out/'report.json').read_text())
assert r['status'] == 'passed'
assert r['workload']['steps'] == 40 and r['workload']['step_hours'] == 6
assert r['workload']['warmup_calls'] == 3
assert r['measurement']['finite'] is True
assert r['measurement']['performance_eligible'] is True
assert math.isfinite(r['measurement']['e2e_seconds']) and r['measurement']['e2e_seconds'] > 0
assert r['precision']['float32_math_mode'] == 'tf32'
if platform == 'hygon':
    assert r['runtime']['flagos_mode'] == 'flaggems'
    assert r['runtime']['flaggems_dispatch']['active_coverage'] == '7/7'
    assert r['workload']['grid2mesh_edge_chunk_size'] == 524288
    assert r['workload']['mesh2grid_edge_chunk_size'] == 524288
    assert r['workload']['index_add_execution'] == 'prevalidated-static'
else:
    assert r['runtime']['flagos_mode'] == 'off'
    assert r['workload']['model_execution_profile']['name'] == 'baseline'
print(json.dumps({'status': 'runner_checked', 'platform': platform,
                  'e2e_seconds': r['measurement']['e2e_seconds'],
                  'steps': r['workload']['steps']}))
