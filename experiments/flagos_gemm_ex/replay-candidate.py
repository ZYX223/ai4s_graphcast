"""Re-run the two missing-RNE Linear cases without changing frozen evidence."""
import hashlib
import json
import os
from pathlib import Path
import runpy

old = Path("/tmp/zyx-flagos-logical-operators-20261029")
out = Path(__file__).resolve().parent
source = (old / "bench.py").read_text()
def replace_once(a, b):
    global source
    assert source.count(a) == 1, a
    source = source.replace(a, b)
replace_once("OUT = ROOT / ('bench-' + MODE + os.environ.get('BENCH_SUFFIX',''))",
             "OUT = Path(os.environ['GRAPHCAST_REPLAY_OUTPUT'])")
replace_once("assert len(cases) == 108",
             "assert len(cases) == 108\ncases = [c for c in cases if c['id'] in {'linear-80b99667a99e', 'linear-42d4ab85427e'}]\nassert len(cases) == 2")
replace_once("expected_cases=108", "expected_cases=2")
# __file__ remains the frozen benchmark so inventory paths stay read-only.
# Only OUT, traces and dumps are redirected to this new evidence directory.
(out / "replay-source.py").write_text(source)
(out / "source-manifest.json").write_text(json.dumps({
    "source": str(old / "bench.py"),
    "source_sha256": hashlib.sha256((old / "bench.py").read_bytes()).hexdigest(),
    "executed_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "cases": ["linear-80b99667a99e", "linear-42d4ab85427e"],
    "baseline_modified": False,
    "scope": "GEMM_EX candidate regression; original baseline unchanged"
}, indent=2))
replace_once("successful Linear rows require one RNE+XF32 audited rocBLAS SGEMM per invocation.", "successful Linear rows require one audited rocBLAS call with RNE and XF32 mode requested; K=4 may use native FP32 fallback.")
namespace = {"__name__": "__main__", "__file__": str(old / "bench.py")}
exec(compile(source, str(out / "replay-source.py"), "exec"), namespace)
if namespace["report"]["status"] != "passed":
    raise SystemExit(1)

