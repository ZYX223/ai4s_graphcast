"""Build an experimental wrapper copy; baseline sources remain read-only.

The only candidate change skips the activation's HIP RNE after JAX has already
rounded it. Weight identity, weight RNE, native rocBLAS, and mode checks stay.
This opt-in switch is ONLY for independently verified single-Linear probes.
"""
from pathlib import Path
import difflib
import json
import hashlib

src = Path('/repo/kernels/xf32_rne/xf32_rne_router.cc').read_text()
patched = src.replace('bool initialized=false;', '''bool initialized=false;
bool activation_pre_rounded=false;
uint64_t skipped_activation_calls=0;''')
marker = 'extern "C" int rocblas_sgemm(void* handle'
api = '''extern "C" void experiment_activation_pre_rounded(int value) {
  std::lock_guard<std::mutex> guard(lock); activation_pre_rounded=value!=0;
}
extern "C" uint64_t experiment_skipped_activation_calls() {
  std::lock_guard<std::mutex> guard(lock); return skipped_activation_calls;
}

'''
assert patched.count(marker) == 1
patched = patched.replace(marker,api+marker)
old = '  if(!matched)matched=identify(b,'
assert patched.count(old) == 1
patched = patched.replace(old,'  bool weight_is_a=matched;\n'+old)
old = '''    aa=graphcast_round_scratch(a,trans_a==111?m:k,trans_a==111?k:m,lda,stream,0);
    bb=graphcast_round_scratch(b,trans_b==111?k:n,trans_b==111?n:k,ldb,stream,1);'''
new = '''    if(!activation_pre_rounded || weight_is_a)
      aa=graphcast_round_scratch(a,trans_a==111?m:k,trans_a==111?k:m,lda,stream,0);
    if(!activation_pre_rounded || !weight_is_a)
      bb=graphcast_round_scratch(b,trans_b==111?k:n,trans_b==111?n:k,ldb,stream,1);
    if(activation_pre_rounded)++skipped_activation_calls;'''
assert patched.count(old) == 1
patched = patched.replace(old,new)
out = Path('/output/native'); out.mkdir()
(out/'xf32_rne_router.cc').write_text(patched)
(out/'router.patch').write_text(''.join(difflib.unified_diff(src.splitlines(True),patched.splitlines(True),fromfile='baseline',tofile='experiment')))
(out/'source-sha256.json').write_text(json.dumps({'baseline':hashlib.sha256(src.encode()).hexdigest(),'experiment':hashlib.sha256(patched.encode()).hexdigest()},indent=2)+'\n')
