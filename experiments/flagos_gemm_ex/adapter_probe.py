"""GPU regression: real GEMM_EX, rounding, C!=D, transpose, and mode restore."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from native_probe import hip, blas, device, check, V, I

def rounded(x, rne):
    u=x.copy().view('uint32')
    return ((u + np.uint32(0xfff) + ((u >> 13)&1)) & np.uint32(0xffffe000)).view('float32') if rne else (u & np.uint32(0xffffe000)).view('float32')

def main():
    out=Path(os.environ['PROBE_OUTPUT']);out.mkdir(exist_ok=False)
    a=np.asfortranarray(np.random.default_rng(17).uniform(-2,2,(512,512)).astype('float32'))
    # Fingerprint both physical layouts to exercise the transpose ABI.
    weights=[a,np.asfortranarray(a.T)]
    (out/'weights.tsv').write_text(''.join('{} {} 0 probe{}\n'.format(w.nbytes,hashlib.sha256(w.tobytes(order='F')).hexdigest(),i) for i,w in enumerate(weights)))
    os.environ['GRAPHCAST_STAGE_WEIGHTS']=str(out/'weights.tsv')
    os.environ['GRAPHCAST_STAGE_LOG']=str(out/'gemm.jsonl')
    lib=C.CDLL(os.environ['LD_PRELOAD'])
    lib.graphcast_stage_select.argtypes=[C.c_uint64,C.c_char_p]
    lib.graphcast_round_select.argtypes=[C.c_uint64]
    lib.graphcast_runtime_set.argtypes=[I]
    for name in ['graphcast_stage_calls','graphcast_round_calls','graphcast_stage_unknown','graphcast_gemm_ex_calls']:
        getattr(lib,name).restype=C.c_uint64
    blas.rocblas_get_math_mode.argtypes=[V,C.POINTER(I)]
    h=V();check(blas.rocblas_create_handle(C.byref(h)))
    rows=[]
    for ta,tb in [(111,111),(112,111),(111,112),(112,112)]:
        b=np.eye(512,dtype='float32',order='F')
        aa=weights[ta==112];bb=np.asfortranarray(b.T if tb==112 else b)
        c=np.full((520,512),.25,dtype='float32',order='F')
        d=np.full((528,512),-19,dtype='float32',order='F')
        da,db,dc,dd=[device(x) for x in (aa,bb,c,d)]
        for mode, rne in [(0,0),(1,0),(1,1)]:
            for algo,solution in [(0,0),(1,24310)]:
                label='ta{}_tb{}_m{}_rne{}_algo{}'.format(ta,tb,mode,rne,algo)
                lib.graphcast_runtime_set(1)
                lib.graphcast_stage_select(mode,label.encode());lib.graphcast_round_select(rne)
                previous=1-mode
                check(blas.rocblas_set_math_mode(h,previous))
                alpha,beta=C.c_float(1),C.c_float(.5)
                before=lib.graphcast_gemm_ex_calls()
                check(blas.rocblas_gemm_ex(h,ta,tb,512,512,512,C.byref(alpha),da,151,512,db,151,512,C.byref(beta),dc,151,520,dd,151,528,151,algo,solution,0))
                check(hip.hipDeviceSynchronize())
                actual=np.empty_like(d);check(hip.hipMemcpy(actual.ctypes.data,dd,d.nbytes,2))
                expected=(rounded(a,bool(rne)) if mode else a)+np.float32(.125)
                # Identity multiply permits exact equality, not a loose tolerance.
                np.testing.assert_array_equal(actual[:512],expected)
                np.testing.assert_array_equal(actual[512:],d[512:])
                for p,x in [(da,aa),(db,bb),(dc,c)]:
                    unchanged=np.empty_like(x);check(hip.hipMemcpy(unchanged.ctypes.data,p,x.nbytes,2));np.testing.assert_array_equal(unchanged,x)
                restored=I(-1);check(blas.rocblas_get_math_mode(h,C.byref(restored)))
                assert restored.value==previous
                assert lib.graphcast_gemm_ex_calls()-before==1
                rows.append(dict(label=label,status='passed',exact_oracle=True,c_distinct_d=True,mode_restored=restored.value))
        for p in (da,db,dc,dd):check(hip.hipFree(p))
    assert lib.graphcast_stage_unknown()==0
    result=dict(status='passed',rows=rows,total=len(rows),gemm_ex_calls=lib.graphcast_gemm_ex_calls(),rounded_calls=lib.graphcast_round_calls(),notes='512x512 identity probe distinguishes native XF32 truncation and explicit RNE. K=4 native fallback is measured separately.')
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
