"""Check actual rocBLAS GEMM_EX arithmetic before changing the adapter.

One-hot B makes A @ B exactly representable: FP32 vs XF32 truncation are
distinguishable without relying on a math-mode flag or a kernel name.
"""
import ctypes as C
import json
import os
from pathlib import Path
import numpy as np

V, I = C.c_void_p, C.c_int
hip = C.CDLL('libamdhip64.so')
blas = C.CDLL('/public/software/compiler/dtk-26.04/lib/librocblas.so')
hip.hipMalloc.argtypes = [C.POINTER(V), C.c_size_t]
hip.hipMemcpy.argtypes = [V, V, C.c_size_t, I]
hip.hipFree.argtypes = [V]
blas.rocblas_create_handle.argtypes = [C.POINTER(V)]
blas.rocblas_set_math_mode.argtypes = [V, I]
blas.rocblas_gemm_ex.argtypes = [V,I,I,I,I,I,V,V,I,I,V,I,I,V,V,I,I,V,I,I,I,I,I,C.c_uint32]
blas.rocblas_sgemm.argtypes = [V,I,I,I,I,I,V,V,I,V,I,V,V,I]

def check(status):
    if status != 0:
        raise RuntimeError('native status {}'.format(status))

def device(x):
    p = V()
    check(hip.hipMalloc(C.byref(p), x.nbytes))
    check(hip.hipMemcpy(p, x.ctypes.data, x.nbytes, 1))
    return p

def main():
    out = Path(os.environ['PROBE_OUTPUT'])
    out.mkdir(exist_ok=False)
    h = V(); check(blas.rocblas_create_handle(C.byref(h)))
    rows = []
    for n, k in [(327660, 4), (1618745, 4), (512, 512)]:
        m = 512
        a = np.asfortranarray(np.random.default_rng(19).uniform(-2,2,(m,k)).astype('float32'))
        b = np.zeros((k,n),dtype='float32',order='F')
        b[np.arange(n)%k,np.arange(n)] = 1
        c = np.zeros((m,n),dtype='float32',order='F')
        da, db, dc, dd = [device(x) for x in (a,b,c,c)]
        one, zero = C.c_float(1), C.c_float(0)
        expected = a[:,np.arange(n)%k]
        trunc = (expected.copy().view('uint32') & np.uint32(0xffffe000)).view('float32')
        for mode in (0,1):
            check(blas.rocblas_set_math_mode(h,mode))
            for api, algo, solution in [('ex',0,0),('ex',1,24310),('sgemm',0,0)]:
                if api == 'ex':
                    status = blas.rocblas_gemm_ex(h,111,111,m,n,k,C.byref(one),da,151,m,db,151,k,C.byref(zero),dc,151,m,dd,151,m,151,algo,solution,0)
                    result_ptr = dd
                else:
                    status = blas.rocblas_sgemm(h,111,111,m,n,k,C.byref(one),da,m,db,k,C.byref(zero),dc,m)
                    result_ptr = dc
                row = dict(m=m,n=n,k=k,mode=mode,api=api,algo=algo,solution=solution,status=status)
                if status == 0:
                    check(hip.hipDeviceSynchronize())
                    check(hip.hipMemcpy(c.ctypes.data,result_ptr,c.nbytes,2))
                    row.update(fp32_equal=bool(np.array_equal(c,expected)),xf32_trunc_equal=bool(np.array_equal(c,trunc)),max_abs_fp32=float(np.max(np.abs(c-expected))))
                rows.append(row)
                print(json.dumps(row),flush=True)
        for p in (da,db,dc,dd): check(hip.hipFree(p))
    (out/'report.json').write_text(json.dumps(rows,indent=2))

if __name__ == '__main__': main()
