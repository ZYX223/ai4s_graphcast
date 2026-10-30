"""Build an isolated SGEMM/GEMM_EX RNE candidate from the frozen baseline.

The native rocBLAS GEMM_EX ABI, C/D, algorithm, solution index and flags are
preserved. No custom GEMM is substituted. Unsupported runtime types fail closed.
"""
from pathlib import Path
import os
import subprocess
import sys
from graphcast_bench.assets import ROOT, sha256, write_json

src = ROOT / 'kernels/xf32_rne'
out = Path(os.environ['NATIVE_OUTPUT'])
out.mkdir(exist_ok=False)
s = (src / 'xf32_rne_router.cc').read_text()

def replace(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)

ex_signature = '''void*,int,int,int,int,int,const void*,const void*,int,int,
    const void*,int,int,const void*,const void*,int,int,void*,int,int,int,int,int32_t,uint32_t'''
replace('namespace {\nusing Lookup', 'extern "C" int rocblas_gemm_ex(' + ex_signature + ');\nnamespace {\nusing Lookup')
replace('void* native_sgemm=nullptr;', 'void* native_sgemm=nullptr;\nvoid* native_gemm_ex=nullptr;')
replace('  return native;\n}', '''  if(std::strcmp(name,"rocblas_gemm_ex")==0 && native && native!=reinterpret_cast<void*>(&rocblas_gemm_ex)) {
    native_gemm_ex=native;
    return reinterpret_cast<void*>(&rocblas_gemm_ex);
  }
  return native;
}''')
replace('  if(!ptr)ptr=real_lookup()', '  if(std::strcmp(name,"rocblas_gemm_ex")==0)ptr=native_gemm_ex;\n  if(!ptr)ptr=real_lookup()')
replace('uint64_t mask=0,', '''bool runtime_enabled=false;
std::set<std::string> fused_layers;
uint64_t fused_activation_calls=0, ex_calls=0;
struct Ex { const void* c; int ldc; void* d; int ldd; int algo; int32_t solution; uint32_t flags; };
uint64_t mask=0,''')
replace('  initialized=true;', '''  const char* path_fused=getenv("GRAPHCAST_FUSED_LAYERS");
  if(path_fused && *path_fused) {
    std::ifstream fused_input(path_fused);
    if(!fused_input)die("cannot open fused layer policy");
    std::string name;
    while(std::getline(fused_input,name)) {
      bool found=false;
      for(const auto& w:weights)if(w.second.name==name)found=true;
      if(!found || !fused_layers.insert(name).second)die("invalid fused layer policy");
    }
    if(fused_layers.size()!=79)die("expected 79 fused Linear inputs");
  }
  initialized=true;''')
replace('extern "C" void graphcast_stage_select(', '''extern "C" void graphcast_runtime_set(int enabled) {
  std::lock_guard<std::mutex> guard(lock); runtime_enabled=enabled!=0; cache.clear();
}
extern "C" void graphcast_runtime_enable() { graphcast_runtime_set(1); }
extern "C" uint64_t graphcast_gemm_ex_calls() {
  std::lock_guard<std::mutex> guard(lock); return ex_calls;
}
extern "C" uint64_t graphcast_fused_activation_calls() {
  std::lock_guard<std::mutex> guard(lock); return fused_activation_calls;
}
extern "C" void graphcast_stage_select(''')
replace('extern "C" int rocblas_sgemm(void* handle,', 'static int route(void* handle,')
replace('const float* b, int ldb, const float* beta, float* c, int ldc) {',
        'const float* b, int ldb, const float* beta, float* c, int ldc, const Ex* ex) {')
replace('  auto native=symbol<Gemm>("rocblas_sgemm");', '''  using GemmEx=int(*)(''' + ex_signature + ''');
  // Both dispatches use the caller's handle/stream/pointer mode. C and D may
  // differ; only the two input operands are replaced by rounded scratch data.
  auto native=[&](void* h,int ta,int tb,int mm,int nn,int kk,const float* al,
                  const float* aa,int la,const float* bb,int lb,const float* be,float* cc,int lc) {
    if(ex)return symbol<GemmEx>("rocblas_gemm_ex")(h,ta,tb,mm,nn,kk,al,
      aa,151,la,bb,151,lb,be,ex->c,151,ex->ldc,ex->d,151,ex->ldd,
      151,ex->algo,ex->solution,ex->flags);
    return symbol<Gemm>("rocblas_sgemm")(h,ta,tb,mm,nn,kk,al,aa,la,bb,lb,be,cc,lc);
  };
  if(!runtime_enabled)return native(handle,trans_a,trans_b,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);''')
replace('  if(!matched)matched=identify(b,', '  const bool weight_is_a=matched;\n  if(!matched)matched=identify(b,')
replace('  const float *aa=a,*bb=b;', '''  const bool pre_rounded=fused_layers.count(layer.name)!=0;
  if(pre_rounded && !rounded)die("fused input policy requires RNE+XF32");
  const float *aa=a,*bb=b;''')
replace('    aa=graphcast_round_scratch', '    if(!pre_rounded || weight_is_a)aa=graphcast_round_scratch')
replace('    bb=graphcast_round_scratch', '    if(!pre_rounded || !weight_is_a)bb=graphcast_round_scratch')
replace('    ++rounded_calls;', '    ++rounded_calls;\n    if(pre_rounded)++fused_activation_calls;')
replace('  ++calls; ++counts[layer.bit];', '  ++calls; ++counts[layer.bit];\n  if(ex)++ex_calls;')
replace('\\"rne_inputs\\":%s,', '\\"rne_inputs\\":%s,\\"activation_rne_skipped\\":%s,\\"api\\":\\"%s\\",\\"algo\\":%d,\\"solution_index\\":%d,\\"flags\\":%u,\\"c_equals_d\\":%s,')
replace('rounded?"true":"false",previous,status);', '''rounded?"true":"false",pre_rounded?"true":"false",ex?"gemm_ex":"sgemm",
          ex?ex->algo:0,ex?ex->solution:0,ex?ex->flags:0,
          (!ex || ex->c==ex->d)?"true":"false",previous,status);''')
s += '''
extern "C" int rocblas_sgemm(void* h,int ta,int tb,int m,int n,int k,const float* alpha,
 const float* a,int lda,const float* b,int ldb,const float* beta,float* c,int ldc) {
  return route(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,nullptr);
}
extern "C" int rocblas_gemm_ex(void* h,int ta,int tb,int m,int n,int k,const void* alpha,
 const void* a,int at,int lda,const void* b,int bt,int ldb,const void* beta,
 const void* c,int ct,int ldc,void* d,int dt,int ldd,int compute,int algo,int32_t solution,uint32_t flags) {
  using GemmEx=int(*)(''' + ex_signature + ''');
  if(at!=151 || bt!=151 || ct!=151 || dt!=151 || compute!=151) {
    std::lock_guard<std::mutex> guard(lock);
    if(runtime_enabled)die("runtime GEMM_EX must use FP32 storage and compute");
    return symbol<GemmEx>("rocblas_gemm_ex")(h,ta,tb,m,n,k,alpha,a,at,lda,b,bt,ldb,beta,c,ct,ldc,d,dt,ldd,compute,algo,solution,flags);
  }
  Ex ex{c,ldc,d,ldd,algo,solution,flags};
  return route(h,ta,tb,m,n,k,static_cast<const float*>(alpha),static_cast<const float*>(a),lda,
    static_cast<const float*>(b),ldb,static_cast<const float*>(beta),static_cast<float*>(d),ldd,&ex);
}
'''
s = s.replace('/opt/dtk/lib/librocblas.so','/public/software/compiler/dtk-26.04/lib/librocblas.so')
patched = out / 'xf32_rne_router.cc'
patched.write_text(s)
library = out / 'libxf32_rne_router.so'
cmd = ['/public/software/compiler/dtk-26.04/bin/hipcc','--offload-arch=gfx938','-O2','-fPIC','-shared',
       '-I'+sys.prefix+'/include','-L'+sys.prefix+'/lib','-I'+str(src),str(patched),str(src/'rne_kernel.hip'),
       '-o',str(library),'-lcrypto','-ldl','-pthread']
subprocess.run(cmd,check=True)
policy = Path(os.environ.get('GRAPHCAST_EXPERIMENT','/tmp/zyx-flagos-rne-inference-20261029/experiment'))
write_json(out/'build.json',dict(command=cmd,library_sha256=sha256(library),
    sources={f:sha256(src/f) for f in ['rne_kernel.h','rne_kernel.hip','xf32_rne_router.cc','weights.tsv','weights.json']},
    patched_source_sha256=sha256(patched),builder_sha256=sha256(Path(__file__)),
    policy_sha256=sha256(policy/'policy.json'),fused_layers_sha256=sha256(policy/'fused_layers.txt'),
    variant='gemm_ex_candidate_optional_producer_fused',compiler=subprocess.check_output([cmd[0],'--version'],text=True)))
