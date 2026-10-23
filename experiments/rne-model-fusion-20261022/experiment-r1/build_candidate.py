"""Build an isolated activation-skip router; original rocBLAS remains untouched."""
from pathlib import Path
import subprocess
from graphcast_bench.assets import ROOT, sha256, write_json
src = ROOT/'kernels/xf32_rne'
out = Path('/output/native')
out.mkdir(exist_ok=False)
s = (src/'xf32_rne_router.cc').read_text()
def replace(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
replace('std::string label="initialization";', 'std::string label="initialization";\nstd::set<std::string> fused_layers;\nuint64_t fused_activation_calls=0;')
replace('  initialized=true;', '''  const char* fused_path=getenv("GRAPHCAST_FUSED_LAYERS");
  if(!fused_path)die("fused layer policy required");
  std::ifstream fused_input(fused_path);
  std::string fused_name;
  while(std::getline(fused_input,fused_name)) {
    if(fused_name.empty() || !fused_layers.insert(fused_name).second)die("invalid fused layer policy");
    bool found=false;
    for(const auto& weight: weights)if(weight.second.name==fused_name)found=true;
    if(!found)die("fused policy contains unknown weight");
  }
  if(fused_layers.size()!=79)die("expected 79 fused Linear inputs");
  initialized=true;''')
replace('extern "C" uint64_t graphcast_round_calls() {', '''extern "C" uint64_t graphcast_fused_activation_calls() {
  std::lock_guard<std::mutex> guard(lock); return fused_activation_calls;
}
extern "C" uint64_t graphcast_round_calls() {''')
replace('  if(!matched)matched=identify(b,', '  const bool weight_is_a=matched;\n  if(!matched)matched=identify(b,')
replace('  const float *aa=a,*bb=b;', '''  const bool pre_rounded=fused_layers.count(layer.name)!=0;
  if(pre_rounded && !rounded)die("JAX RNE policy requires RNE+XF32 mode");
  const float *aa=a,*bb=b;''')
replace('    aa=graphcast_round_scratch', '    if(!pre_rounded || weight_is_a)aa=graphcast_round_scratch')
replace('    bb=graphcast_round_scratch', '    if(!pre_rounded || !weight_is_a)bb=graphcast_round_scratch')
replace('    ++rounded_calls;', '    ++rounded_calls;\n    if(pre_rounded)++fused_activation_calls;')
replace('\\"rne_inputs\\":%s,', '\\"rne_inputs\\":%s,\\"activation_rne_skipped\\":%s,')
replace('rounded?"true":"false",previous,status);', 'rounded?"true":"false",pre_rounded?"true":"false",previous,status);')
patched=out/'xf32_rne_router.cc';patched.write_text(s)
library=out/'libxf32_rne_router.so'
cmd=['/opt/dtk/bin/hipcc','--offload-arch=gfx938','-O2','-fPIC','-shared','-I'+str(src),str(patched),str(src/'rne_kernel.hip'),'-o',str(library),'-lcrypto','-ldl','-pthread']
subprocess.run(cmd,check=True)
write_json(out/'build.json',dict(command=cmd,library_sha256=sha256(library),sources={f:sha256(src/f) for f in ['rne_kernel.h','rne_kernel.hip','xf32_rne_router.cc','weights.tsv','weights.json']},patched_source_sha256=sha256(patched),policy_sha256=sha256(Path('/experiment/policy.json')),fused_layers_sha256=sha256(Path('/experiment/fused_layers.txt')),variant='producer_fused',compiler=subprocess.check_output(['/opt/dtk/bin/hipcc','--version'],text=True)))
