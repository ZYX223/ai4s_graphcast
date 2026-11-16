// Diagnostic-only SGEMM interposer: identify frozen weights by SHA256, select
// actual rocBLAS math mode per call, verify it, and restore the caller's mode.
// Experimental operand rounding only; original input buffers and checkpoint are unchanged.
#include "rne_kernel.h"
#include <openssl/sha.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>

extern "C" int rocblas_sgemm(void*,int,int,int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int);
extern "C" int rocblas_gemm_ex(void*,int,int,int,int,int,const void*,const void*,int,int,
    const void*,int,int,const void*,const void*,int,int,void*,int,int,int,int,int32_t,uint32_t);
namespace {
using Lookup=void*(*)(void*,const char*);
Lookup real_lookup() {
  static Lookup fn=reinterpret_cast<Lookup>(dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5"));
  if(!fn)std::abort();
  return fn;
}
void* native_sgemm=nullptr;
void* native_gemm_ex=nullptr;
}
// XLA resolves rocBLAS functions from a DSO handle directly, bypassing ordinary
// ELF interposition. Preserve that native address and route only this symbol.
extern "C" void* dlsym(void* handle,const char* name) {
  void* native=real_lookup()(handle,name);
  if(std::strcmp(name,"rocblas_sgemm")==0 && native && native!=reinterpret_cast<void*>(&rocblas_sgemm)) {
    native_sgemm=native;
    return reinterpret_cast<void*>(&rocblas_sgemm);
  }
  if(std::strcmp(name,"rocblas_gemm_ex")==0 && native && native!=reinterpret_cast<void*>(&rocblas_gemm_ex)) {
    native_gemm_ex=native;
    return reinterpret_cast<void*>(&rocblas_gemm_ex);
  }
  return native;
}

namespace {
struct Layer { size_t bytes; unsigned bit; std::string name, digest; };
std::mutex lock;
std::map<std::pair<size_t,std::string>,Layer> weights;
std::set<size_t> known_sizes;
std::map<std::pair<const void*,size_t>,Layer> cache;
bool runtime_enabled=false;
std::set<std::string> fused_layers;
uint64_t fused_activation_calls=0, ex_calls=0;
struct Ex { const void* c; int ldc; void* d; int ldd; int algo; int32_t solution; uint32_t flags; };
uint64_t mask=0, round_mask=0, rounded_calls=0, calls=0, unknown=0, counts[64]={0};
std::string label="initialization";
FILE* log_file=nullptr;
bool initialized=false;

[[noreturn]] void die(const std::string& message) {
  fprintf(stderr,"XF32 stage router fatal: %s\n",message.c_str());
  fflush(stderr); std::abort();
}
template<typename T> T symbol(const char* name) {
  void* ptr=nullptr;
  if(std::strcmp(name,"rocblas_sgemm")==0)ptr=native_sgemm;
  if(std::strcmp(name,"rocblas_gemm_ex")==0)ptr=native_gemm_ex;
  if(!ptr)ptr=real_lookup()(RTLD_NEXT,name);
  if(!ptr) {
    const char* lib=std::strncmp(name,"hip",3)==0?"libamdhip64.so":"/public/software/compiler/dtk-26.04/lib/librocblas.so";
    void* handle=dlopen(lib,RTLD_LAZY|RTLD_NOLOAD);
    if(handle)ptr=real_lookup()(handle,name);
  }
  if (!ptr) die(std::string("missing native symbol ")+name);
  return reinterpret_cast<T>(ptr);
}
void initialize() {
  if(initialized)return;
  const char* path=getenv("GRAPHCAST_STAGE_WEIGHTS");
  const char* log=getenv("GRAPHCAST_STAGE_LOG");
  if(!path || !log)die("weight map and log path are required");
  std::ifstream input(path);
  if(!input)die("cannot open weight map");
  Layer item;
  while(input>>item.bytes>>item.digest>>item.bit>>item.name) {
    if(item.bit>=64)die("invalid stage bit");
    auto key=std::make_pair(item.bytes,item.digest);
    if(weights.count(key) && weights[key].name!=item.name)die("ambiguous weight fingerprint");
    weights[key]=item; known_sizes.insert(item.bytes);
  }
  if(weights.empty())die("empty weight map");
  log_file=fopen(log,"a");
  if(!log_file)die("cannot open trace log");
  const char* path_fused=getenv("GRAPHCAST_FUSED_LAYERS");
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
  initialized=true;
}
bool identify(const float* data, int rows, int cols, int leading, void* stream, Layer& item) {
  if(rows<=0 || cols<=0 || rows!=leading)return false;
  size_t bytes=size_t(rows)*size_t(cols)*sizeof(float);
  if(!known_sizes.count(bytes))return false;
  auto key=std::make_pair(static_cast<const void*>(data),bytes);
  auto found=cache.find(key);
  if(found!=cache.end()) { item=found->second; return true; }
  auto sync=symbol<int(*)(void*)>("hipStreamSynchronize");
  auto copy=symbol<int(*)(void*,const void*,size_t,int)>("hipMemcpy");
  int sync_status=sync(stream);
  if(sync_status!=0) {
    auto describe=symbol<const char*(*)(int)>("hipGetErrorString");
    die(std::string("stream synchronization failed: ")+std::to_string(sync_status)+" "+describe(sync_status));
  }
  std::vector<unsigned char> host(bytes);
  if(copy(host.data(),data,bytes,2)!=0)die("weight device-to-host copy failed");
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(host.data(),host.size(),digest);
  char hex[65];
  for(unsigned i=0;i<SHA256_DIGEST_LENGTH;++i)sprintf(hex+2*i,"%02x",digest[i]);
  auto match=weights.find(std::make_pair(bytes,std::string(hex)));
  if(match==weights.end())return false;
  item=match->second; cache.emplace(key,item); return true;
}
}

extern "C" void graphcast_runtime_set(int enabled) {
  std::lock_guard<std::mutex> guard(lock); runtime_enabled=enabled!=0; cache.clear();
}
extern "C" void graphcast_runtime_enable() { graphcast_runtime_set(1); }
extern "C" uint64_t graphcast_gemm_ex_calls() {
  std::lock_guard<std::mutex> guard(lock); return ex_calls;
}
extern "C" uint64_t graphcast_fused_activation_calls() {
  std::lock_guard<std::mutex> guard(lock); return fused_activation_calls;
}
extern "C" void graphcast_stage_select(uint64_t selected, const char* call_label) {
  std::lock_guard<std::mutex> guard(lock);
  initialize(); mask=selected; label=call_label;
}
extern "C" void graphcast_round_select(uint64_t selected) {
  std::lock_guard<std::mutex> guard(lock); round_mask=selected;
}
extern "C" uint64_t graphcast_round_calls() {
  std::lock_guard<std::mutex> guard(lock); return rounded_calls;
}
extern "C" uint64_t graphcast_stage_calls() {
  std::lock_guard<std::mutex> guard(lock); return calls;
}
extern "C" uint64_t graphcast_stage_unknown() {
  std::lock_guard<std::mutex> guard(lock); return unknown;
}
extern "C" uint64_t graphcast_stage_count(unsigned bit) {
  std::lock_guard<std::mutex> guard(lock); return bit<64?counts[bit]:0;
}

static int route(void* handle, int trans_a, int trans_b,
    int m, int n, int k, const float* alpha, const float* a, int lda,
    const float* b, int ldb, const float* beta, float* c, int ldc, const Ex* ex) {
  std::lock_guard<std::mutex> guard(lock);
  initialize();
  using Gemm=int(*)(void*,int,int,int,int,int,const float*,const float*,int,
                    const float*,int,const float*,float*,int);
  using GemmEx=int(*)(void*,int,int,int,int,int,const void*,const void*,int,int,
    const void*,int,int,const void*,const void*,int,int,void*,int,int,int,int,int32_t,uint32_t);
  // Both dispatches use the caller's handle/stream/pointer mode. C and D may
  // differ; only the two input operands are replaced by rounded scratch data.
  auto native=[&](void* h,int ta,int tb,int mm,int nn,int kk,const float* al,
                  const float* aa,int la,const float* bb,int lb,const float* be,float* cc,int lc) {
    if(ex)return symbol<GemmEx>("rocblas_gemm_ex")(h,ta,tb,mm,nn,kk,al,
      aa,151,la,bb,151,lb,be,ex->c,151,ex->ldc,ex->d,151,ex->ldd,
      151,ex->algo,ex->solution,ex->flags);
    return symbol<Gemm>("rocblas_sgemm")(h,ta,tb,mm,nn,kk,al,aa,la,bb,lb,be,cc,lc);
  };
  if(!runtime_enabled)return native(handle,trans_a,trans_b,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
  auto get_mode=symbol<int(*)(void*,int*)>("rocblas_get_math_mode");
  auto set_mode=symbol<int(*)(void*,int)>("rocblas_set_math_mode");
  auto get_stream=symbol<int(*)(void*,void**)>("rocblas_get_stream");
  void* stream=nullptr;
  if(get_stream(handle,&stream)!=0)die("get_stream failed");
  // rocblas_operation_none = 111. Transposed storage is fingerprinted too.
  Layer layer;
  bool matched=identify(a,trans_a==111?m:k,trans_a==111?k:m,lda,stream,layer);
  const bool weight_is_a=matched;
  if(!matched)matched=identify(b,trans_b==111?k:n,trans_b==111?n:k,ldb,stream,layer);
  if(!matched) {
    ++unknown;
    fprintf(log_file,"{\"label\":\"%s\",\"unknown\":true,\"m\":%d,\"n\":%d,\"k\":%d,\"ta\":%d,\"tb\":%d,\"lda\":%d,\"ldb\":%d}\n",label.c_str(),m,n,k,trans_a,trans_b,lda,ldb);
    fflush(log_file); die("SGEMM operand does not match a frozen weight");
  }
  int previous=-1, effective=-1;
  int wanted=(mask & (uint64_t(1)<<layer.bit))?1:0;
  int get_status=get_mode(handle,&previous);
  int set_status=set_mode(handle,wanted);
  int read_status=get_mode(handle,&effective);
  if(get_status!=0 || set_status!=0 || read_status!=0 || effective!=wanted)
    die("math mode selection/readback failed: get="+std::to_string(get_status)+
        " set="+std::to_string(set_status)+" read="+std::to_string(read_status)+
        " previous="+std::to_string(previous)+" wanted="+std::to_string(wanted)+
        " effective="+std::to_string(effective));
  bool rounded=(round_mask & (uint64_t(1)<<layer.bit))!=0;
  if(rounded && wanted!=1)die("rounding selected outside XF32 mask");
  const bool pre_rounded=fused_layers.count(layer.name)!=0;
  if(pre_rounded && !rounded)die("fused input policy requires RNE+XF32");
  const float *aa=a,*bb=b;
  if(rounded) {
    if(!pre_rounded || weight_is_a)aa=graphcast_round_scratch(a,trans_a==111?m:k,trans_a==111?k:m,lda,stream,0);
    if(!pre_rounded || !weight_is_a)bb=graphcast_round_scratch(b,trans_b==111?k:n,trans_b==111?n:k,ldb,stream,1);
    if(!aa || !bb)die("GPU rounding failed");
    ++rounded_calls;
    if(pre_rounded)++fused_activation_calls;
  }
  int status=native(handle,trans_a,trans_b,m,n,k,alpha,aa,lda,bb,ldb,beta,c,ldc);
  if(set_mode(handle,previous)!=0)die("math mode restoration failed");
  ++calls; ++counts[layer.bit];
  if(ex)++ex_calls;
  fprintf(log_file,"{\"label\":\"%s\",\"call\":%llu,\"layer\":\"%s\",\"bit\":%u,\"weight_sha256\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"mode\":%d,\"rne_inputs\":%s,\"activation_rne_skipped\":%s,\"api\":\"%s\",\"algo\":%d,\"solution_index\":%d,\"flags\":%u,\"c_equals_d\":%s,\"previous_mode\":%d,\"status\":%d}\n",
          label.c_str(),(unsigned long long)calls,layer.name.c_str(),layer.bit,layer.digest.c_str(),m,n,k,effective,rounded?"true":"false",pre_rounded?"true":"false",ex?"gemm_ex":"sgemm",
          ex?ex->algo:0,ex?ex->solution:0,ex?ex->flags:0,
          (!ex || ex->c==ex->d)?"true":"false",previous,status);
  fflush(log_file);
  if(status!=0)die("native SGEMM failed");
  return status;
}

extern "C" int rocblas_sgemm(void* h,int ta,int tb,int m,int n,int k,const float* alpha,
 const float* a,int lda,const float* b,int ldb,const float* beta,float* c,int ldc) {
  return route(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,nullptr);
}
extern "C" int rocblas_gemm_ex(void* h,int ta,int tb,int m,int n,int k,const void* alpha,
 const void* a,int at,int lda,const void* b,int bt,int ldb,const void* beta,
 const void* c,int ct,int ldc,void* d,int dt,int ldd,int compute,int algo,int32_t solution,uint32_t flags) {
  using GemmEx=int(*)(void*,int,int,int,int,int,const void*,const void*,int,int,
    const void*,int,int,const void*,const void*,int,int,void*,int,int,int,int,int32_t,uint32_t);
  if(at!=151 || bt!=151 || ct!=151 || dt!=151 || compute!=151) {
    std::lock_guard<std::mutex> guard(lock);
    if(runtime_enabled)die("runtime GEMM_EX must use FP32 storage and compute");
    return symbol<GemmEx>("rocblas_gemm_ex")(h,ta,tb,m,n,k,alpha,a,at,lda,b,bt,ldb,beta,c,ct,ldc,d,dt,ldd,compute,algo,solution,flags);
  }
  Ex ex{c,ldc,d,ldd,algo,solution,flags};
  return route(h,ta,tb,m,n,k,static_cast<const float*>(alpha),static_cast<const float*>(a),lda,
    static_cast<const float*>(b),ldb,static_cast<const float*>(beta),static_cast<float*>(d),ldd,&ex);
}
