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
namespace {
using Lookup=void*(*)(void*,const char*);
Lookup real_lookup() {
  static Lookup fn=reinterpret_cast<Lookup>(dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5"));
  if(!fn)std::abort();
  return fn;
}
void* native_sgemm=nullptr;
}
// XLA resolves rocBLAS functions from a DSO handle directly, bypassing ordinary
// ELF interposition. Preserve that native address and route only this symbol.
extern "C" void* dlsym(void* handle,const char* name) {
  void* native=real_lookup()(handle,name);
  if(std::strcmp(name,"rocblas_sgemm")==0 && native && native!=reinterpret_cast<void*>(&rocblas_sgemm)) {
    native_sgemm=native;
    return reinterpret_cast<void*>(&rocblas_sgemm);
  }
  return native;
}

namespace {
struct Layer { size_t bytes; unsigned bit; std::string name, digest; };
std::mutex lock;
std::map<std::pair<size_t,std::string>,Layer> weights;
std::set<size_t> known_sizes;
std::map<std::pair<const void*,size_t>,Layer> cache;
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
  if(!ptr)ptr=real_lookup()(RTLD_NEXT,name);
  if(!ptr) {
    const char* lib=std::strncmp(name,"hip",3)==0?"libamdhip64.so":"/opt/dtk/lib/librocblas.so";
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

extern "C" int rocblas_sgemm(void* handle, int trans_a, int trans_b,
    int m, int n, int k, const float* alpha, const float* a, int lda,
    const float* b, int ldb, const float* beta, float* c, int ldc) {
  std::lock_guard<std::mutex> guard(lock);
  initialize();
  using Gemm=int(*)(void*,int,int,int,int,int,const float*,const float*,int,
                    const float*,int,const float*,float*,int);
  auto native=symbol<Gemm>("rocblas_sgemm");
  auto get_mode=symbol<int(*)(void*,int*)>("rocblas_get_math_mode");
  auto set_mode=symbol<int(*)(void*,int)>("rocblas_set_math_mode");
  auto get_stream=symbol<int(*)(void*,void**)>("rocblas_get_stream");
  void* stream=nullptr;
  if(get_stream(handle,&stream)!=0)die("get_stream failed");
  // rocblas_operation_none = 111. Transposed storage is fingerprinted too.
  Layer layer;
  bool matched=identify(a,trans_a==111?m:k,trans_a==111?k:m,lda,stream,layer);
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
  const float *aa=a,*bb=b;
  if(rounded) {
    aa=graphcast_round_scratch(a,trans_a==111?m:k,trans_a==111?k:m,lda,stream,0);
    bb=graphcast_round_scratch(b,trans_b==111?k:n,trans_b==111?n:k,ldb,stream,1);
    if(!aa || !bb)die("GPU rounding failed");
    ++rounded_calls;
  }
  int status=native(handle,trans_a,trans_b,m,n,k,alpha,aa,lda,bb,ldb,beta,c,ldc);
  if(set_mode(handle,previous)!=0)die("math mode restoration failed");
  ++calls; ++counts[layer.bit];
  fprintf(log_file,"{\"label\":\"%s\",\"call\":%llu,\"layer\":\"%s\",\"bit\":%u,\"weight_sha256\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"mode\":%d,\"rne_inputs\":%s,\"previous_mode\":%d,\"status\":%d}\n",
          label.c_str(),(unsigned long long)calls,layer.name.c_str(),layer.bit,layer.digest.c_str(),m,n,k,effective,rounded?"true":"false",previous,status);
  fflush(log_file);
  if(status!=0)die("native SGEMM failed");
  return status;
}
