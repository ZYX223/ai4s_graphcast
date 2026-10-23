// Training-only interposer. No weight identity/pointer caching: every GEMM
// operand (including dY, activations and updated weights) is rounded afresh.
#include "rne_kernel.h"
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <mutex>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdint>

extern "C" int rocblas_sgemm(void*,int,int,int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int);
namespace {
using Lookup=void*(*)(void*,const char*);
Lookup lookup(){static auto f=(Lookup)dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");if(!f)abort();return f;}
void* native_gemm=nullptr;
std::mutex gate;
int mode=0;uint64_t calls=0,rounded=0;std::string label="init";
FILE* log_file=nullptr;
[[noreturn]] void fail(const char* why){fprintf(stderr,"training precision fatal: %s\n",why);fflush(stderr);abort();}
template<class T>T symbol(const char* name){
 void* p=lookup()(RTLD_NEXT,name);
 if(!p){void* h=dlopen("/opt/dtk/lib/librocblas.so",RTLD_LAZY|RTLD_NOLOAD);if(h)p=lookup()(h,name);}
 if(!p)fail(name);return (T)p;
}
}
extern "C" void* dlsym(void* handle,const char* name){
 void* p=lookup()(handle,name);
 if(!strcmp(name,"rocblas_sgemm")&&p&&p!=(void*)&rocblas_sgemm){native_gemm=p;return (void*)&rocblas_sgemm;}
 return p;
}
extern "C" void training_select(int selected,const char* value){
 std::lock_guard<std::mutex> hold(gate);if(selected<0||selected>1)fail("invalid mode");
 mode=selected;label=value;
 if(!log_file){const char* path=getenv("TRAINING_GEMM_LOG");log_file=fopen(path?path:"/dev/null","a");if(!log_file)fail("log");}
}
extern "C" uint64_t training_calls(){std::lock_guard<std::mutex> hold(gate);return calls;}
extern "C" uint64_t training_rounded(){std::lock_guard<std::mutex> hold(gate);return rounded;}
extern "C" int rocblas_sgemm(void* h,int ta,int tb,int m,int n,int k,const float* alpha,
 const float* a,int lda,const float* b,int ldb,const float* beta,float* c,int ldc){
 std::lock_guard<std::mutex> hold(gate);
 using Gemm=int(*)(void*,int,int,int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int);
 auto gemm=native_gemm?(Gemm)native_gemm:symbol<Gemm>("rocblas_sgemm");
 auto get=symbol<int(*)(void*,int*)>("rocblas_get_math_mode");
 auto set=symbol<int(*)(void*,int)>("rocblas_set_math_mode");
 auto stream_get=symbol<int(*)(void*,void**)>("rocblas_get_stream");
 int old=-1,actual=-1;void* stream=nullptr;
 if(get(h,&old)||set(h,mode)||get(h,&actual)||actual!=mode||stream_get(h,&stream))fail("math mode or stream");
 const float* aa=a;const float* bb=b;
 if(mode){
  aa=graphcast_round_scratch(a,ta==111?m:k,ta==111?k:m,lda,stream,0);
  bb=graphcast_round_scratch(b,tb==111?k:n,tb==111?n:k,ldb,stream,1);
  if(!aa||!bb)fail("operand rounding");++rounded;
 }
 int status=gemm(h,ta,tb,m,n,k,alpha,aa,lda,bb,ldb,beta,c,ldc);
 if(set(h,old)||status)fail("GEMM or restoration");++calls;
 if(log_file){fprintf(log_file,"{\"label\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"ta\":%d,\"tb\":%d,\"mode\":%d,\"rounded\":%s,\"stream\":\"%p\"}\n",label.c_str(),m,n,k,ta,tb,actual,mode?"true":"false",stream);fflush(log_file);}
 return status;
}
