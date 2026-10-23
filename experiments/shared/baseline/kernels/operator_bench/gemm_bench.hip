// Standalone native GEMM and unchanged GraphCast RNE kernel benchmark.
// Data are deterministic synthetic FP32 values; dimensions come from model logs.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "../xf32_rne/rne_kernel.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define HIP(call) do { hipError_t s=(call); if(s!=hipSuccess) { \
  fprintf(stderr,"HIP %s: %s\n",#call,hipGetErrorString(s)); exit(2); } } while(0)
#define BLAS(call) do { rocblas_status s=(call); if(s!=rocblas_status_success) { \
  fprintf(stderr,"rocBLAS %s: %d\n",#call,int(s)); exit(3); } } while(0)

__host__ __device__ float input_value(size_t i, uint32_t seed) {
  uint32_t x=uint32_t(i)^uint32_t(i>>32)^seed;
  x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16;
  return float(x>>8)*0x1p-24f-0.5f;
}
__global__ void fill(float* p,size_t count,uint32_t seed) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
      i+=size_t(blockDim.x)*gridDim.x) p[i]=input_value(i,seed);
}
__global__ void compare_bits(const float* a,const float* b,size_t count,int* bad) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
      i+=size_t(blockDim.x)*gridDim.x)
    if(__float_as_uint(a[i])!=__float_as_uint(b[i])) atomicExch(bad,1);
}
float quantize(float x,bool nearest) {
  // Independent quotient/remainder oracle, not the GPU add-and-mask expression.
  uint32_t bits; memcpy(&bits,&x,4);
  if((bits&0x7f800000u)!=0x7f800000u) {
    uint32_t quotient=bits>>13, remainder=bits&8191u;
    if(nearest && (remainder>4096u || (remainder==4096u && (quotient&1u)))) ++quotient;
    bits=quotient<<13;
  }
  memcpy(&x,&bits,4); return x;
}
float fetch(const float* p,size_t i) {
  float x; HIP(hipMemcpy(&x,p+i,4,hipMemcpyDeviceToHost)); return x;
}
double measure(const std::function<void()>& operation,int iterations,hipStream_t stream) {
  hipEvent_t start,end; HIP(hipEventCreate(&start)); HIP(hipEventCreate(&end));
  HIP(hipEventRecord(start,stream));
  for(int i=0;i<iterations;++i) operation();
  HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
  float ms; HIP(hipEventElapsedTime(&ms,start,end));
  HIP(hipEventDestroy(start)); HIP(hipEventDestroy(end)); return double(ms)/iterations;
}
int main(int argc,char** argv) {
  if(argc!=6) { fprintf(stderr,"usage: gemm_bench M N K ROUNDS ITERATIONS (RNE+XF32 only)\n"); return 1; }
  int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]),rounds=atoi(argv[4]),iterations=atoi(argv[5]);
  if(m<=0 || n<=0 || k<=0 || rounds<=0 || iterations<=0) return 1;
  int devices; HIP(hipGetDeviceCount(&devices)); if(devices!=1) return 4;
  char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
  const char* expected=getenv("GRAPHCAST_EXPECTED_PCI");
  if(!expected || strcasecmp(pci,expected)!=0) { fprintf(stderr,"wrong PCI: %s\n",pci);return 4; }
  hipDeviceProp_t properties; HIP(hipGetDeviceProperties(&properties,0));
  const char* expected_name=getenv("GRAPHCAST_EXPECTED_DEVICE");
  if(!expected_name || !strstr(properties.name,expected_name)) {
    fprintf(stderr,"wrong device name: %s\n",properties.name);return 4;
  }
  hipStream_t stream; HIP(hipStreamCreate(&stream));
  rocblas_handle handle; BLAS(rocblas_create_handle(&handle)); BLAS(rocblas_set_stream(handle,stream));
  size_t na=size_t(m)*k,nb=size_t(k)*n,nc=size_t(m)*n;
  float *a,*b,*ar,*br,*c,*reference;
  HIP(hipMalloc(&a,4*na)); HIP(hipMalloc(&ar,4*na));
  HIP(hipMalloc(&b,4*nb)); HIP(hipMalloc(&br,4*nb));
  HIP(hipMalloc(&c,4*nc)); HIP(hipMalloc(&reference,4*nc));
  fill<<<4096,256,0,stream>>>(a,na,20261022u);
  fill<<<4096,256,0,stream>>>(b,nb,20261023u);
  HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
  float alpha=1,beta=0;
  auto round_a=[&]() { if(graphcast_round_into(a,ar,m,k,m,stream)) exit(5); };
  auto round_b=[&]() { if(graphcast_round_into(b,br,k,n,k,stream)) exit(5); };
  auto gemm=[&](const float* aa,const float* bb) {
    BLAS(rocblas_sgemm(handle,rocblas_operation_none,rocblas_operation_none,
                      m,n,k,&alpha,aa,m,bb,k,&beta,c,m));
  };
  round_a(); round_b(); HIP(hipStreamSynchronize(stream));
  // Check conversion and original-buffer integrity at 257 distributed positions.
  for(int i=0;i<257;++i) {
    size_t ia=(na-1)*size_t(i)/256,ib=(nb-1)*size_t(i)/256;
    if(fetch(a,ia)!=input_value(ia,20261022u) || fetch(b,ib)!=input_value(ib,20261023u) ||
       fetch(ar,ia)!=quantize(input_value(ia,20261022u),true) ||
       fetch(br,ib)!=quantize(input_value(ib,20261023u),true)) return 6;
  }
  const char* names[]={"fp32","xf32","rne_xf32","rne_gemm_only","round_weight","round_activation"};
  std::vector<std::vector<double>> times(6);
  std::vector<double> max_error(4,0),max_error_vs_fp64(4,0);
  int* bad; HIP(hipMalloc(&bad,sizeof(int)));
  for(int mode=2;mode<4;++mode) {
    BLAS(rocblas_set_math_mode(handle,static_cast<rocblas_math_mode>(mode==0?0:1)));
    rocblas_math_mode effective;
    BLAS(rocblas_get_math_mode(handle,&effective)); if(int(effective)!=(mode==0?0:1))return 7;
    if(mode==2) {round_a();round_b();}
    gemm(mode>=2?ar:a,mode>=2?br:b); HIP(hipStreamSynchronize(stream));
    if(mode==2) HIP(hipMemcpy(reference,c,nc*4,hipMemcpyDeviceToDevice));
    if(mode==3) {
      HIP(hipMemset(bad,0,sizeof(int)));
      compare_bits<<<4096,256,0,stream>>>(reference,c,nc,bad);
      HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
      int mismatch; HIP(hipMemcpy(&mismatch,bad,sizeof(int),hipMemcpyDeviceToHost));
      if(mismatch) {fprintf(stderr,"RNE GEMM differs from explicit pre-rounding\n");return 8;}
    }
    // Independent FP64 scalar dot products at 32 deterministic output positions.
    for(int sample=0;sample<32;++sample) {
      int row=(sample*17)%m; size_t col=size_t(sample)*(n-1)/31;
      double truth=0,original=0,sum_abs=0;
      for(int j=0;j<k;++j) {
        float av=input_value(row+size_t(j)*m,20261022u);
        float bv=input_value(j+col*k,20261023u);
        original+=double(av)*bv;
        if(mode) {av=quantize(av,mode>=2);bv=quantize(bv,mode>=2);}
        truth+=double(av)*bv; sum_abs+=fabs(double(av)*bv);
      }
      double actual=fetch(c,row+col*m),error=fabs(actual-truth);
      max_error[mode]=std::max(max_error[mode],error);
      max_error_vs_fp64[mode]=std::max(max_error_vs_fp64[mode],fabs(actual-original));
      if(!std::isfinite(actual) || error>2e-5*sum_abs+1e-6) {
        fprintf(stderr,"oracle failed mode=%s sample=%d actual=%.9g truth=%.9g\n",names[mode],sample,actual,truth);
        return 9;
      }
    }
  }
  for(int round=0;round<rounds;++round) {
    for(int position=0;position<4;++position) {
      int mode=2+(position+round)%4;
      BLAS(rocblas_set_math_mode(handle,static_cast<rocblas_math_mode>(mode==0?0:1)));
      auto operation=[&]() {
        if(mode==4) round_a();
        else if(mode==5) round_b();
        else {if(mode==2){round_a();round_b();} gemm(mode>=2?ar:a,mode>=2?br:b);}
      };
      for(int w=0;w<5;++w)operation();
      HIP(hipStreamSynchronize(stream));
      times[mode].push_back(measure(operation,iterations,stream));
    }
  }
  // Verify the complete original input buffers by regenerating into scratch.
  fill<<<4096,256,0,stream>>>(ar,na,20261022u);
  fill<<<4096,256,0,stream>>>(br,nb,20261023u);
  HIP(hipMemsetAsync(bad,0,sizeof(int),stream));
  compare_bits<<<4096,256,0,stream>>>(a,ar,na,bad);
  compare_bits<<<4096,256,0,stream>>>(b,br,nb,bad);
  HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
  int changed; HIP(hipMemcpy(&changed,bad,sizeof(int),hipMemcpyDeviceToHost)); if(changed)return 10;
  printf("{\"status\":\"passed\",\"pci\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,",pci,m,n,k);
  printf("\"device_name\":\"%s\",",properties.name);
  printf("\"iterations\":%d,\"warmup\":5,\"input_seed\":[20261022,20261023],",iterations);
  printf("\"inputs_unchanged_full\":true,\"rne_explicit_match_full\":true,\"oracle_output_samples\":32,\"timings\":{");
  for(int mode=2;mode<6;++mode) {
    if(mode>2)printf(",");printf("\"%s\":{\"round_ms\":[",names[mode]);
    for(size_t i=0;i<times[mode].size();++i){if(i)printf(",");printf("%.9g",times[mode][i]);}
    printf("]");
    if(mode<4)printf(",\"max_abs_error_vs_mode_fp64\":%.9g,\"max_abs_error_vs_original_fp64\":%.9g",max_error[mode],max_error_vs_fp64[mode]);
    printf("}");
  }
  printf("}}\n");
  HIP(hipFree(bad)); HIP(hipFree(a)); HIP(hipFree(ar)); HIP(hipFree(b)); HIP(hipFree(br));
  HIP(hipFree(c)); HIP(hipFree(reference)); BLAS(rocblas_destroy_handle(handle)); HIP(hipStreamDestroy(stream));
  return 0;
}
