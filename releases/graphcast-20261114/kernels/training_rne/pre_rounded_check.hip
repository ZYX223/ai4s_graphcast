// Debug-only proof at the rocBLAS boundary. Never include this overhead in a
// performance claim; a validated identical graph can then use the skip policy.
#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void check_rne(const float* src,int rows,int cols,int ld,unsigned* bad) {
  size_t total=size_t(rows)*cols;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;
      i+=size_t(gridDim.x)*blockDim.x) {
    uint32_t b=__float_as_uint(src[i%rows+(i/rows)*ld]);
    if((b&0x7f800000u)!=0x7f800000u && (b&0x1fffu)) atomicOr(bad,1u);
  }
}
extern "C" int graphcast_verify_prerounded(const float* src,int rows,int cols,int ld,void* stream) {
  static unsigned* bad=nullptr;
  if(rows<=0 || cols<=0 || ld<rows)return -1;
  if(!bad && hipMalloc((void**)&bad,sizeof(unsigned))!=hipSuccess)return -2;
  auto s=reinterpret_cast<hipStream_t>(stream);
  if(hipMemsetAsync(bad,0,sizeof(unsigned),s)!=hipSuccess)return -3;
  size_t blocks=(size_t(rows)*cols+255)/256;if(blocks>65535)blocks=65535;
  hipLaunchKernelGGL(check_rne,dim3(blocks),dim3(256),0,s,src,rows,cols,ld,bad);
  if(hipGetLastError()!=hipSuccess)return -4;
  unsigned host_bad=0;
  if(hipMemcpyAsync(&host_bad,bad,sizeof(unsigned),hipMemcpyDeviceToHost,s)!=hipSuccess)return -5;
  if(hipStreamSynchronize(s)!=hipSuccess)return -6;
  return host_bad?1:0;
}
