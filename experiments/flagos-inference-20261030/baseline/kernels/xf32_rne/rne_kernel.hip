#include "rne_kernel.h"
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

// IEEE binary32 bits, preserving sign, exponent, NaN payloads and infinities.
// Add half an ULP minus one plus the retained LSB: nearest, ties to even.
__global__ void round_rne(const float* src,float* dst,int rows,int cols,int ld) {
  size_t total=size_t(rows)*cols;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;
      i+=size_t(gridDim.x)*blockDim.x) {
    size_t offset=i%rows+(i/rows)*ld;
    uint32_t bits=__float_as_uint(src[offset]);
    if((bits & 0x7f800000u)!=0x7f800000u)
      bits=(bits+0x00000fffu+((bits>>13)&1u)) & 0xffffe000u;
    dst[offset]=__uint_as_float(bits);
  }
}
extern "C" int graphcast_round_into(const float* src,float* dst,
    int rows,int cols,int ld,void* stream) {
  if(rows<=0 || cols<=0 || ld<rows)return -1;
  size_t blocks=(size_t(rows)*cols+255)/256;
  if(blocks>65535)blocks=65535;
  hipLaunchKernelGGL(round_rne,dim3(blocks),dim3(256),0,
                    reinterpret_cast<hipStream_t>(stream),src,dst,rows,cols,ld);
  return int(hipGetLastError());
}
extern "C" const float* graphcast_round_scratch(const float* src,
    int rows,int cols,int ld,void* stream,int slot) {
  // Calls are serialized by the interposer mutex; all model work uses one stream.
  static float* buffer[2]={nullptr,nullptr};
  static size_t capacity[2]={0,0};
  static void* last_stream=nullptr;
  if(slot<0 || slot>1 || rows<=0 || cols<=0 || ld<rows)return nullptr;
  if(last_stream && last_stream!=stream && hipDeviceSynchronize()!=hipSuccess)return nullptr;
  last_stream=stream;
  size_t bytes=(size_t(cols-1)*ld+rows)*sizeof(float);
  if(bytes>capacity[slot]) {
    if(buffer[slot] && hipFree(buffer[slot])!=hipSuccess)return nullptr;
    buffer[slot]=nullptr;
    if(hipMalloc(reinterpret_cast<void**>(&buffer[slot]),bytes)!=hipSuccess)return nullptr;
    capacity[slot]=bytes;
  }
  if(graphcast_round_into(src,buffer[slot],rows,cols,ld,stream)!=0)return nullptr;
  return buffer[slot];
}
