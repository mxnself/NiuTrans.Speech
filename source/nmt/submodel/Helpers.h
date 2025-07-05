#pragma once
#include <algorithm>
#include <cstdint>
namespace nmt {
  // The following kernels are adapted from:
  // https://github.com/pytorch/pytorch/blob/40eff454ce5638fbff638a7f4502e29ffb9a2f0d/aten/src/ATen/native/cuda/SoftMax.cu
  // we reimplement the reduce kernel with shfl function, which is similar to faster transformers.
  using index_t = unsigned int;
  constexpr int ILP = 2;
  constexpr int max_threads = 1024;
  
  inline dim3 get_block_size(index_t dim_size) {
    index_t block_size = 1;
    index_t max_block_size = std::min(dim_size / ILP, static_cast<index_t>(max_threads));
    while (block_size < max_block_size)
      block_size *= 2;
    // Launch at least a single warp - the kernel assumes that.
    block_size = std::max(static_cast<index_t>(block_size), static_cast<index_t>(32));
    return dim3(block_size);
  }
 
  template <typename Reduction>
  __device__ __forceinline__ float warp_reduce(float val,
                                               const Reduction& r) {
    unsigned mask = 0xffffffff;
    int laneMask = 16;
    #pragma unroll
    for(;laneMask>0;laneMask>>=1) {
      val = r(val, __shfl_xor_sync(mask, val, laneMask, 32));
    }
    return val;
  }
 
  template <typename Reduction>
  __device__ __forceinline__ float block_reduce(float* smem,
                                                float val,
                                                const Reduction& r,
                                                const float defaultVal)
  {
    const int lane = threadIdx.x & 0x1f;
    const int warpId = threadIdx.x >> 5;
  
    val = warp_reduce(val, r);

    if (lane == 0) smem[warpId] = val;

    __syncthreads();
  
    val = (threadIdx.x < blockDim.x / 32) ? smem[lane] : defaultVal;

    val = warp_reduce(val, r);

    if(threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
  }
  
  template <typename Reduction>
  __device__ __forceinline__ float ilp_reduce(const float* data,
                                               index_t size,
                                               const Reduction& r,
                                               float defaultVal)
  {
    float threadVal = defaultVal;
    index_t offset = threadIdx.x;
    index_t last = size % (ILP * blockDim.x);
  
    // Body (unroll by ILP times)
    for (; offset < size - last; offset += blockDim.x * ILP) {
      float tmp[ILP];
  
      #pragma unroll
      for (index_t j = 0; j < ILP; ++j)
        tmp[j] = data[offset + j * blockDim.x];
  
      #pragma unroll
      for (index_t j = 0; j < ILP; ++j)
        threadVal = r(threadVal, tmp[j]);
    }
  
    // Epilogue
    for (; offset < size; offset += blockDim.x)
      threadVal = r(threadVal, data[offset]);
  
    return threadVal;
  }

  template <typename Epilogue>
  __device__ __forceinline__ void
  apply_epilogue(const float* input,
                 index_t cols,
                 const Epilogue& epilogue,
                 float* output) {
    index_t offset = threadIdx.x;
    index_t last = cols % (ILP * blockDim.x);
    for (; offset < cols - last; offset += blockDim.x * ILP) {
      float tmp[ILP];
  
      #pragma unroll
      for (index_t j = 0; j < ILP; ++j)
        tmp[j] = input[offset + j * blockDim.x];
  
      #pragma unroll
      for (index_t j = 0; j < ILP; ++j)
        output[offset + j * blockDim.x] = epilogue(tmp[j]);
    }
  
    for (; offset < cols; offset += blockDim.x)
      output[offset] = epilogue(input[offset]);
  }

}
