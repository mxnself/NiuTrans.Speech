#include "Softmax.cuh"
#include "Helpers.h"
#include "../../niutensor/tensor/XDevice.h"
#include <stdexcept>
#include <limits>
namespace nmt {
// The following CUDA kernels are adapted from:
// https://github.com/pytorch/pytorch/blob/40eff454ce5638fbff638a7f4502e29ffb9a2f0d/aten/src/ATen/native/cuda/SoftMax.cu
// which has the following license notice:

/*
  From PyTorch:

  Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
  Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
  Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
  Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
  Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
  Copyright (c) 2011-2013 NYU                      (Clement Farabet)
  Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
  Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
  Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

  From Caffe2:

  Copyright (c) 2016-present, Facebook Inc. All rights reserved.

  All contributions by Facebook:
  Copyright (c) 2016 Facebook Inc.

  All contributions by Google:
  Copyright (c) 2015 Google Inc.
  All rights reserved.

  All contributions by Yangqing Jia:
  Copyright (c) 2015 Yangqing Jia
  All rights reserved.

  All contributions from Caffe:
  Copyright(c) 2013, 2014, 2015, the respective contributors
  All rights reserved.

  All other contributions:
  Copyright(c) 2015, 2016 the respective contributors
  All rights reserved.

  Caffe2 uses a copyright model similar to Caffe: each contributor holds
  copyright over their contributions to Caffe2. The project versioning records
  all such contribution and copyright details. If a contributor wants to further
  mark their specific copyright on a particular contribution, they should
  indicate their copyright solely in the commit message of the change when it is
  committed.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
     and IDIAP Research Institute nor the names of its contributors may be
     used to endorse or promote products derived from this software without
     specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/
  constexpr float max_float = std::numeric_limits<float>::max();
  struct SoftMaxForwardEpilogue {
    __device__ __forceinline__ SoftMaxForwardEpilogue(float max_input, float sum)
      : max_input(max_input)
      , sum(sum) {}
  
    __device__ __forceinline__ float operator()(float input) const {
      return static_cast<float>(std::exp(input - max_input) / sum);
    }
  
    const float max_input;
    const float sum;
  };
  
  struct SumExpFloat
  {
    __device__ __forceinline__ SumExpFloat(float v)
      : max_k(v) {}
  
    __device__ __forceinline__ float operator()(float sum, float v) const {
      return sum + std::exp(v - max_k);
    }
  
    const float max_k;
  };
  
  struct Max {
    __device__ __forceinline__ float operator()(float a, float b) const {
      return a < b ? b : a;
    }
  };
  
  struct Add {
    __device__ __forceinline__ float operator()(float a, float b) const {
      return a + b;
    }
  };

  __global__ void
  cunn_SoftMaxForward(float *output,
                      const float *input,
                      const index_t cols) {
    extern __shared__ unsigned char smem[];
    auto sdata = reinterpret_cast<float*>(smem);
    // forward pointers to batch[blockIdx.x]
    // each block handles a sample in the mini-batch
    const index_t row = blockIdx.x;
    input += row * cols;
    output += row * cols;
  
    // find the max
    float threadMax = ilp_reduce(
      input, cols, Max(), -max_float);
    float max_k = block_reduce(
      sdata, threadMax, Max(), -max_float);
  
    // reduce all values
    float threadExp = ilp_reduce(
      input, cols, SumExpFloat(max_k), 0);
    float sumAll = block_reduce(
      sdata, threadExp, Add(), 0);
  
    // apply epilogue
    apply_epilogue(
      input, cols, SoftMaxForwardEpilogue(max_k, sumAll), output);
  }
  
  void softmax_kernel_impl(const float* const x,
                           const index_t rows,
                           const index_t cols,
                           float* const y) {
    const dim3 grid(rows);
    const dim3 block(get_block_size(cols));
  
    cunn_SoftMaxForward
      <<<grid, block, 32 * sizeof (float), 0>>>(y, x, cols);
  }
  
  nts::XTensor softmax(const nts::XTensor& x) {
    nts::XTensor y(&x);
    y.SetTMPFlag();
  
    const index_t cols = x.dimSize[x.order-1];
    const index_t rows = x.unitNum / cols;
    int devIDBackup;
    ProtectCudaDev(x.devID, devIDBackup);
    softmax_kernel_impl(static_cast<float*>(x.data),
                        rows,
                        cols,
                        static_cast<float*>(y.data));
    BacktoCudaDev(x.devID, devIDBackup);
  
    return y;
  }
}
