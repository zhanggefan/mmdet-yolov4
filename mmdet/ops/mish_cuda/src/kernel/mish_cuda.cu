#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/macros/Macros.h>
#include <cuda_runtime.h>
#include <torch/types.h>

// TORCH_CHECK replaces AT_CHECK in PyTorch 1,2, support 1.1 as well.
#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

using at::TensorIterator;
using at::native::gpu_kernel;

namespace mish_cuda_kernel {
#include "../mish.h"

void mish_kernel(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(),
      "mish_cuda_kernel", [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t inp) -> scalar_t {
          return mish_fwd_func<scalar_t>(inp);
        });
      });
}

void mish_backward_kernel(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(),
      "mish_backward_cuda_kernel", [&]() {
        gpu_kernel(iter,
                   [] GPU_LAMBDA(scalar_t grad_out, scalar_t inp) -> scalar_t {
                     return mish_bwd_func<scalar_t>(grad_out, inp);
                   });
      });
}

} // namespace mish_cuda_kernel
