#include <ATen/native/cpu/Loops.h>
#include <torch/types.h>

using at::TensorIterator;
using at::native::cpu_kernel;

namespace mish_cpu_kernel {
#include "../mish.h"

void mish_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "mish_cpu_kernel", [&]() {
    cpu_kernel(iter, [](scalar_t inp) -> scalar_t {
      return mish_fwd_func<scalar_t>(inp);
    });
  });
}

void mish_backward_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "mish_backward_cpu_kernel", [&]() {
    cpu_kernel(iter, [](scalar_t grad_out, scalar_t inp) -> scalar_t {
      return mish_bwd_func<scalar_t>(grad_out, inp);
    });
  });
}

} // namespace mish_cpu_kernel