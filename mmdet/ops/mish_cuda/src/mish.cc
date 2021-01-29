#include <ATen/native/TensorIterator.h>
#include <torch/extension.h>

using namespace pybind11::literals;

namespace mish_cuda_kernel {
void mish_kernel(torch::TensorIterator &iter);
void mish_backward_kernel(torch::TensorIterator &iter);
} // namespace mish_cuda_kernel
namespace mish_cpu_kernel {
void mish_kernel(torch::TensorIterator &iter);
void mish_backward_kernel(torch::TensorIterator &iter);
} // namespace mish_cpu_kernel

torch::Tensor mish_forward(const torch::Tensor &input) {
  auto output = torch::empty_like(input);
  auto iter = torch::TensorIterator::unary_op(output, input);
  switch (iter.device_type()) {
  case torch::kCUDA:
    mish_cuda_kernel::mish_kernel(iter);
    break;
  case torch::kCPU:
    mish_cpu_kernel::mish_kernel(iter);
    break;
  default:
    TORCH_CHECK(false,
                "Unsupported device type, should be CPU or CUDA but got ",
                input.device().type());
  }
  return output;
}

torch::Tensor mish_backward(const torch::Tensor &grad_out,
                            const torch::Tensor &input) {
  torch::Tensor grad_inp = torch::empty_like(input);;
  auto iter = torch::TensorIterator::binary_op(grad_inp, grad_out, input);
  switch (iter.device_type()) {
  case torch::kCUDA:
    mish_cuda_kernel::mish_backward_kernel(iter);
    break;
  case torch::kCPU:
    mish_cpu_kernel::mish_backward_kernel(iter);
    break;
  default:
    TORCH_CHECK(false,
                "Unsupported device type, should be CPU or CUDA but got ",
                input.device().type());
  }
  return grad_inp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mish_forward", &mish_forward, "Mish activation forward", "input"_a);
  m.def("mish_backward", &mish_backward, "Mish activation backward",
        "grad_out"_a, "input"_a);
}