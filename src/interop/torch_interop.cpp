#include "dragon_tensor/interop/torch_interop.h"

#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include <vector>

#include "dragon_tensor/buffer.h"
#include "dragon_tensor/tensor.h"

namespace dragon_tensor {

// PyTorch interop implementation
// Provides zero-copy conversion via DLPack protocol

template <typename T>
Tensor<T> from_torch_tensor(pybind11::object torch_tensor) {
  // PyTorch's .numpy() already returns a zero-copy view when possible
  // Convert to numpy first (this is zero-copy if tensor is on CPU and
  // contiguous)
  pybind11::array arr =
      torch_tensor.attr("detach")().attr("cpu")().attr("numpy")();

  // Extract buffer info from numpy array
  pybind11::buffer_info buf_info = arr.request();

  // Extract shape
  std::vector<size_t> shape(buf_info.shape.begin(), buf_info.shape.end());

  // Create tensor from torch memory (via numpy view)
  // Note: Tensor currently uses std::vector<T> internally, so we need to copy
  // data
  // TODO: Add Tensor constructor that accepts Buffer for true zero-copy
  std::vector<T> data(static_cast<const T*>(buf_info.ptr),
                      static_cast<const T*>(buf_info.ptr) + buf_info.size);

  return Tensor<T>(shape, data);
}

// Explicit template instantiations
template Tensor<float> from_torch_tensor(pybind11::object);
template Tensor<double> from_torch_tensor(pybind11::object);
template Tensor<int32_t> from_torch_tensor(pybind11::object);
template Tensor<int64_t> from_torch_tensor(pybind11::object);

}  // namespace dragon_tensor
