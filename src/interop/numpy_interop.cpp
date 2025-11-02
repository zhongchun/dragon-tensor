#include "dragon_tensor/interop/numpy_interop.h"

#include "dragon_tensor/buffer.h"
#include "dragon_tensor/tensor.h"

#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

namespace dragon_tensor {

// NumPy interop implementation
// Provides zero-copy conversion using array interface protocol

template <typename T>
Tensor<T> from_numpy_array(const pybind11::array_t<T>& arr) {
  // Get buffer info from NumPy array
  pybind11::buffer_info buf_info = arr.request();

  // Check if array is C-contiguous (row-major)
  bool is_contiguous = arr.flags() & pybind11::array::c_style;

  // Extract shape and strides
  std::vector<size_t> shape(buf_info.shape.begin(), buf_info.shape.end());
  std::vector<size_t> strides;
  
  if (!is_contiguous && buf_info.strides.size() > 0) {
    // Convert strides from bytes to elements
    strides.reserve(buf_info.strides.size());
    for (auto s : buf_info.strides) {
      strides.push_back(static_cast<size_t>(s) / sizeof(T));
    }
  }

  // Create tensor from NumPy array data
  // Note: Tensor currently uses std::vector<T> internally, so we need to copy data
  // TODO: Add Tensor constructor that accepts Buffer for true zero-copy
  std::vector<T> data(static_cast<const T*>(buf_info.ptr),
                      static_cast<const T*>(buf_info.ptr) + buf_info.size);

  // Create tensor with shape and data
  return Tensor<T>(shape, data);
}

// Explicit template instantiations
template Tensor<float> from_numpy_array(const pybind11::array_t<float>&);
template Tensor<double> from_numpy_array(const pybind11::array_t<double>&);
template Tensor<int32_t> from_numpy_array(const pybind11::array_t<int32_t>&);
template Tensor<int64_t> from_numpy_array(const pybind11::array_t<int64_t>&);

}  // namespace dragon_tensor

