#pragma once

#include <pybind11/pytypes.h>

namespace dragon_tensor {

// Forward declarations
template <typename T>
class Tensor;

// PyTorch interop functions
// Provides zero-copy conversion via DLPack protocol

/**
 * Create a Tensor from a PyTorch tensor (zero-copy when possible)
 * Tensor must be on CPU and contiguous for zero-copy conversion
 * @param torch_tensor PyTorch tensor object
 * @return Tensor with data from PyTorch tensor
 */
template <typename T>
Tensor<T> from_torch_tensor(pybind11::object torch_tensor);

}  // namespace dragon_tensor
