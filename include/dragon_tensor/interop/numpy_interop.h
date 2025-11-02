#pragma once

#include <pybind11/numpy.h>

namespace dragon_tensor {

// Forward declarations
template <typename T>
class Tensor;

// NumPy interop functions
// Provides zero-copy conversion using array interface protocol

/**
 * Create a Tensor from a NumPy array (zero-copy when possible)
 * @param arr NumPy array
 * @return Tensor with data from NumPy array
 */
template <typename T>
Tensor<T> from_numpy_array(const pybind11::array_t<T>& arr);

}  // namespace dragon_tensor
