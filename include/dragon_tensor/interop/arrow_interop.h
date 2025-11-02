#pragma once

namespace dragon_tensor {

// Forward declarations
template <typename T>
class Tensor;

// Arrow interop functions
// Provides zero-copy conversion between Dragon Tensor and Arrow Arrays
// Note: These functions require Apache Arrow library to be available

// Forward declaration for Arrow types (when Arrow is available)
// class arrow::Array;
// class arrow::RecordBatch;

/**
 * Create a Tensor from an Arrow Array (zero-copy when possible)
 * @param array Arrow Array object
 * @return Tensor with data from Arrow Array
 */
// template <typename T>
// Tensor<T> from_arrow_array(const arrow::Array& array);

/**
 * Convert a Tensor to an Arrow Array (zero-copy when possible)
 * @param tensor Tensor to convert
 * @return Shared pointer to Arrow Array
 */
// template <typename T>
// std::shared_ptr<arrow::Array> to_arrow_array(const Tensor<T>& tensor);

}  // namespace dragon_tensor
