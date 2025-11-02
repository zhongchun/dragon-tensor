#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dragon_tensor {

// Shape utilities for multi-dimensional tensors
using Shape = std::vector<size_t>;

// Calculate total number of elements from shape
inline size_t calculate_size(const Shape& shape) {
  if (shape.empty()) return 0;
  size_t size = 1;
  for (size_t dim : shape) {
    size *= dim;
  }
  return size;
}

// Validate shape (check for zero dimensions, overflow, etc.)
inline void validate_shape(const Shape& shape) {
  for (size_t dim : shape) {
    if (dim == 0) {
      throw std::invalid_argument("Shape dimensions must be > 0");
    }
  }
}

// Calculate strides from shape (row-major)
inline std::vector<size_t> calculate_strides_row_major(const Shape& shape,
                                                       size_t element_size) {
  if (shape.empty()) return {};
  std::vector<size_t> strides(shape.size());
  strides[shape.size() - 1] = element_size;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Calculate strides from shape (column-major)
inline std::vector<size_t> calculate_strides_column_major(const Shape& shape,
                                                          size_t element_size) {
  if (shape.empty()) return {};
  std::vector<size_t> strides(shape.size());
  strides[0] = element_size;
  for (size_t i = 1; i < shape.size(); ++i) {
    strides[i] = strides[i - 1] * shape[i - 1];
  }
  return strides;
}

// Calculate offset from indices and strides
inline size_t calculate_offset(const std::vector<size_t>& indices,
                               const std::vector<size_t>& strides) {
  if (indices.size() != strides.size()) {
    throw std::invalid_argument("Indices size must match strides size");
  }
  size_t offset = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

}  // namespace dragon_tensor
