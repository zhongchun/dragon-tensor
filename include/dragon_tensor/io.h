#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

#include "dragon_tensor/storage.h"

namespace dragon_tensor {
// DType is defined in storage.h (already included above)

// Forward declarations
template <typename T>
class Tensor;
class Buffer;

// File I/O utilities
namespace io {

// Save tensor to file with specified layout
template <typename T>
void save_tensor(const Tensor<T>& tensor, const std::string& path,
                 Layout layout = Layout::RowMajor);

// Load tensor from file (with optional mmap)
template <typename T>
Tensor<T> load_tensor(const std::string& path, bool mmap = true);

// Helper to get dtype enum from type
template <typename T>
constexpr DType get_dtype() {
  if constexpr (std::is_same_v<T, float>) return DType::FLOAT32;
  if constexpr (std::is_same_v<T, double>) return DType::FLOAT64;
  if constexpr (std::is_same_v<T, int32_t>) return DType::INT32;
  if constexpr (std::is_same_v<T, int64_t>) return DType::INT64;
  if constexpr (std::is_same_v<T, uint8_t>) return DType::UINT8;
  return DType::FLOAT64;  // Default fallback
}

// Helper to get type size in bytes
template <typename T>
constexpr size_t get_type_size() {
  return sizeof(T);
}

}  // namespace io

}  // namespace dragon_tensor
