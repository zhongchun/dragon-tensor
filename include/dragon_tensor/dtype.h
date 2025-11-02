#pragma once

#include <cstdint>
#include <string>

namespace dragon_tensor {

// Data type enumeration
enum class DType {
  FLOAT32,
  FLOAT64,
  INT32,
  INT64,
  UINT8,
  DECIMAL64,  // Planned: fixed-point decimal (64-bit)
  DECIMAL128  // Planned: fixed-point decimal (128-bit)
};

// Convert DType to string
inline std::string dtype_to_string(DType dtype) {
  switch (dtype) {
    case DType::FLOAT32:
      return "float32";
    case DType::FLOAT64:
      return "float64";
    case DType::INT32:
      return "int32";
    case DType::INT64:
      return "int64";
    case DType::UINT8:
      return "uint8";
    case DType::DECIMAL64:
      return "decimal64";
    case DType::DECIMAL128:
      return "decimal128";
    default:
      return "unknown";
  }
}

// Get size of DType in bytes
inline size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::FLOAT32:
      return 4;
    case DType::FLOAT64:
      return 8;
    case DType::INT32:
      return 4;
    case DType::INT64:
      return 8;
    case DType::UINT8:
      return 1;
    case DType::DECIMAL64:
      return 8;
    case DType::DECIMAL128:
      return 16;
    default:
      return 0;
  }
}

}  // namespace dragon_tensor
