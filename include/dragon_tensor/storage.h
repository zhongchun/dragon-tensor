#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dragon_tensor {

// Data type enumeration (moved from tensor.h to avoid circular dependencies)
enum class DType {
  FLOAT32,
  FLOAT64,
  INT32,
  INT64,
  UINT8,
  DECIMAL64,  // Planned
  DECIMAL128  // Planned
};

// Storage mode
enum class StorageMode { InMemory, MMap, SharedMemory };

// Memory layout
enum class Layout { RowMajor, ColumnMajor };

// Tensor metadata (optional descriptive information)
struct TensorMeta {
  std::vector<std::string> dim_names;
  std::unordered_map<std::string, std::string> labels;
  std::string description;
};

// File format header structure
struct TensorHeader {
  static constexpr uint32_t MAGIC = 0x44544E53;  // 'DTNS' in ASCII
  static constexpr uint32_t VERSION = 1;

  uint32_t magic = MAGIC;
  uint32_t version = VERSION;
  uint32_t ndim = 0;
  uint32_t dtype = 0;
  uint32_t layout = 0;  // 0=RowMajor, 1=ColumnMajor
  uint32_t endian = 0;  // 0=little, 1=big (host endian at write time)
  uint64_t data_offset = 0;
  uint64_t checksum = 0;  // CRC64 checksum (optional, 0 if disabled)
  // Shape array follows header (variable length: ndim * uint64_t)
  // Total header size: sizeof(TensorHeader) + ndim * sizeof(uint64_t)
};

// Helper functions
inline bool is_little_endian() {
  uint32_t test = 0x01020304;
  return *reinterpret_cast<uint8_t*>(&test) == 0x04;
}

// Calculate header size including shape array
inline size_t calculate_header_size(uint32_t ndim) {
  return sizeof(TensorHeader) + ndim * sizeof(uint64_t);
}

// Calculate simple checksum (CRC32 placeholder, can be upgraded to CRC64)
uint32_t calculate_checksum(const void* data, size_t size);

}  // namespace dragon_tensor
