#include "dragon_tensor/storage.h"

#include <cstddef>
#include <cstdint>

namespace dragon_tensor {

// Simple checksum calculation (CRC32, can be upgraded to CRC64)
uint32_t calculate_checksum(const void* data, size_t size) {
  // Simple polynomial checksum (simplified CRC32)
  const uint32_t polynomial = 0xEDB88320;
  uint32_t crc = 0xFFFFFFFF;

  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    crc ^= bytes[i];
    for (int j = 0; j < 8; ++j) {
      if (crc & 1) {
        crc = (crc >> 1) ^ polynomial;
      } else {
        crc >>= 1;
      }
    }
  }

  return crc ^ 0xFFFFFFFF;
}

}  // namespace dragon_tensor
