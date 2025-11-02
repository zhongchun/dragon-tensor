#include "dragon_tensor/backends/memory_backend.h"

#include "dragon_tensor/buffer.h"

#include <memory>

namespace dragon_tensor {

std::shared_ptr<Buffer> MemoryBackend::allocate(size_t size_bytes, Layout) {
  return std::make_shared<MemoryBuffer>(size_bytes);
}

void MemoryBackend::release(Buffer& buffer) {
  // For memory backend, buffers are automatically managed by shared_ptr
  // No explicit release needed
  (void)buffer;
}

}  // namespace dragon_tensor

