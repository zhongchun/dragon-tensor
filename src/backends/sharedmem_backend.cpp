#include "dragon_tensor/backends/sharedmem_backend.h"

#include <memory>
#include <stdexcept>

#include "dragon_tensor/buffer.h"

namespace dragon_tensor {

SharedMemoryBackend::SharedMemoryBackend(std::string_view name)
    : name_(name), owns_(false) {
  // Implementation would create or attach to shared memory here
}

SharedMemoryBackend::~SharedMemoryBackend() {
  // Cleanup if needed
}

SharedMemoryBackend::SharedMemoryBackend(SharedMemoryBackend&& other) noexcept
    : name_(std::move(other.name_)), owns_(other.owns_) {
  other.name_.clear();
  other.owns_ = false;
}

SharedMemoryBackend& SharedMemoryBackend::operator=(
    SharedMemoryBackend&& other) noexcept {
  if (this != &other) {
    name_ = std::move(other.name_);
    owns_ = other.owns_;
    other.name_.clear();
    other.owns_ = false;
  }
  return *this;
}

std::shared_ptr<Buffer> SharedMemoryBackend::allocate(size_t size_bytes,
                                                      Layout) {
  // Create or attach to shared memory buffer
  auto buffer = SharedMemoryBuffer::create(name_, size_bytes);
  owns_ = true;
  return buffer;
}

void SharedMemoryBackend::release(Buffer& buffer) { buffer.detach(); }

}  // namespace dragon_tensor
