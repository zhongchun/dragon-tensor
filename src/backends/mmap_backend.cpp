#include "dragon_tensor/backends/mmap_backend.h"

#include "dragon_tensor/buffer.h"

#include <memory>
#include <stdexcept>

namespace dragon_tensor {

MMapBackend::MMapBackend(std::string_view path, bool read_only)
    : path_(path), read_only_(read_only) {
  // Implementation would open the file here if needed
}

MMapBackend::~MMapBackend() = default;

MMapBackend::MMapBackend(MMapBackend&& other) noexcept
    : path_(std::move(other.path_)), read_only_(other.read_only_) {
  other.path_.clear();
}

MMapBackend& MMapBackend::operator=(MMapBackend&& other) noexcept {
  if (this != &other) {
    path_ = std::move(other.path_);
    read_only_ = other.read_only_;
    other.path_.clear();
  }
  return *this;
}

std::shared_ptr<Buffer> MMapBackend::allocate(size_t size_bytes, Layout) {
  // For mmap backend, we'd typically map an existing file
  // For now, return a placeholder - full implementation would map the file
  // This is a simplified version
  throw std::runtime_error("MMapBackend::allocate not yet fully implemented");
}

void MMapBackend::release(Buffer& buffer) {
  buffer.detach();
}

void MMapBackend::flush() {
  // Flush any mapped regions
}

}  // namespace dragon_tensor

