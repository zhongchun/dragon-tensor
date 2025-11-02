#pragma once

#include "dragon_tensor/backend.h"

#include <memory>
#include <string>
#include <string_view>

namespace dragon_tensor {

// Shared memory backend (POSIX shared memory)
class SharedMemoryBackend : public Backend {
 public:
  explicit SharedMemoryBackend(std::string_view name);
  ~SharedMemoryBackend() override;

  // Disable copy
  SharedMemoryBackend(const SharedMemoryBackend&) = delete;
  SharedMemoryBackend& operator=(const SharedMemoryBackend&) = delete;

  // Enable move
  SharedMemoryBackend(SharedMemoryBackend&& other) noexcept;
  SharedMemoryBackend& operator=(SharedMemoryBackend&& other) noexcept;

  [[nodiscard]] std::shared_ptr<Buffer> allocate(size_t size_bytes,
                                                  Layout layout) override;

  void release(Buffer& buffer) override;

  [[nodiscard]] std::string name() const override {
    return "SharedMemoryBackend";
  }

  [[nodiscard]] bool is_writable() const override { return true; }
  [[nodiscard]] bool is_readable() const override { return true; }

  const std::string& shared_name() const { return name_; }

 private:
  std::string name_;
  bool owns_;
};

}  // namespace dragon_tensor

