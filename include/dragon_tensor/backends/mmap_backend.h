#pragma once

#include "dragon_tensor/backend.h"

#include <memory>
#include <string>
#include <string_view>

namespace dragon_tensor {

// Memory-mapped file backend
class MMapBackend : public Backend {
 public:
  explicit MMapBackend(std::string_view path, bool read_only = false);
  ~MMapBackend() override;

  // Disable copy
  MMapBackend(const MMapBackend&) = delete;
  MMapBackend& operator=(const MMapBackend&) = delete;

  // Enable move
  MMapBackend(MMapBackend&& other) noexcept;
  MMapBackend& operator=(MMapBackend&& other) noexcept;

  [[nodiscard]] std::shared_ptr<Buffer> allocate(size_t size_bytes,
                                                  Layout layout) override;

  void release(Buffer& buffer) override;

  [[nodiscard]] std::string name() const override { return "MMapBackend"; }

  void flush() override;

  [[nodiscard]] bool supports_mmap() const override { return true; }

  [[nodiscard]] bool is_writable() const override { return !read_only_; }
  [[nodiscard]] bool is_readable() const override { return true; }

 private:
  std::string path_;
  bool read_only_;
};

}  // namespace dragon_tensor

