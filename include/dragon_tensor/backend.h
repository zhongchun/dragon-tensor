#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

#include "dragon_tensor/buffer.h"
#include "dragon_tensor/layout.h"

namespace dragon_tensor {

// Forward declarations
class Buffer;

// Backend abstraction interface for storage backends
class Backend {
 public:
  virtual ~Backend() = default;

  // Allocate a buffer with specified size and layout
  [[nodiscard]] virtual std::shared_ptr<Buffer> allocate(size_t size_bytes,
                                                         Layout layout) = 0;

  // Release a buffer (may be a no-op for some backends)
  virtual void release(Buffer& buffer) = 0;

  // Get backend name
  [[nodiscard]] virtual std::string name() const = 0;

  // Flush any pending writes (for file-backed backends)
  virtual void flush() {}

  // Check if backend supports memory mapping
  [[nodiscard]] virtual bool supports_mmap() const { return false; }

  // Check if backend is writable
  [[nodiscard]] virtual bool is_writable() const { return true; }

  // Check if backend is readable
  [[nodiscard]] virtual bool is_readable() const { return true; }
};

// Backend factory functions
[[nodiscard]] std::shared_ptr<Backend> create_memory_backend();
[[nodiscard]] std::shared_ptr<Backend> create_mmap_backend(
    std::string_view path, bool read_only = false);
[[nodiscard]] std::shared_ptr<Backend> create_shared_memory_backend(
    std::string_view name);
[[nodiscard]] std::shared_ptr<Backend> create_arrow_backend(
    std::string_view path = "", bool read_only = false);

}  // namespace dragon_tensor
