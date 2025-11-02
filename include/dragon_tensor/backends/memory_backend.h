#pragma once

#include <memory>

#include "dragon_tensor/backend.h"

namespace dragon_tensor {

// In-memory backend (heap allocation)
class MemoryBackend : public Backend {
 public:
  MemoryBackend() = default;
  ~MemoryBackend() override = default;

  [[nodiscard]] std::shared_ptr<Buffer> allocate(size_t size_bytes,
                                                 Layout layout) override;

  void release(Buffer& buffer) override;

  [[nodiscard]] std::string name() const override { return "MemoryBackend"; }

  [[nodiscard]] bool is_writable() const override { return true; }
  [[nodiscard]] bool is_readable() const override { return true; }
};

}  // namespace dragon_tensor
