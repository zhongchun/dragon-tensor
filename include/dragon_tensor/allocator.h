#pragma once

#include <cstddef>
#include <memory>

namespace dragon_tensor {

// Allocator interface for flexible memory management strategies
class Allocator {
 public:
  virtual ~Allocator() = default;

  // Allocate memory with optional alignment
  [[nodiscard]] virtual void* allocate(size_t size,
                                      size_t alignment = alignof(std::max_align_t)) = 0;

  // Deallocate memory
  virtual void deallocate(void* ptr, size_t size) = 0;

  // Clone allocator (for creating independent copies)
  [[nodiscard]] virtual std::shared_ptr<Allocator> clone() const = 0;
};

// Standard heap allocator
class HeapAllocator : public Allocator {
 public:
  void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) override;
  void deallocate(void* ptr, size_t size) override;
  std::shared_ptr<Allocator> clone() const override;

  // Helper function for alignment calculations (used by PoolAllocator)
  static size_t align_size(size_t size, size_t alignment);
};

// Pool allocator for small tensors (reduces allocation overhead)
class PoolAllocator : public Allocator {
 public:
  explicit PoolAllocator(size_t pool_size = 1024 * 1024);  // Default 1MB pool
  ~PoolAllocator();

  void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) override;
  void deallocate(void* ptr, size_t size) override;
  std::shared_ptr<Allocator> clone() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Aligned allocator for SIMD operations (ensures proper alignment)
class AlignedAllocator : public Allocator {
 public:
  explicit AlignedAllocator(size_t alignment = 64);  // Default 64-byte alignment for AVX

  void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) override;
  void deallocate(void* ptr, size_t size) override;
  std::shared_ptr<Allocator> clone() const override;

 private:
  size_t default_alignment_;
};

}  // namespace dragon_tensor

