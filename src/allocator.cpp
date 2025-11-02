#include "dragon_tensor/allocator.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace dragon_tensor {

// ============================================================================
// HeapAllocator Implementation
// ============================================================================

size_t HeapAllocator::align_size(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

void* HeapAllocator::allocate(size_t size, size_t alignment) {
  if (size == 0) {
    return nullptr;
  }

  size_t aligned_size = align_size(size, alignment);

#ifdef _WIN32
  void* ptr = _aligned_malloc(aligned_size, alignment);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
    throw std::bad_alloc();
  }
#endif

  if (ptr == nullptr) {
    throw std::bad_alloc();
  }

  return ptr;
}

void HeapAllocator::deallocate(void* ptr, size_t /* size */) {
  if (ptr == nullptr) {
    return;
  }

#ifdef _WIN32
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

std::shared_ptr<Allocator> HeapAllocator::clone() const {
  return std::make_shared<HeapAllocator>();
}

// ============================================================================
// PoolAllocator Implementation
// ============================================================================

struct PoolAllocator::Impl {
  void* pool_ptr;
  size_t pool_size;
  size_t current_offset;
  bool owns_pool;

  Impl(size_t size) : pool_ptr(nullptr), pool_size(size), current_offset(0), owns_pool(true) {
    pool_ptr = std::malloc(pool_size);
    if (pool_ptr == nullptr) {
      throw std::bad_alloc();
    }
    std::memset(pool_ptr, 0, pool_size);
  }

  ~Impl() {
    if (owns_pool && pool_ptr != nullptr) {
      std::free(pool_ptr);
    }
  }
};

PoolAllocator::PoolAllocator(size_t pool_size) : impl_(std::make_unique<Impl>(pool_size)) {}

PoolAllocator::~PoolAllocator() = default;

void* PoolAllocator::allocate(size_t size, size_t alignment) {
  if (size == 0) {
    return nullptr;
  }

  size_t aligned_size = HeapAllocator::align_size(size, alignment);
  size_t aligned_offset = HeapAllocator::align_size(impl_->current_offset, alignment);

  // Check if there's enough space in the pool
  if (aligned_offset + aligned_size > impl_->pool_size) {
    // Pool exhausted, fall back to heap allocation
    return HeapAllocator().allocate(size, alignment);
  }

  void* ptr = static_cast<char*>(impl_->pool_ptr) + aligned_offset;
  impl_->current_offset = aligned_offset + aligned_size;

  return ptr;
}

void PoolAllocator::deallocate(void* ptr, size_t /* size */) {
  // Pool allocator doesn't free individual blocks
  // Memory is freed when the pool is destroyed
  // Note: In a production implementation, you might want to track allocations
  // and allow reuse of freed blocks
  (void)ptr;  // Suppress unused parameter warning
}

std::shared_ptr<Allocator> PoolAllocator::clone() const {
  return std::make_shared<PoolAllocator>(impl_->pool_size);
}

// ============================================================================
// AlignedAllocator Implementation
// ============================================================================

AlignedAllocator::AlignedAllocator(size_t alignment) : default_alignment_(alignment) {
  // Validate alignment is a power of 2
  if ((alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("Alignment must be a power of 2");
  }
}

void* AlignedAllocator::allocate(size_t size, size_t alignment) {
  // Use the larger of requested alignment or default alignment
  size_t effective_alignment = std::max(alignment, default_alignment_);
  return HeapAllocator().allocate(size, effective_alignment);
}

void AlignedAllocator::deallocate(void* ptr, size_t size) {
  HeapAllocator().deallocate(ptr, size);
}

std::shared_ptr<Allocator> AlignedAllocator::clone() const {
  return std::make_shared<AlignedAllocator>(default_alignment_);
}

}  // namespace dragon_tensor

