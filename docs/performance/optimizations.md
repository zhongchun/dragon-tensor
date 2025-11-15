# Performance Optimizations for Dragon Tensor

This document outlines key performance optimizations that can be applied to improve the efficiency of Dragon Tensor operations. The optimizations are organized according to the 5-layer architecture defined in the requirements document (v0.3): Python API Layer, Interop Layer, Tensor Core, Buffer Layer, and Backend Abstraction Layer.

## Table of Contents

- [Optimization Summary Table](#optimization-summary-table)
  - [Quick Reference](#quick-reference)
- [1. Memory Allocation Optimizations](#1-memory-allocation-optimizations) (Tensor Core)
- [2. In-Place Operator Optimizations](#2-in-place-operator-optimizations) (Tensor Core)
- [3. Scalar Operations Optimization](#3-scalar-operations-optimization) (Tensor Core)
- [4. Rolling Window Optimization (Sliding Window)](#4-rolling-window-optimization-sliding-window) (Tensor Core)
- [5. Matrix Multiplication Optimization](#5-matrix-multiplication-optimization) (Tensor Core)
- [6. Comparison Operations Optimization](#6-comparison-operations-optimization) (Tensor Core)
- [7. Axis Operations Cache Optimization](#7-axis-operations-cache-optimization) (Tensor Core)
- [8. Transpose Optimization](#8-transpose-optimization) (Tensor Core)
- [9. Zero-Copy Arrow Integration](#9-zero-copy-arrow-integration) (Interop Layer)
- [10. Allocator Optimization](#10-allocator-optimization) (Buffer Layer)
- [11. Early Exit Optimizations](#11-early-exit-optimizations) (Tensor Core)
- [12. SIMD Vectorization](#12-simd-vectorization) (Tensor Core)
- [13. Expression Templates (Lazy Evaluation)](#13-expression-templates-lazy-evaluation) (Tensor Core)
- [14. Backend Selection Optimization](#14-backend-selection-optimization) (Backend Layer)
- [15. Build System Optimizations](#15-build-system-optimizations) (Build System) ✅ **Implemented**
- [Priority Ranking](#priority-ranking)
- [Build-Time Optimizations](#build-time-optimizations)
- [Implementation Notes](#implementation-notes)

## Optimization Summary Table

| # | Optimization | Priority | Effort | Expected Gain | Layer | Location |
|---|-------------|----------|--------|---------------|-------|----------|
| 1 | Memory allocation (resize vs reserve+push_back) | High | Easy | 20-30% faster | Tensor Core | Arithmetic ops |
| 2 | In-place operators (remove temporaries) | High | Easy | 50-70% faster | Tensor Core | `operator+=`, etc. |
| 3 | Scalar operations (remove copies) | High | Easy | 30-40% faster | Tensor Core | Scalar ops |
| 4 | Rolling window (sliding algorithm) | High | Medium | 5-10x faster | Tensor Core | Rolling ops |
| 5 | Matrix multiplication (loop reordering) | Medium | Medium | 2-4x faster | Tensor Core | `matmul()` |
| 6 | Comparison operations (`memcmp`) | Medium | Easy | 3-5x faster | Tensor Core | `operator==` |
| 7 | Axis operations (cache optimization) | Medium | Medium | 20-30% faster | Tensor Core | Axis reductions |
| 8 | Transpose (blocking algorithm) | Medium | Medium | 2-3x faster | Tensor Core | `transpose()` |
| 9 | Zero-copy Arrow integration | Medium | Medium | Eliminates copies | Interop Layer | Arrow conversion |
| 10 | Allocator optimization (pool for small tensors) | Medium | Medium | 10-20% faster | Buffer Layer | Construction |
| 11 | Early exit optimizations | Low | Easy | Varies | Tensor Core | Various |
| 12 | SIMD vectorization | Low | High | 2-4x faster | Tensor Core | All element-wise |
| 13 | Expression templates (lazy eval) | Low | Very High | 2-3x faster | Tensor Core | Arithmetic ops |
| 14 | Backend selection optimization | Low | Medium | 10-15% faster | Backend Layer | Storage ops |
| 15 | Build system optimizations | ✅ | ✅ | 5-10x faster builds | Build System | build.sh, setup.py |

### Quick Reference

**Top 4 Quick Wins** (High priority, Easy to implement):

- #1: Memory allocation optimization
- #2: In-place operators
- #3: Scalar operations
- #4: Rolling window sliding algorithm

**Total Expected Cumulative Gain**: 5-15x faster for typical workloads

## 1. Memory Allocation Optimizations

### Current Issue: Reserve + Push Back

**Layer**: Tensor Core

**Location**: `src/tensor.cpp` - Arithmetic operations (`operator+`, `operator-`, `operator*`, `operator/`)

**Problem**: Using `reserve()` + `push_back()` causes multiple reallocations and is slower than direct assignment.

```cpp
// Current (inefficient)
result.data_.reserve(data_.size());
for (size_t i = 0; i < data_.size(); ++i) {
  result.data_.push_back(data_[i] + other.data_[i]);
}
```

**Optimization**: Use `resize()` + direct assignment for better performance:

```cpp
// Optimized
result.data_.resize(data_.size());
for (size_t i = 0; i < data_.size(); ++i) {
  result.data_[i] = data_[i] + other.data_[i];
}
```

**Expected Gain**: 20-30% faster for large tensors

---

## 2. In-Place Operator Optimizations

### Current Issue: Unnecessary Temporary Creation

**Layer**: Tensor Core

**Location**: `src/tensor.cpp` - In-place operators (`operator+=`, `operator-=`, `operator*=`, `operator/=`)

**Problem**: `operator+=(other)` calls `*this = *this + other`, creating a full copy.

```cpp
// Current (inefficient)
Tensor<T>& Tensor<T>::operator+=(const Tensor& other) {
  *this = *this + other;  // Creates temporary!
  return *this;
}
```

**Optimization**: Direct in-place modification:

```cpp
// Optimized
Tensor<T>& Tensor<T>::operator+=(const Tensor& other) {
  if (!shapes_match(other)) {
    throw std::runtime_error("Shape mismatch for in-place addition");
  }
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] += other.data_[i];
  }
  return *this;
}
```

**Expected Gain**: 50-70% faster, eliminates temporary allocations

---

## 3. Scalar Operations Optimization

### Current Issue: Unnecessary Copy

**Layer**: Tensor Core

**Location**: `src/tensor.cpp` - Scalar operators (`operator+(scalar)`, etc.)

**Problem**: `Tensor result = *this` copies the entire tensor before modification.

```cpp
// Current (inefficient)
Tensor<T> Tensor<T>::operator+(T scalar) const {
  Tensor result = *this;  // Full copy!
  for (auto& val : result.data_) {
    val += scalar;
  }
  return result;
}
```

**Optimization**: Create result with correct size and fill directly:

```cpp
// Optimized
Tensor<T> Tensor<T>::operator+(T scalar) const {
  Tensor result;
  result.shape_ = shape_;
  result.data_.resize(data_.size());
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] + scalar;
  }
  return result;
}
```

**Expected Gain**: 30-40% faster, better memory locality

---

## 4. Rolling Window Optimization (Sliding Window)

### Current Issue: O(n×w) Complexity

**Layer**: Tensor Core

**Location**: `src/tensor.cpp` - Rolling operations (`rolling_mean`, `rolling_sum`, etc.)

**Problem**: Recalculates sum for each window position, O(n×w) complexity.

```cpp
// Current (inefficient) - O(n×w)
for (size_t i = 0; i <= data_.size() - window; ++i) {
  T sum = T(0);
  for (size_t j = 0; j < window; ++j) {
    sum += data_[i + j];  // Recalculates everything
  }
  result.data_[i] = sum / static_cast<T>(window);
}
```

**Optimization**: Use sliding window technique, O(n) complexity:

```cpp
// Optimized - O(n)
Tensor<T> Tensor<T>::rolling_mean(size_t window) const {
  // ... validation ...
  Tensor result({data_.size() - window + 1});
  
  // Calculate first window sum
  T sum = T(0);
  for (size_t j = 0; j < window; ++j) {
    sum += data_[j];
  }
  result.data_[0] = sum / static_cast<T>(window);
  
  // Slide window: subtract outgoing, add incoming
  for (size_t i = 1; i <= data_.size() - window; ++i) {
    sum = sum - data_[i - 1] + data_[i + window - 1];
    result.data_[i] = sum / static_cast<T>(window);
  }
  return result;
}
```

**Expected Gain**: 5-10x faster for large windows

---

## 5. Matrix Multiplication Optimization

### Current Issue: Naive O(n³) Algorithm

**Layer**: Tensor Core

**Location**: `src/tensor.cpp` - `matmul()`

**Problem**: Basic triple-loop implementation with poor cache locality.

**Optimizations**:

1. **Loop Reordering**: Swap inner loops for better cache performance
2. **Blocking/Tiling**: Process in blocks to fit in L1/L2 cache
3. **SIMD**: Use vectorized instructions for inner products

```cpp
// Current loop order (i, j, k) - not cache-friendly
for (size_t i = 0; i < shape_[0]; ++i) {
  for (size_t j = 0; j < other.shape_[1]; ++j) {
    for (size_t k = 0; k < shape_[1]; ++k) {
      sum += data_[i * shape_[1] + k] * other.data_[k * other.shape_[1] + j];
    }
  }
}

// Optimized: Better loop order (i, k, j) for cache locality
Tensor result({shape_[0], other.shape_[1]});
result.data_.resize(shape_[0] * other.shape_[1], T(0));
for (size_t i = 0; i < shape_[0]; ++i) {
  for (size_t k = 0; k < shape_[1]; ++k) {
    T a_ik = data_[i * shape_[1] + k];
    for (size_t j = 0; j < other.shape_[1]; ++j) {
      result.data_[i * other.shape_[1] + j] += 
          a_ik * other.data_[k * other.shape_[1] + j];
    }
  }
}
```

**Expected Gain**: 2-4x faster for medium-sized matrices

---

## 6. Comparison Operations Optimization

### Current Issue: Element-by-Element Comparison

**Layer**: Tensor Core

**Location**: `src/tensor.cpp` - `operator==`

**Problem**: Loop with early exit is still slower than memcmp for simple types.

**Optimization**: Use `std::equal` or `memcmp` for POD types:

```cpp
// Optimized
bool Tensor<T>::operator==(const Tensor& other) const {
  if (shape_ != other.shape_ || data_.size() != other.data_.size()) {
    return false;
  }
  // For POD types, use memcmp for better performance
  if constexpr (std::is_trivially_copyable_v<T>) {
    return std::memcmp(data_.data(), other.data_.data(), 
                       data_.size() * sizeof(T)) == 0;
  } else {
    return std::equal(data_.begin(), data_.end(), other.data_.begin());
  }
}
```

**Expected Gain**: 3-5x faster for large tensors

---

## 7. Axis Operations Cache Optimization

### Current Issue: Poor Cache Locality

**Layer**: Tensor Core

**Location**: `src/tensor.cpp` - Axis operations (`sum(axis)`, `mean(axis)`, etc.)

**Problem**: When reducing along axis=0, accessing column-major causes cache misses.

**Optimization**: Better memory access patterns:

```cpp
// For axis=0 (reduce rows), current code accesses data column-wise
// This is actually good for column-major, but we can optimize further
// by processing in blocks to improve cache reuse
```

**Expected Gain**: 20-30% faster for large 2D tensors

---

## 8. Transpose Optimization

### Current Issue: Naive Element-by-Element Copy

**Layer**: Tensor Core

**Location**: `src/tensor.cpp` - `transpose()`

**Problem**: Non-contiguous memory access patterns cause cache misses.

**Optimization**: Block-based transpose for better cache performance:

```cpp
// Use blocking for cache-friendly transpose
constexpr size_t BLOCK_SIZE = 64;  // Cache line friendly
// ... block-based transpose implementation ...
```

**Expected Gain**: 2-3x faster for large matrices

---

## 9. Zero-Copy Arrow Integration

### Current Issue: Data Copying for Arrow Conversion

**Layer**: Interop Layer

**Location**: `src/interop/arrow_interop.cpp` (future) - Arrow conversion functions

**Problem**: Converting between Dragon Tensor and Arrow Arrays may require copying data when memory layouts are incompatible.

**Optimization**: Leverage Arrow's zero-copy capabilities:

```cpp
// Optimized: Check if Arrow array memory can be directly wrapped
Tensor from_arrow(const arrow::Array& array) {
  // Check if Arrow buffer is compatible (contiguous, correct dtype, aligned)
  if (array.data()->buffers.size() == 2 && 
      array.IsNull() == 0 && 
      is_contiguous_layout(array)) {
    // Zero-copy: wrap Arrow memory directly
    auto buffer = std::make_shared<ArrowBuffer>(array.data()->buffers[1]);
    return Tensor(buffer, shape_from_arrow(array), dtype_from_arrow(array));
  }
  // Fallback: copy for incompatible layouts
  return copy_from_arrow(array);
}

// Optimized: Create Arrow array view when possible
std::shared_ptr<arrow::Array> to_arrow() const {
  if (is_contiguous() && dtype_matches_arrow()) {
    // Zero-copy: Arrow array wraps our memory
    return arrow::MakeArray(std::make_shared<ArrowArrayData>(buffer_));
  }
  // Fallback: copy for incompatible layouts
  return copy_to_arrow();
}
```

**Expected Gain**: Eliminates data copying overhead for compatible layouts, enables seamless Parquet integration

---

## 10. Allocator Optimization

### Opportunity: Optimized Memory Allocation Strategies

**Layer**: Buffer Layer

**Location**: `src/buffer.cpp` - Allocator implementations

**Problem**: Generic heap allocation has overhead for frequently allocated small tensors.

**Optimization**: Implement pool allocator for small tensors:

```cpp
// Pool allocator for small tensors (< 1KB)
class PoolAllocator : public Allocator {
private:
  struct Pool {
    std::vector<std::unique_ptr<char[]>> blocks_;
    std::vector<size_t> free_list_;
    size_t block_size_;
  };
  std::unordered_map<size_t, Pool> pools_;
  
public:
  void* allocate(size_t size, size_t alignment) override {
    if (size < 1024) {
      // Use pool for small allocations
      auto& pool = pools_[size];
      if (!pool.free_list_.empty()) {
        size_t idx = pool.free_list_.back();
        pool.free_list_.pop_back();
        return pool.blocks_[idx].get();
      }
      // Allocate new block
      pool.blocks_.emplace_back(std::make_unique<char[]>(size));
      return pool.blocks_.back().get();
    }
    // Use standard allocator for large tensors
    return std::aligned_alloc(alignment, size);
  }
};
```

**Expected Gain**: 10-20% faster for many small tensor operations, reduced fragmentation

---

## 11. Early Exit Optimizations

### Current Issue: No Short-Circuit in Comparison

**Layer**: Tensor Core

**Location**: `src/tensor.cpp` - Various operations

**Already optimized**: `operator==` uses early exit. Consider adding more early exits where applicable.

---

## 12. SIMD Vectorization

### Opportunity: Use CPU SIMD Instructions

**Layer**: Tensor Core

**Location**: All element-wise operations in `src/tensor.cpp`

**Approach**: Use compiler intrinsics or libraries like xsimd for:

- Arithmetic operations (`+`, `-`, `*`, `/`)
- Mathematical functions (`abs`, `sqrt`, `exp`, `log`)
- Statistical reductions (`sum`, `mean`)

**Example**:

```cpp
#include <xsimd/xsimd.hpp>

// SIMD-optimized element-wise addition
template<typename T>
void add_simd(const T* a, const T* b, T* result, size_t n) {
  using batch_type = xsimd::batch<T>;
  size_t simd_size = batch_type::size;
  size_t i = 0;
  
  for (; i + simd_size <= n; i += simd_size) {
    auto va = xsimd::load_aligned(&a[i]);
    auto vb = xsimd::load_aligned(&b[i]);
    auto vresult = va + vb;
    vresult.store_aligned(&result[i]);
  }
  
  // Handle remaining elements
  for (; i < n; ++i) {
    result[i] = a[i] + b[i];
  }
}
```

**Expected Gain**: 2-4x faster on modern CPUs with AVX/AVX2 support

---

## 13. Expression Templates (Lazy Evaluation)

### Opportunity: Defer Computation for Complex Expressions

**Layer**: Tensor Core

**Location**: Arithmetic operations in `src/tensor.cpp`

**Approach**: Use expression templates to defer computation until final assignment, enabling:

- Common subexpression elimination
- Fused operations (e.g., `a + b * c` without intermediate)
- Loop fusion for chained operations

**Complexity**: Very High - requires significant refactoring of Tensor class and operations

**Expected Gain**: 2-3x faster for complex expressions with multiple chained operations

---

## 14. Backend Selection Optimization

### Opportunity: Intelligent Backend Selection

**Layer**: Backend Abstraction Layer

**Location**: `src/backend/backend_factory.cpp` (future) - Backend selection logic

**Problem**: Choosing the optimal backend (Memory, MMap, SharedMemory, Arrow) based on use case characteristics.

**Optimization**: Implement intelligent backend selection:

```cpp
// Optimize backend selection based on data characteristics
std::shared_ptr<Backend> select_backend(const BackendConfig& config) {
  if (config.size > LARGE_THRESHOLD && config.persistent) {
    // Large, persistent data -> use MMapBackend
    return std::make_shared<MMapBackend>(config.path);
  }
  if (config.shared && config.multi_process) {
    // Multi-process access -> use SharedMemoryBackend
    return std::make_shared<SharedMemoryBackend>(config.name);
  }
  if (config.columnar_query && config.arrow_compatible) {
    // Columnar analytics -> use ArrowBackend
    return std::make_shared<ArrowBackend>(config.schema);
  }
  // Default to MemoryBackend for small, temporary data
  return std::make_shared<MemoryBackend>();
}
```

**Expected Gain**: 10-15% faster through optimal storage backend selection based on access patterns

---

## Priority Ranking

1. **High Priority** (Easy wins):

   - Fix `reserve()` + `push_back()` → `resize()` + assignment (Tensor Core)
   - Optimize in-place operators (remove temporary copies) (Tensor Core)
   - Optimize scalar operations (remove unnecessary copies) (Tensor Core)
   - Rolling window sliding algorithm (Tensor Core)

2. **Medium Priority** (Moderate effort):

   - Matrix multiplication loop reordering (Tensor Core)
   - Comparison operations optimization (Tensor Core)
   - Transpose blocking (Tensor Core)
   - Zero-copy Arrow integration (Interop Layer)
   - Allocator optimization with pool allocator (Buffer Layer)

3. **Low Priority** (Complex/Advanced):

   - Early exit optimizations (Tensor Core)
   - SIMD vectorization (Tensor Core)
   - Expression templates (Tensor Core)
   - Backend selection optimization (Backend Layer)

---

## Build-Time Optimizations

### 15. Build System Optimizations

**Layer**: Build System

**Location**: `build.sh`, `setup.py`, `CMakeLists.txt`

**Status**: ✅ **Implemented**

Dragon Tensor includes several build-time optimizations to speed up compilation and improve developer productivity:

#### ccache (Compiler Cache)

**What it does**: Caches compilation results to avoid recompiling unchanged files.

**Benefits**:
- **10-100x faster rebuilds** for unchanged files
- Automatic cache invalidation on source changes
- Works transparently with CMake

**Usage**:
```bash
# Install ccache
brew install ccache  # macOS
apt-get install ccache  # Linux

# Automatically enabled by build.sh
./build.sh
```

**Configuration**: Set `USE_CCACHE=0` to disable if needed.

#### Ninja Build System

**What it does**: Faster build system than traditional Make.

**Benefits**:
- **2-3x faster builds** compared to Make
- Better parallel build performance
- More efficient dependency tracking

**Usage**:
```bash
# Install Ninja
brew install ninja  # macOS
apt-get install ninja-build  # Linux

# Automatically used by build.sh if available
./build.sh
```

#### Parallel Builds

**What it does**: Uses all available CPU cores for compilation.

**Benefits**:
- **N× faster** where N is number of CPU cores
- Auto-detects optimal number of jobs
- Can be overridden with `-j N` flag

**Usage**:
```bash
# Auto-detect cores (default)
./build.sh

# Override with specific number
./build.sh -j 8
```

#### Release Optimizations

**What it does**: Applies aggressive compiler optimizations for production builds.

**Benefits**:
- **20-50% faster runtime** performance
- Smaller binary size with `-DNDEBUG`
- Optimized for production use

**Configuration**: Automatically applied for Release builds:
- `-O3`: Maximum optimization level
- `-DNDEBUG`: Disables assertions for production

#### Two-Stage Build Architecture

**What it does**: Separates C++ compilation (CMake) from Python packaging (setup.py).

**Benefits**:
- **Faster Python builds**: No C++ compilation in setup.py
- **Better caching**: CMake handles C++ with proper dependency tracking
- **Consistent builds**: Same C++ build process for all targets
- **Cleaner separation**: C++ compilation vs Python packaging

**How it works**:
1. CMake builds C++ extension module
2. `setup.py` finds and copies pre-built extension
3. No C++ compilation during Python package build

**Expected Gain**: 5-10x faster Python package builds, especially for rebuilds

---

## Implementation Notes

- All optimizations should maintain API compatibility
- Add benchmark tests before and after to measure gains
- Consider compiler flags: `-O3`, `-march=native` for best performance
- Profile with tools like `perf` or `valgrind` to identify actual bottlenecks
- **Build optimizations are already implemented** - use `./build.sh` to benefit from them
