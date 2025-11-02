# Performance Optimizations for Dragon Tensor

This document outlines key performance optimizations that can be applied to improve the efficiency of Dragon Tensor operations.

## Table of Contents

- [Optimization Summary Table](#optimization-summary-table)
  - [Quick Reference](#quick-reference)
- [1. Memory Allocation Optimizations](#1-memory-allocation-optimizations)
- [2. In-Place Operator Optimizations](#2-in-place-operator-optimizations)
- [3. Scalar Operations Optimization](#3-scalar-operations-optimization)
- [4. Rolling Window Optimization (Sliding Window)](#4-rolling-window-optimization-sliding-window)
- [5. Matrix Multiplication Optimization](#5-matrix-multiplication-optimization)
- [6. Comparison Operations Optimization](#6-comparison-operations-optimization)
- [7. Axis Operations Cache Optimization](#7-axis-operations-cache-optimization)
- [8. Transpose Optimization](#8-transpose-optimization)
- [9. Early Exit Optimizations](#9-early-exit-optimizations)
- [10. SIMD Vectorization](#10-simd-vectorization)
- [11. Memory Pool for Small Tensors](#11-memory-pool-for-small-tensors)
- [12. Lazy Evaluation for Chained Operations](#12-lazy-evaluation-for-chained-operations)
- [Priority Ranking](#priority-ranking)
- [Implementation Notes](#implementation-notes)

## Optimization Summary Table

| # | Optimization | Priority | Effort | Expected Gain | Location |
|---|-------------|----------|--------|---------------|----------|
| 1 | Memory allocation (resize vs reserve+push_back) | High | Easy | 20-30% faster | Arithmetic ops |
| 2 | In-place operators (remove temporaries) | High | Easy | 50-70% faster | `operator+=`, etc. |
| 3 | Scalar operations (remove copies) | High | Easy | 30-40% faster | Scalar ops |
| 4 | Rolling window (sliding algorithm) | High | Medium | 5-10x faster | Rolling ops |
| 5 | Matrix multiplication (loop reordering) | Medium | Medium | 2-4x faster | `matmul()` |
| 6 | Comparison operations (`memcmp`) | Medium | Easy | 3-5x faster | `operator==` |
| 7 | Axis operations (cache optimization) | Medium | Medium | 20-30% faster | Axis reductions |
| 8 | Transpose (blocking algorithm) | Medium | Medium | 2-3x faster | `transpose()` |
| 9 | Early exit optimizations | Low | Easy | Varies | Various |
| 10 | SIMD vectorization | Low | High | 2-4x faster | All element-wise |
| 11 | Memory pool for small tensors | Low | High | 10-20% faster | Construction |
| 12 | Expression templates (lazy eval) | Low | Very High | 2-3x faster | Arithmetic ops |

### Quick Reference

**Top 4 Quick Wins** (High priority, Easy to implement):

- #1: Memory allocation optimization
- #2: In-place operators
- #3: Scalar operations
- #4: Rolling window sliding algorithm

**Total Expected Cumulative Gain**: 5-15x faster for typical workloads

## 1. Memory Allocation Optimizations

### Current Issue: Reserve + Push Back

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

## 9. Early Exit Optimizations

### Current Issue: No Short-Circuit in Comparison

**Location**: `src/tensor.cpp` - Various operations

**Already optimized**: `operator==` uses early exit. Consider adding more early exits where applicable.

---

## 10. SIMD Vectorization

### Opportunity: Use CPU SIMD Instructions

**Location**: All element-wise operations

**Approach**: Use compiler intrinsics or libraries like xsimd for:

- Arithmetic operations (`+`, `-`, `*`, `/`)
- Mathematical functions (`abs`, `sqrt`, `exp`, `log`)
- Statistical reductions (`sum`, `mean`)

**Expected Gain**: 2-4x faster on modern CPUs

---

## 11. Memory Pool for Small Tensors

### Opportunity: Reduce Allocation Overhead

**Location**: Tensor construction

**Approach**: Use memory pools for frequently allocated small tensors (e.g., < 1KB).

**Expected Gain**: 10-20% faster for many small tensor operations

---

## 12. Lazy Evaluation for Chained Operations

### Opportunity: Expression Templates

**Location**: Arithmetic operations

**Approach**: Use expression templates to defer computation until final assignment, enabling:

- Common subexpression elimination
- Fused operations (e.g., `a + b * c` without intermediate)

**Complexity**: High - requires significant refactoring
**Expected Gain**: 2-3x faster for complex expressions

---

## Priority Ranking

1. **High Priority** (Easy wins):

   - Fix `reserve()` + `push_back()` → `resize()` + assignment
   - Optimize in-place operators (remove temporary copies)
   - Optimize scalar operations (remove unnecessary copies)
   - Rolling window sliding algorithm

2. **Medium Priority** (Moderate effort):

   - Matrix multiplication loop reordering
   - Comparison operations optimization
   - Transpose blocking

3. **Low Priority** (Complex/Advanced):

   - SIMD vectorization
   - Memory pools
   - Expression templates

---

## Implementation Notes

- All optimizations should maintain API compatibility
- Add benchmark tests before and after to measure gains
- Consider compiler flags: `-O3`, `-march=native` for best performance
- Profile with tools like `perf` or `valgrind` to identify actual bottlenecks
