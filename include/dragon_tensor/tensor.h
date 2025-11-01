#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dragon_tensor/buffer.h"
#include "dragon_tensor/storage.h"

namespace dragon_tensor {

// Forward declarations
class Buffer;
enum class StorageMode;
enum class Layout;
struct TensorMeta;
// DType is now defined in storage.h

template <typename T>
class Tensor {
 public:
  // Constructors
  Tensor() = default;

  explicit Tensor(const std::vector<size_t>& shape);

  Tensor(const std::vector<size_t>& shape, const std::vector<T>& data);

  Tensor(const std::vector<size_t>& shape, T fill_value);

  Tensor(const Tensor& other);

  Tensor(Tensor&& other) noexcept;

  Tensor& operator=(const Tensor& other);

  Tensor& operator=(Tensor&& other) noexcept;

  // Shape operations
  [[nodiscard]] const std::vector<size_t>& shape() const { return shape_; }
  [[nodiscard]] size_t ndim() const { return shape_.size(); }
  [[nodiscard]] size_t size() const { return data_.size(); }
  [[nodiscard]] bool empty() const { return data_.empty(); }

  // Reshape
  [[nodiscard]] Tensor reshape(const std::vector<size_t>& new_shape) const;

  // Flatten
  [[nodiscard]] Tensor flatten() const;

  // Element access
  T& operator[](size_t index) { return data_[index]; }
  const T& operator[](size_t index) const { return data_[index]; }

  T& at(size_t index);
  const T& at(size_t index) const;

  T& at(const std::vector<size_t>& indices);
  const T& at(const std::vector<size_t>& indices) const;

  // Data access
  const std::vector<T>& data() const { return data_; }
  T* raw_data() { return data_.data(); }
  const T* raw_data() const { return data_.data(); }

  // Arithmetic operations
  [[nodiscard]] Tensor operator+(const Tensor& other) const;
  [[nodiscard]] Tensor operator+(T scalar) const;
  [[nodiscard]] Tensor operator-(const Tensor& other) const;
  [[nodiscard]] Tensor operator-(T scalar) const;
  [[nodiscard]] Tensor operator*(const Tensor& other) const;
  [[nodiscard]] Tensor operator*(T scalar) const;
  [[nodiscard]] Tensor operator/(const Tensor& other) const;
  [[nodiscard]] Tensor operator/(T scalar) const;

  Tensor& operator+=(const Tensor& other);
  Tensor& operator+=(T scalar);
  Tensor& operator-=(const Tensor& other);
  Tensor& operator-=(T scalar);
  Tensor& operator*=(const Tensor& other);
  Tensor& operator*=(T scalar);
  Tensor& operator/=(const Tensor& other);
  Tensor& operator/=(T scalar);

  // Comparison
  bool operator==(const Tensor& other) const;
  bool operator!=(const Tensor& other) const;

  // Mathematical operations
  [[nodiscard]] Tensor abs() const;
  [[nodiscard]] Tensor sqrt() const;
  [[nodiscard]] Tensor exp() const;
  [[nodiscard]] Tensor log() const;
  [[nodiscard]] Tensor pow(T exponent) const;

  // Statistical operations
  [[nodiscard]] T sum() const;
  [[nodiscard]] T mean() const;
  [[nodiscard]] T max() const;
  [[nodiscard]] T min() const;
  [[nodiscard]] T std() const;
  [[nodiscard]] T var() const;

  // Statistical operations along axis (for 2D tensors)
  [[nodiscard]] Tensor sum(size_t axis) const;
  [[nodiscard]] Tensor mean(size_t axis) const;
  [[nodiscard]] Tensor max(size_t axis) const;
  [[nodiscard]] Tensor min(size_t axis) const;
  [[nodiscard]] Tensor std(size_t axis) const;
  [[nodiscard]] Tensor var(size_t axis) const;

  // Financial operations
  [[nodiscard]] Tensor returns()
      const;  // Calculate returns: (x[i] - x[i-1]) / x[i-1]
  [[nodiscard]] Tensor rolling_mean(size_t window) const;
  [[nodiscard]] Tensor rolling_std(size_t window) const;
  [[nodiscard]] Tensor rolling_sum(size_t window) const;
  [[nodiscard]] Tensor rolling_max(size_t window) const;
  [[nodiscard]] Tensor rolling_min(size_t window) const;

  // Correlation and covariance
  [[nodiscard]] Tensor correlation(const Tensor& other) const;
  [[nodiscard]] Tensor covariance(const Tensor& other) const;

  // Slicing (for 1D and 2D)
  [[nodiscard]] Tensor slice(size_t start, size_t end) const;
  [[nodiscard]] Tensor slice_row(size_t row) const;
  [[nodiscard]] Tensor slice_column(size_t col) const;

  // Matrix operations (for 2D tensors)
  [[nodiscard]] Tensor transpose() const;
  [[nodiscard]] Tensor matmul(const Tensor& other) const;

  // Copy
  [[nodiscard]] Tensor copy() const;

  // Storage operations (new in v0.2)
  void save(std::string_view path, Layout layout = Layout::RowMajor) const;
  [[nodiscard]] static Tensor load(std::string_view path, bool mmap = true);

  // Shared memory operations
  [[nodiscard]] static Tensor create_shared(std::string_view name,
                                            const std::vector<size_t>& shape,
                                            Layout layout = Layout::RowMajor);
  [[nodiscard]] static Tensor attach_shared(std::string_view name);
  void detach();
  static void destroy_shared(std::string_view name);
  void flush();  // Force write-back for file-backed tensors

 private:
  std::vector<size_t> shape_;
  std::vector<T> data_;
  StorageMode storage_mode_ = StorageMode::InMemory;
  Layout layout_ = Layout::RowMajor;
  std::shared_ptr<Buffer> buffer_;  // Optional: for mmap/shared memory

  size_t calculate_offset(const std::vector<size_t>& indices) const;
  void validate_shape(const std::vector<size_t>& shape) const;
  bool shapes_match(const Tensor& other) const;
  bool is_broadcastable(const Tensor& other) const;
  Tensor broadcast_to(const std::vector<size_t>& target_shape) const;
};

// Type aliases for common types
using TensorFloat = Tensor<float>;
using TensorDouble = Tensor<double>;
using TensorInt = Tensor<int32_t>;
using TensorLong = Tensor<int64_t>;

}  // namespace dragon_tensor

// Template implementations are in src/tensor.cpp
// Explicit instantiations are provided for Tensor<float>, Tensor<double>,
// Tensor<int32_t>, and Tensor<int64_t>
