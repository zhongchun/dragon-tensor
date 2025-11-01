#include "dragon_tensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "dragon_tensor/io.h"

namespace dragon_tensor {

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
  validate_shape(shape);
  size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                      std::multiplies<size_t>());
  data_.resize(total_size, T(0));
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, const std::vector<T>& data)
    : shape_(shape), data_(data) {
  validate_shape(shape);
  size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                      std::multiplies<size_t>());
  if (data.size() != total_size) {
    throw std::runtime_error("Data size does not match shape");
  }
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, T fill_value)
    : shape_(shape) {
  validate_shape(shape);
  size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                      std::multiplies<size_t>());
  data_.resize(total_size, fill_value);
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : shape_(other.shape_), data_(other.data_) {}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
  if (this != &other) {
    shape_ = other.shape_;
    data_ = other.data_;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
  }
  return *this;
}

template <typename T>
void Tensor<T>::validate_shape(const std::vector<size_t>& shape) const {
  if (shape.empty()) {
    throw std::runtime_error("Shape cannot be empty");
  }
  for (size_t dim : shape) {
    if (dim == 0) {
      throw std::runtime_error("Shape dimensions must be > 0");
    }
  }
}

template <typename T>
size_t Tensor<T>::calculate_offset(const std::vector<size_t>& indices) const {
  if (indices.size() != shape_.size()) {
    throw std::runtime_error("Index dimensions do not match tensor dimensions");
  }

  size_t offset = 0;
  size_t stride = 1;
  for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
    if (indices[i] >= shape_[i]) {
      throw std::runtime_error("Index out of bounds");
    }
    offset += indices[i] * stride;
    stride *= shape_[i];
  }
  return offset;
}

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<size_t>& new_shape) const {
  size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                    size_t(1), std::multiplies<size_t>());
  if (new_size != data_.size()) {
    throw std::runtime_error("Cannot reshape: total size mismatch");
  }

  Tensor result = *this;
  result.shape_ = new_shape;
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::flatten() const {
  Tensor result = *this;
  result.shape_ = {data_.size()};
  return result;
}

template <typename T>
T& Tensor<T>::at(size_t index) {
  if (index >= data_.size()) {
    throw std::runtime_error("Index out of bounds");
  }
  return data_[index];
}

template <typename T>
const T& Tensor<T>::at(size_t index) const {
  if (index >= data_.size()) {
    throw std::runtime_error("Index out of bounds");
  }
  return data_[index];
}

template <typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices) {
  return data_[calculate_offset(indices)];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const {
  return data_[calculate_offset(indices)];
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
  if (!shapes_match(other) && !is_broadcastable(other)) {
    throw std::runtime_error("Shape mismatch for addition");
  }

  Tensor result;
  if (shapes_match(other)) {
    result.shape_ = shape_;
    result.data_.reserve(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
      result.data_.push_back(data_[i] + other.data_[i]);
    }
  } else {
    // Broadcasting
    auto broadcasted = other.broadcast_to(shape_);
    result = *this + broadcasted;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(T scalar) const {
  Tensor result = *this;
  for (auto& val : result.data_) {
    val += scalar;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor& other) const {
  if (!shapes_match(other) && !is_broadcastable(other)) {
    throw std::runtime_error("Shape mismatch for subtraction");
  }

  Tensor result;
  if (shapes_match(other)) {
    result.shape_ = shape_;
    result.data_.reserve(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
      result.data_.push_back(data_[i] - other.data_[i]);
    }
  } else {
    auto broadcasted = other.broadcast_to(shape_);
    result = *this - broadcasted;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(T scalar) const {
  Tensor result = *this;
  for (auto& val : result.data_) {
    val -= scalar;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor& other) const {
  if (!shapes_match(other) && !is_broadcastable(other)) {
    throw std::runtime_error("Shape mismatch for multiplication");
  }

  Tensor result;
  if (shapes_match(other)) {
    result.shape_ = shape_;
    result.data_.reserve(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
      result.data_.push_back(data_[i] * other.data_[i]);
    }
  } else {
    auto broadcasted = other.broadcast_to(shape_);
    result = *this * broadcasted;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(T scalar) const {
  Tensor result = *this;
  for (auto& val : result.data_) {
    val *= scalar;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor& other) const {
  if (!shapes_match(other) && !is_broadcastable(other)) {
    throw std::runtime_error("Shape mismatch for division");
  }

  Tensor result;
  if (shapes_match(other)) {
    result.shape_ = shape_;
    result.data_.reserve(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
      if (other.data_[i] == T(0)) {
        throw std::runtime_error("Division by zero");
      }
      result.data_.push_back(data_[i] / other.data_[i]);
    }
  } else {
    auto broadcasted = other.broadcast_to(shape_);
    result = *this / broadcasted;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(T scalar) const {
  if (scalar == T(0)) {
    throw std::runtime_error("Division by zero");
  }
  Tensor result = *this;
  for (auto& val : result.data_) {
    val /= scalar;
  }
  return result;
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor& other) {
  *this = *this + other;
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(T scalar) {
  for (auto& val : data_) {
    val += scalar;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor& other) {
  *this = *this - other;
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(T scalar) {
  for (auto& val : data_) {
    val -= scalar;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor& other) {
  *this = *this * other;
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(T scalar) {
  for (auto& val : data_) {
    val *= scalar;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor& other) {
  *this = *this / other;
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(T scalar) {
  if (scalar == T(0)) {
    throw std::runtime_error("Division by zero");
  }
  for (auto& val : data_) {
    val /= scalar;
  }
  return *this;
}

template <typename T>
bool Tensor<T>::operator==(const Tensor& other) const {
  if (shape_ != other.shape_ || data_.size() != other.data_.size()) {
    return false;
  }
  for (size_t i = 0; i < data_.size(); ++i) {
    if (data_[i] != other.data_[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Tensor<T>::operator!=(const Tensor& other) const {
  return !(*this == other);
}

template <typename T>
bool Tensor<T>::shapes_match(const Tensor& other) const {
  return shape_ == other.shape_;
}

template <typename T>
bool Tensor<T>::is_broadcastable(const Tensor& other) const {
  // Simple broadcasting: check if one is scalar-like or shapes can be aligned
  if (other.shape_.size() == 1 && other.shape_[0] == 1) {
    return true;  // Scalar-like
  }
  // More sophisticated broadcasting can be added here
  return false;
}

template <typename T>
Tensor<T> Tensor<T>::broadcast_to(
    const std::vector<size_t>& target_shape) const {
  // Simple broadcasting implementation
  if (shape_ == target_shape) {
    return *this;
  }
  // For now, return copy if shapes don't match exactly
  // Full broadcasting implementation would go here
  throw std::runtime_error("Broadcasting not fully implemented");
}

template <typename T>
Tensor<T> Tensor<T>::abs() const {
  Tensor result = *this;
  for (auto& val : result.data_) {
    if constexpr (std::is_signed_v<T>) {
      val = std::abs(val);
    }
    // For unsigned types, abs is a no-op (already non-negative)
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::sqrt() const {
  Tensor result = *this;
  for (auto& val : result.data_) {
    val = std::sqrt(val);
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::exp() const {
  Tensor result = *this;
  for (auto& val : result.data_) {
    val = std::exp(val);
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::log() const {
  Tensor result = *this;
  for (auto& val : result.data_) {
    if (val <= T(0)) {
      throw std::runtime_error("Cannot take log of non-positive value");
    }
    val = std::log(val);
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::pow(T exponent) const {
  Tensor result = *this;
  for (auto& val : result.data_) {
    val = std::pow(val, exponent);
  }
  return result;
}

template <typename T>
T Tensor<T>::sum() const {
  return std::accumulate(data_.begin(), data_.end(), T(0));
}

template <typename T>
T Tensor<T>::mean() const {
  if (data_.empty()) {
    return T(0);
  }
  return sum() / static_cast<T>(data_.size());
}

template <typename T>
T Tensor<T>::max() const {
  if (data_.empty()) {
    throw std::runtime_error("Cannot find max of empty tensor");
  }
  return *std::max_element(data_.begin(), data_.end());
}

template <typename T>
T Tensor<T>::min() const {
  if (data_.empty()) {
    throw std::runtime_error("Cannot find min of empty tensor");
  }
  return *std::min_element(data_.begin(), data_.end());
}

template <typename T>
T Tensor<T>::var() const {
  if (data_.empty()) {
    return T(0);
  }
  T m = mean();
  T variance = T(0);
  for (const auto& val : data_) {
    T diff = val - m;
    variance += diff * diff;
  }
  return variance / static_cast<T>(data_.size());
}

template <typename T>
T Tensor<T>::std() const {
  return std::sqrt(var());
}

// Axis operations for 2D tensors
template <typename T>
Tensor<T> Tensor<T>::sum(size_t axis) const {
  if (ndim() != 2) {
    throw std::runtime_error("Axis operations only supported for 2D tensors");
  }
  if (axis >= 2) {
    throw std::runtime_error("Axis out of bounds");
  }

  Tensor result;
  if (axis == 0) {  // Sum along columns (reduce rows)
    result.shape_ = {shape_[1]};
    result.data_.resize(shape_[1], T(0));
    for (size_t i = 0; i < shape_[0]; ++i) {
      for (size_t j = 0; j < shape_[1]; ++j) {
        result.data_[j] += data_[i * shape_[1] + j];
      }
    }
  } else {  // axis == 1, Sum along rows (reduce columns)
    result.shape_ = {shape_[0]};
    result.data_.resize(shape_[0], T(0));
    for (size_t i = 0; i < shape_[0]; ++i) {
      for (size_t j = 0; j < shape_[1]; ++j) {
        result.data_[i] += data_[i * shape_[1] + j];
      }
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::mean(size_t axis) const {
  Tensor summed = sum(axis);
  size_t divisor = (axis == 0) ? shape_[0] : shape_[1];
  for (auto& val : summed.data_) {
    val /= static_cast<T>(divisor);
  }
  return summed;
}

template <typename T>
Tensor<T> Tensor<T>::max(size_t axis) const {
  if (ndim() != 2) {
    throw std::runtime_error("Axis operations only supported for 2D tensors");
  }
  if (axis >= 2) {
    throw std::runtime_error("Axis out of bounds");
  }

  Tensor result;
  if (axis == 0) {
    result.shape_ = {shape_[1]};
    result.data_.resize(shape_[1]);
    for (size_t j = 0; j < shape_[1]; ++j) {
      T max_val = data_[j];
      for (size_t i = 1; i < shape_[0]; ++i) {
        max_val = std::max(max_val, data_[i * shape_[1] + j]);
      }
      result.data_[j] = max_val;
    }
  } else {
    result.shape_ = {shape_[0]};
    result.data_.resize(shape_[0]);
    for (size_t i = 0; i < shape_[0]; ++i) {
      T max_val = data_[i * shape_[1]];
      for (size_t j = 1; j < shape_[1]; ++j) {
        max_val = std::max(max_val, data_[i * shape_[1] + j]);
      }
      result.data_[i] = max_val;
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::min(size_t axis) const {
  if (ndim() != 2) {
    throw std::runtime_error("Axis operations only supported for 2D tensors");
  }
  if (axis >= 2) {
    throw std::runtime_error("Axis out of bounds");
  }

  Tensor result;
  if (axis == 0) {
    result.shape_ = {shape_[1]};
    result.data_.resize(shape_[1]);
    for (size_t j = 0; j < shape_[1]; ++j) {
      T min_val = data_[j];
      for (size_t i = 1; i < shape_[0]; ++i) {
        min_val = std::min(min_val, data_[i * shape_[1] + j]);
      }
      result.data_[j] = min_val;
    }
  } else {
    result.shape_ = {shape_[0]};
    result.data_.resize(shape_[0]);
    for (size_t i = 0; i < shape_[0]; ++i) {
      T min_val = data_[i * shape_[1]];
      for (size_t j = 1; j < shape_[1]; ++j) {
        min_val = std::min(min_val, data_[i * shape_[1] + j]);
      }
      result.data_[i] = min_val;
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::std(size_t axis) const {
  Tensor result = var(axis);
  for (auto& val : result.data_) {
    val = std::sqrt(val);
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::var(size_t axis) const {
  if (ndim() != 2) {
    throw std::runtime_error("Axis operations only supported for 2D tensors");
  }

  Tensor means = mean(axis);
  Tensor result;

  if (axis == 0) {
    result.shape_ = {shape_[1]};
    result.data_.resize(shape_[1], T(0));
    for (size_t j = 0; j < shape_[1]; ++j) {
      T variance = T(0);
      for (size_t i = 0; i < shape_[0]; ++i) {
        T diff = data_[i * shape_[1] + j] - means.data_[j];
        variance += diff * diff;
      }
      result.data_[j] = variance / static_cast<T>(shape_[0]);
    }
  } else {
    result.shape_ = {shape_[0]};
    result.data_.resize(shape_[0], T(0));
    for (size_t i = 0; i < shape_[0]; ++i) {
      T variance = T(0);
      for (size_t j = 0; j < shape_[1]; ++j) {
        T diff = data_[i * shape_[1] + j] - means.data_[i];
        variance += diff * diff;
      }
      result.data_[i] = variance / static_cast<T>(shape_[1]);
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::returns() const {
  if (ndim() != 1 || data_.size() < 2) {
    throw std::runtime_error(
        "Returns calculation requires 1D tensor with at least 2 elements");
  }

  Tensor result({data_.size() - 1});
  for (size_t i = 1; i < data_.size(); ++i) {
    // Check if denominator is effectively zero
    T denom = data_[i - 1];
    bool is_zero = false;
    if constexpr (std::is_signed_v<T>) {
      is_zero = std::abs(denom) < T(1e-10);
    } else {
      // For unsigned types, just check if it's zero
      is_zero = (denom == T(0));
    }

    if (is_zero) {
      result.data_[i - 1] = T(0);
    } else {
      result.data_[i - 1] = (data_[i] - data_[i - 1]) / data_[i - 1];
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::rolling_mean(size_t window) const {
  if (ndim() != 1) {
    throw std::runtime_error(
        "Rolling operations only supported for 1D tensors");
  }
  if (window > data_.size()) {
    throw std::runtime_error("Window size exceeds tensor size");
  }

  Tensor result({data_.size() - window + 1});
  for (size_t i = 0; i <= data_.size() - window; ++i) {
    T sum = T(0);
    for (size_t j = 0; j < window; ++j) {
      sum += data_[i + j];
    }
    result.data_[i] = sum / static_cast<T>(window);
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::rolling_std(size_t window) const {
  if (ndim() != 1) {
    throw std::runtime_error(
        "Rolling operations only supported for 1D tensors");
  }
  if (window > data_.size()) {
    throw std::runtime_error("Window size exceeds tensor size");
  }

  Tensor result({data_.size() - window + 1});
  for (size_t i = 0; i <= data_.size() - window; ++i) {
    // Calculate mean
    T mean = T(0);
    for (size_t j = 0; j < window; ++j) {
      mean += data_[i + j];
    }
    mean /= static_cast<T>(window);

    // Calculate variance
    T variance = T(0);
    for (size_t j = 0; j < window; ++j) {
      T diff = data_[i + j] - mean;
      variance += diff * diff;
    }
    variance /= static_cast<T>(window);
    result.data_[i] = std::sqrt(variance);
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::rolling_sum(size_t window) const {
  if (ndim() != 1) {
    throw std::runtime_error(
        "Rolling operations only supported for 1D tensors");
  }
  if (window > data_.size()) {
    throw std::runtime_error("Window size exceeds tensor size");
  }

  Tensor result({data_.size() - window + 1});
  for (size_t i = 0; i <= data_.size() - window; ++i) {
    T sum = T(0);
    for (size_t j = 0; j < window; ++j) {
      sum += data_[i + j];
    }
    result.data_[i] = sum;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::rolling_max(size_t window) const {
  if (ndim() != 1) {
    throw std::runtime_error(
        "Rolling operations only supported for 1D tensors");
  }
  if (window > data_.size()) {
    throw std::runtime_error("Window size exceeds tensor size");
  }

  Tensor result({data_.size() - window + 1});
  for (size_t i = 0; i <= data_.size() - window; ++i) {
    T max_val = data_[i];
    for (size_t j = 1; j < window; ++j) {
      max_val = std::max(max_val, data_[i + j]);
    }
    result.data_[i] = max_val;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::rolling_min(size_t window) const {
  if (ndim() != 1) {
    throw std::runtime_error(
        "Rolling operations only supported for 1D tensors");
  }
  if (window > data_.size()) {
    throw std::runtime_error("Window size exceeds tensor size");
  }

  Tensor result({data_.size() - window + 1});
  for (size_t i = 0; i <= data_.size() - window; ++i) {
    T min_val = data_[i];
    for (size_t j = 1; j < window; ++j) {
      min_val = std::min(min_val, data_[i + j]);
    }
    result.data_[i] = min_val;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::correlation(const Tensor& other) const {
  if (!shapes_match(other)) {
    throw std::runtime_error(
        "Tensors must have matching shapes for correlation");
  }
  if (data_.size() < 2) {
    throw std::runtime_error("Need at least 2 elements for correlation");
  }

  T mean_x = mean();
  T mean_y = other.mean();

  T cov = T(0);
  T var_x = T(0);
  T var_y = T(0);

  for (size_t i = 0; i < data_.size(); ++i) {
    T diff_x = data_[i] - mean_x;
    T diff_y = other.data_[i] - mean_y;
    cov += diff_x * diff_y;
    var_x += diff_x * diff_x;
    var_y += diff_y * diff_y;
  }

  T std_x = std::sqrt(var_x / static_cast<T>(data_.size()));
  T std_y = std::sqrt(var_y / static_cast<T>(data_.size()));

  if (std_x == T(0) || std_y == T(0)) {
    return Tensor({1}, T(0));
  }

  T corr = (cov / static_cast<T>(data_.size())) / (std_x * std_y);
  return Tensor({1}, corr);
}

template <typename T>
Tensor<T> Tensor<T>::covariance(const Tensor& other) const {
  if (!shapes_match(other)) {
    throw std::runtime_error(
        "Tensors must have matching shapes for covariance");
  }
  if (data_.size() < 2) {
    throw std::runtime_error("Need at least 2 elements for covariance");
  }

  T mean_x = mean();
  T mean_y = other.mean();

  T cov = T(0);
  for (size_t i = 0; i < data_.size(); ++i) {
    cov += (data_[i] - mean_x) * (other.data_[i] - mean_y);
  }

  return Tensor({1}, cov / static_cast<T>(data_.size()));
}

template <typename T>
Tensor<T> Tensor<T>::slice(size_t start, size_t end) const {
  if (ndim() != 1) {
    throw std::runtime_error("Slice only supported for 1D tensors");
  }
  if (start >= data_.size() || end > data_.size() || start >= end) {
    throw std::runtime_error("Invalid slice indices");
  }

  std::vector<T> sliced_data(data_.begin() + start, data_.begin() + end);
  return Tensor({end - start}, sliced_data);
}

template <typename T>
Tensor<T> Tensor<T>::slice_row(size_t row) const {
  if (ndim() != 2) {
    throw std::runtime_error("slice_row only supported for 2D tensors");
  }
  if (row >= shape_[0]) {
    throw std::runtime_error("Row index out of bounds");
  }

  std::vector<T> row_data(shape_[1]);
  for (size_t j = 0; j < shape_[1]; ++j) {
    row_data[j] = data_[row * shape_[1] + j];
  }
  return Tensor({shape_[1]}, row_data);
}

template <typename T>
Tensor<T> Tensor<T>::slice_column(size_t col) const {
  if (ndim() != 2) {
    throw std::runtime_error("slice_column only supported for 2D tensors");
  }
  if (col >= shape_[1]) {
    throw std::runtime_error("Column index out of bounds");
  }

  std::vector<T> col_data(shape_[0]);
  for (size_t i = 0; i < shape_[0]; ++i) {
    col_data[i] = data_[i * shape_[1] + col];
  }
  return Tensor({shape_[0]}, col_data);
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
  if (ndim() != 2) {
    throw std::runtime_error("Transpose only supported for 2D tensors");
  }

  Tensor result({shape_[1], shape_[0]});
  for (size_t i = 0; i < shape_[0]; ++i) {
    for (size_t j = 0; j < shape_[1]; ++j) {
      result.data_[j * shape_[0] + i] = data_[i * shape_[1] + j];
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor& other) const {
  if (ndim() != 2 || other.ndim() != 2) {
    throw std::runtime_error("matmul only supported for 2D tensors");
  }
  if (shape_[1] != other.shape_[0]) {
    throw std::runtime_error(
        "Matrix dimensions incompatible for multiplication");
  }

  Tensor result({shape_[0], other.shape_[1]});
  for (size_t i = 0; i < shape_[0]; ++i) {
    for (size_t j = 0; j < other.shape_[1]; ++j) {
      T sum = T(0);
      for (size_t k = 0; k < shape_[1]; ++k) {
        sum += data_[i * shape_[1] + k] * other.data_[k * other.shape_[1] + j];
      }
      result.data_[i * other.shape_[1] + j] = sum;
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::copy() const {
  return Tensor(*this);
}

// Storage operations implementation
template <typename T>
void Tensor<T>::save(std::string_view path, Layout layout) const {
  io::save_tensor(*this, path, layout);
}

template <typename T>
Tensor<T> Tensor<T>::load(std::string_view path, bool mmap) {
  return io::load_tensor<T>(path, mmap);
}

template <typename T>
Tensor<T> Tensor<T>::create_shared(std::string_view name,
                                   const std::vector<size_t>& shape,
                                   Layout layout) {
  // Calculate size needed
  size_t total_elements = 1;
  for (size_t dim : shape) {
    total_elements *= dim;
  }
  size_t size_bytes = total_elements * sizeof(T);

  // Create shared memory buffer
  std::string name_str(
      name);  // Convert string_view to string for shm operations
  auto buffer = SharedMemoryBuffer::create(name_str, size_bytes);

  // Create tensor wrapping the shared memory
  Tensor<T> tensor;
  tensor.shape_ = shape;
  tensor.storage_mode_ = StorageMode::SharedMemory;
  tensor.layout_ = layout;
  tensor.buffer_ = buffer;

  // Initialize data vector to point to shared memory
  T* shared_data = static_cast<T*>(buffer->data());
  tensor.data_.assign(shared_data, shared_data + total_elements);

  return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::attach_shared(std::string_view name) {
  // Attach to existing shared memory
  std::string name_str(
      name);  // Convert string_view to string for shm operations
  auto buffer = SharedMemoryBuffer::attach(name_str);

  // We need to read shape from shared memory header (simplified for now)
  // For a full implementation, we'd store metadata in shared memory
  throw std::runtime_error(
      "attach_shared: Full implementation requires metadata in shared memory");

  // Placeholder - would need to read shape from shared memory header
  Tensor<T> tensor;
  tensor.buffer_ = buffer;
  tensor.storage_mode_ = StorageMode::SharedMemory;
  return tensor;
}

template <typename T>
void Tensor<T>::detach() {
  if (buffer_) {
    buffer_->detach();
    buffer_.reset();
  }
}

template <typename T>
void Tensor<T>::destroy_shared(std::string_view name) {
  std::string name_str(
      name);  // Convert string_view to string for shm operations
  SharedMemoryBuffer::destroy(name_str);
}

template <typename T>
void Tensor<T>::flush() {
  if (buffer_) {
    buffer_->flush();
  }
}

}  // namespace dragon_tensor

// Explicit template instantiation for common types
namespace dragon_tensor {
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int32_t>;
template class Tensor<int64_t>;
template class Tensor<uint8_t>;
}  // namespace dragon_tensor
