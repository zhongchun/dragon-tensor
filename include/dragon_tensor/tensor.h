#pragma once

#include <vector>
#include <string>
#include <memory>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <functional>

namespace dragon_tensor {

enum class DType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64
};

template<typename T>
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
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    
    // Reshape
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    
    // Flatten
    Tensor flatten() const;
    
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
    Tensor operator+(const Tensor& other) const;
    Tensor operator+(T scalar) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator-(T scalar) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(T scalar) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator/(T scalar) const;
    
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
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor pow(T exponent) const;
    
    // Statistical operations
    T sum() const;
    T mean() const;
    T max() const;
    T min() const;
    T std() const;
    T var() const;
    
    // Statistical operations along axis (for 2D tensors)
    Tensor sum(size_t axis) const;
    Tensor mean(size_t axis) const;
    Tensor max(size_t axis) const;
    Tensor min(size_t axis) const;
    Tensor std(size_t axis) const;
    Tensor var(size_t axis) const;
    
    // Financial operations
    Tensor returns() const;  // Calculate returns: (x[i] - x[i-1]) / x[i-1]
    Tensor rolling_mean(size_t window) const;
    Tensor rolling_std(size_t window) const;
    Tensor rolling_sum(size_t window) const;
    Tensor rolling_max(size_t window) const;
    Tensor rolling_min(size_t window) const;
    
    // Correlation and covariance
    Tensor correlation(const Tensor& other) const;
    Tensor covariance(const Tensor& other) const;
    
    // Slicing (for 1D and 2D)
    Tensor slice(size_t start, size_t end) const;
    Tensor slice_row(size_t row) const;
    Tensor slice_column(size_t col) const;
    
    // Matrix operations (for 2D tensors)
    Tensor transpose() const;
    Tensor matmul(const Tensor& other) const;
    
    // Copy
    Tensor copy() const;
    
private:
    std::vector<size_t> shape_;
    std::vector<T> data_;
    
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

} // namespace dragon_tensor

// Template implementations are in src/tensor.cpp
// Explicit instantiations are provided for Tensor<float>, Tensor<double>, 
// Tensor<int32_t>, and Tensor<int64_t>

