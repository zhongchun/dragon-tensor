# C++ API Reference

This document provides comprehensive API reference for the Dragon Tensor C++ library.

## Table of Contents

- [Namespace](#namespace)
- [Data Types](#data-types)
- [Tensor Class](#tensor-class)
  - [Type Aliases](#type-aliases)
  - [Constructors](#constructors)
  - [Shape Operations](#shape-operations)
  - [Element Access](#element-access)
  - [Data Access](#data-access)
  - [Arithmetic Operations](#arithmetic-operations)
  - [Comparison Operations](#comparison-operations)
  - [Mathematical Operations](#mathematical-operations)
  - [Statistical Operations](#statistical-operations)
  - [Financial Operations](#financial-operations)
  - [Slicing Operations](#slicing-operations)
  - [Matrix Operations](#matrix-operations)
  - [Copy Operations](#copy-operations)
  - [Storage Operations](#storage-operations)
- [Buffer Classes](#buffer-classes)
- [Storage Types](#storage-types)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Namespace

All C++ APIs are in the `dragon_tensor` namespace:

```cpp
using namespace dragon_tensor;
```

---

## Data Types

### DType

Enumeration of supported data types:

```cpp
enum class DType {
  FLOAT32,    // 32-bit floating point
  FLOAT64,    // 64-bit floating point
  INT32,      // 32-bit signed integer
  INT64,      // 64-bit signed integer
  UINT8,      // 8-bit unsigned integer
  DECIMAL64,  // Planned: fixed-point decimal (64-bit)
  DECIMAL128  // Planned: fixed-point decimal (128-bit)
};
```

### StorageMode

Storage backend mode:

```cpp
enum class StorageMode {
  InMemory,     // Standard heap-allocated memory
  MMap,         // Memory-mapped file
  SharedMemory  // POSIX shared memory
};
```

### Layout

Memory layout for tensor data:

```cpp
enum class Layout {
  RowMajor,     // C-style row-major layout
  ColumnMajor   // Fortran-style column-major layout
};
```

### TensorMeta

Optional metadata structure for descriptive information:

```cpp
struct TensorMeta {
  std::vector<std::string> dim_names;                    // Dimension names
  std::unordered_map<std::string, std::string> labels;  // Custom labels
  std::string description;                               // Description string
};
```

---

## Tensor Class

The `Tensor` class is a template class that provides multi-dimensional tensor operations.

### Template Parameters

- `T`: Element type (must be one of: `float`, `double`, `int32_t`, `int64_t`, `uint8_t`)

### Type Aliases

```cpp
using TensorFloat = Tensor<float>;   // 32-bit floating point
using TensorDouble = Tensor<double>; // 64-bit floating point
using TensorInt = Tensor<int32_t>;   // 32-bit signed integer
using TensorLong = Tensor<int64_t>;  // 64-bit signed integer
```

---

### Constructors

#### Default Constructor
```cpp
Tensor();
```
Creates an empty tensor.

**Example:**
```cpp
TensorDouble empty;
```

#### Shape Constructor
```cpp
explicit Tensor(const std::vector<size_t>& shape);
```
Creates a tensor with the specified shape, initialized to zero.

**Parameters:**
- `shape`: Vector of dimension sizes

**Example:**
```cpp
TensorDouble tensor({5, 10});  // 5x10 matrix of zeros
```

#### Shape and Data Constructor
```cpp
Tensor(const std::vector<size_t>& shape, const std::vector<T>& data);
```
Creates a tensor with the specified shape and data.

**Parameters:**
- `shape`: Vector of dimension sizes
- `data`: Vector containing tensor data (must match total size)

**Example:**
```cpp
std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
TensorDouble tensor({2, 2}, data);  // 2x2 matrix
```

#### Fill Value Constructor
```cpp
Tensor(const std::vector<size_t>& shape, T fill_value);
```
Creates a tensor with the specified shape, filled with a constant value.

**Parameters:**
- `shape`: Vector of dimension sizes
- `fill_value`: Value to fill all elements with

**Example:**
```cpp
TensorDouble tensor({3, 3}, 1.0);  // 3x3 matrix filled with 1.0
```

#### Copy/Move Constructors
```cpp
Tensor(const Tensor& other);
Tensor(Tensor&& other) noexcept;
Tensor& operator=(const Tensor& other);
Tensor& operator=(Tensor&& other) noexcept;
```
Standard copy and move semantics.

---

### Shape Operations

#### `shape()`
```cpp
const std::vector<size_t>& shape() const;
```
Returns the shape (dimensions) of the tensor.

**Returns:** Constant reference to shape vector

**Example:**
```cpp
TensorDouble tensor({2, 3});
auto shape = tensor.shape();  // Returns {2, 3}
```

#### `ndim()`
```cpp
size_t ndim() const;
```
Returns the number of dimensions.

**Returns:** Number of dimensions

**Example:**
```cpp
TensorDouble tensor({2, 3, 4});
size_t dims = tensor.ndim();  // Returns 3
```

#### `size()`
```cpp
size_t size() const;
```
Returns the total number of elements.

**Returns:** Total element count

**Example:**
```cpp
TensorDouble tensor({2, 3});
size_t total = tensor.size();  // Returns 6
```

#### `empty()`
```cpp
bool empty() const;
```
Checks if the tensor is empty.

**Returns:** `true` if tensor has no elements

#### `reshape()`
```cpp
Tensor reshape(const std::vector<size_t>& new_shape) const;
```
Reshapes the tensor to a new shape (total size must match).

**Parameters:**
- `new_shape`: New shape dimensions

**Returns:** New tensor with reshaped data

**Example:**
```cpp
TensorDouble tensor({2, 3});
auto reshaped = tensor.reshape({6});  // Flatten to 1D
```

#### `flatten()`
```cpp
Tensor flatten() const;
```
Flattens the tensor to 1D.

**Returns:** 1D tensor with all elements

**Example:**
```cpp
TensorDouble tensor({2, 3});
auto flattened = tensor.flatten();  // Returns 1D tensor of size 6
```

---

### Element Access

#### Index Operator
```cpp
T& operator[](size_t index);
const T& operator[](size_t index) const;
```
Access element by linear index (for 1D tensors or flattened access).

**Parameters:**
- `index`: Linear index

**Returns:** Reference to element

**Example:**
```cpp
TensorDouble tensor({5}, {1.0, 2.0, 3.0, 4.0, 5.0});
double value = tensor[2];  // Returns 3.0
tensor[0] = 10.0;          // Modify element
```

#### `at()` - Linear Index
```cpp
T& at(size_t index);
const T& at(size_t index) const;
```
Bounds-checked element access by linear index.

**Parameters:**
- `index`: Linear index

**Returns:** Reference to element

**Throws:** `std::runtime_error` if index out of bounds

**Example:**
```cpp
TensorDouble tensor({5}, {1.0, 2.0, 3.0, 4.0, 5.0});
double value = tensor.at(2);  // Returns 3.0
```

#### `at()` - Multi-dimensional Index
```cpp
T& at(const std::vector<size_t>& indices);
const T& at(const std::vector<size_t>& indices) const;
```
Bounds-checked element access by multi-dimensional indices.

**Parameters:**
- `indices`: Vector of indices, one per dimension

**Returns:** Reference to element

**Throws:** `std::runtime_error` if indices invalid

**Example:**
```cpp
TensorDouble tensor({3, 4, 5});
double value = tensor.at({1, 2, 3});  // Access element at [1,2,3]
```

---

### Data Access

#### `data()`
```cpp
const std::vector<T>& data() const;
```
Returns a const reference to the underlying data vector.

**Returns:** Const reference to data vector

**Example:**
```cpp
TensorDouble tensor({3}, {1.0, 2.0, 3.0});
const auto& data = tensor.data();  // Access underlying vector
```

#### `raw_data()`
```cpp
T* raw_data();
const T* raw_data() const;
```
Returns a pointer to the raw data array.

**Returns:** Pointer to data array

**Example:**
```cpp
TensorDouble tensor({3}, {1.0, 2.0, 3.0});
const double* ptr = tensor.raw_data();  // Get raw pointer
```

---

### Arithmetic Operations

#### Element-wise Operations
```cpp
Tensor operator+(const Tensor& other) const;
Tensor operator+(T scalar) const;
Tensor operator-(const Tensor& other) const;
Tensor operator-(T scalar) const;
Tensor operator*(const Tensor& other) const;
Tensor operator*(T scalar) const;
Tensor operator/(const Tensor& other) const;
Tensor operator/(T scalar) const;
```
Element-wise arithmetic operations.

**Parameters:**
- `other`: Another tensor (must have compatible shape for broadcasting)
- `scalar`: Scalar value

**Returns:** New tensor with results

**Example:**
```cpp
TensorDouble a({3}, {1.0, 2.0, 3.0});
TensorDouble b({3}, {4.0, 5.0, 6.0});
auto sum = a + b;        // Element-wise addition: {5.0, 7.0, 9.0}
auto scaled = a * 2.0;   // Scalar multiplication: {2.0, 4.0, 6.0}
auto diff = b - a;       // Element-wise subtraction: {3.0, 3.0, 3.0}
auto div = b / a;        // Element-wise division: {4.0, 2.5, 2.0}
```

#### In-place Operations
```cpp
Tensor& operator+=(const Tensor& other);
Tensor& operator+=(T scalar);
Tensor& operator-=(const Tensor& other);
Tensor& operator-=(T scalar);
Tensor& operator*=(const Tensor& other);
Tensor& operator*=(T scalar);
Tensor& operator/=(const Tensor& other);
Tensor& operator/=(T scalar);
```
In-place arithmetic operations (modify the tensor in place).

**Returns:** Reference to `*this`

**Example:**
```cpp
TensorDouble a({3}, {1.0, 2.0, 3.0});
a += 5.0;   // a becomes {6.0, 7.0, 8.0}
a *= 2.0;   // a becomes {12.0, 14.0, 16.0}
```

---

### Comparison Operations

```cpp
bool operator==(const Tensor& other) const;
bool operator!=(const Tensor& other) const;
```
Element-wise comparison.

**Returns:** `true` if all elements match (for `==`)

**Example:**
```cpp
TensorDouble a({3}, {1.0, 2.0, 3.0});
TensorDouble b({3}, {1.0, 2.0, 3.0});
bool equal = (a == b);  // Returns true
```

---

### Mathematical Operations

```cpp
Tensor abs() const;           // Absolute value
Tensor sqrt() const;          // Square root
Tensor exp() const;           // Exponential
Tensor log() const;           // Natural logarithm
Tensor pow(T exponent) const; // Power function
```
Element-wise mathematical functions.

**Returns:** New tensor with results

**Example:**
```cpp
TensorDouble tensor({3}, {-1.0, 4.0, 9.0});
auto abs_result = tensor.abs();    // {1.0, 4.0, 9.0}
auto sqrt_result = tensor.sqrt();  // {NaN, 2.0, 3.0} (sqrt of negative is NaN)
auto exp_result = tensor.exp();    // Exponential of each element
auto log_result = tensor.log();    // Natural log of each element
auto pow_result = tensor.pow(2.0); // Each element raised to power 2.0
```

---

### Statistical Operations

#### Aggregate Operations (No Axis)

```cpp
T sum() const;   // Sum of all elements
T mean() const;  // Mean of all elements
T max() const;   // Maximum element
T min() const;   // Minimum element
T std() const;   // Standard deviation
T var() const;   // Variance
```
Computes statistics over all elements.

**Returns:** Scalar result

**Example:**
```cpp
TensorDouble tensor({5}, {1.0, 2.0, 3.0, 4.0, 5.0});
double total = tensor.sum();    // 15.0
double avg = tensor.mean();     // 3.0
double max_val = tensor.max();  // 5.0
double min_val = tensor.min();  // 1.0
double std_dev = tensor.std();  // ~1.414
double variance = tensor.var(); // 2.0
```

#### Aggregate Operations (With Axis)

```cpp
Tensor sum(size_t axis) const;
Tensor mean(size_t axis) const;
Tensor max(size_t axis) const;
Tensor min(size_t axis) const;
Tensor std(size_t axis) const;
Tensor var(size_t axis) const;
```
Computes statistics along a specific axis (for 2D+ tensors).

**Parameters:**
- `axis`: Axis to reduce over (0-based)

**Returns:** Tensor with reduced dimension

**Example:**
```cpp
TensorDouble tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
// [[1, 2, 3],
//  [4, 5, 6]]

auto col_means = tensor.mean(0);  // Mean of each column: {2.5, 3.5, 4.5}
auto row_sums = tensor.sum(1);    // Sum of each row: {6.0, 15.0}
auto col_max = tensor.max(0);     // Max of each column: {4.0, 5.0, 6.0}
```

---

### Financial Operations

#### `returns()`
```cpp
Tensor returns() const;
```
Calculates percentage returns: `(x[i] - x[i-1]) / x[i-1]`.

**Returns:** Tensor of returns (size = original size - 1)

**Example:**
```cpp
TensorDouble prices({6}, {100.0, 102.0, 101.0, 105.0, 108.0, 110.0});
auto returns = prices.returns();
// Returns: {0.02, -0.0098, 0.0396, 0.0286, 0.0185}
```

#### Rolling Window Operations

```cpp
Tensor rolling_mean(size_t window) const;
Tensor rolling_std(size_t window) const;
Tensor rolling_sum(size_t window) const;
Tensor rolling_max(size_t window) const;
Tensor rolling_min(size_t window) const;
```
Compute rolling window statistics.

**Parameters:**
- `window`: Window size

**Returns:** Tensor with rolling statistics

**Example:**
```cpp
TensorDouble prices({6}, {100.0, 102.0, 101.0, 105.0, 108.0, 110.0});
auto rolling_avg = prices.rolling_mean(3);
// 3-element rolling average: {101.0, 102.67, 104.67, 107.67}

auto rolling_vol = prices.rolling_std(3);
// 3-element rolling standard deviation
```

#### `correlation()` and `covariance()`
```cpp
Tensor correlation(const Tensor& other) const;
Tensor covariance(const Tensor& other) const;
```
Compute correlation and covariance with another tensor.

**Parameters:**
- `other`: Another tensor (must have matching size)

**Returns:** Tensor containing correlation/covariance values

**Example:**
```cpp
TensorDouble asset1({5}, {100.0, 102.0, 101.0, 105.0, 108.0});
TensorDouble asset2({5}, {50.0, 51.0, 50.5, 52.5, 54.0});
auto corr = asset1.correlation(asset2);  // Correlation coefficient
auto cov = asset1.covariance(asset2);    // Covariance
```

---

### Slicing Operations

```cpp
Tensor slice(size_t start, size_t end) const;        // 1D slice
Tensor slice_row(size_t row) const;                  // Extract row (2D)
Tensor slice_column(size_t col) const;               // Extract column (2D)
```
Extract sub-tensors by slicing.

**Parameters:**
- `start`, `end`: Range for 1D slicing (end is exclusive)
- `row`, `col`: Row/column index

**Returns:** New tensor with sliced data

**Example:**
```cpp
TensorDouble tensor({6}, {10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
auto sliced = tensor.slice(1, 4);  // {20.0, 30.0, 40.0}

TensorDouble matrix({3, 4}, {...});
auto row = matrix.slice_row(1);     // Extract row 1
auto col = matrix.slice_column(2);  // Extract column 2
```

---

### Matrix Operations (2D Only)

#### `transpose()`
```cpp
Tensor transpose() const;
```
Transposes a 2D matrix.

**Returns:** Transposed tensor

**Example:**
```cpp
TensorDouble matrix({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
// [[1, 2, 3],
//  [4, 5, 6]]
auto transposed = matrix.transpose();
// [[1, 4],
//  [2, 5],
//  [3, 6]]
```

#### `matmul()`
```cpp
Tensor matmul(const Tensor& other) const;
```
Matrix multiplication (2D tensors only).

**Parameters:**
- `other`: Right-hand side matrix

**Returns:** Matrix product

**Note:** Left tensor columns must match right tensor rows.

**Example:**
```cpp
TensorDouble a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
TensorDouble b({3, 2}, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
auto result = a.matmul(b);  // 2x2 matrix product
```

---

### Copy Operations

#### `copy()`
```cpp
Tensor copy() const;
```
Creates a deep copy of the tensor.

**Returns:** New tensor with copied data

**Example:**
```cpp
TensorDouble original({3}, {1.0, 2.0, 3.0});
auto copied = original.copy();
original[0] = 99.0;  // Modifying original doesn't affect copy
```

---

### Storage Operations

#### `save()`
```cpp
void save(std::string_view path, Layout layout = Layout::RowMajor) const;
```
Saves tensor to file with versioned binary format.

**Parameters:**
- `path`: File path to save to
- `layout`: Storage layout (default: RowMajor)

**Throws:** `std::runtime_error` on file I/O errors

**Example:**
```cpp
TensorDouble tensor({100, 1000}, {...});
tensor.save("data.dt", Layout::RowMajor);
tensor.save("data_col.dt", Layout::ColumnMajor);
```

#### `load()`
```cpp
static Tensor load(std::string_view path, bool mmap = true);
```
Loads tensor from file.

**Parameters:**
- `path`: File path to load from
- `mmap`: If `true`, use memory-mapped I/O (default: `true`)

**Returns:** Loaded tensor

**Throws:** `std::runtime_error` on file I/O or format errors

**Example:**
```cpp
auto loaded = TensorDouble::load("data.dt", false);  // Load without mmap
auto mapped = TensorDouble::load("large_data.dt", true);  // Memory-mapped
```

#### `create_shared()`
```cpp
static Tensor create_shared(std::string_view name,
                            const std::vector<size_t>& shape,
                            Layout layout = Layout::RowMajor);
```
Creates a shared-memory tensor (POSIX shared memory).

**Parameters:**
- `name`: Shared memory segment name
- `shape`: Tensor shape
- `layout`: Storage layout (default: RowMajor)

**Returns:** Tensor backed by shared memory

**Throws:** `std::runtime_error` if shared memory creation fails

**Example:**
```cpp
auto shared = TensorDouble::create_shared("risk_data", {252, 500}, Layout::RowMajor);
```

#### `attach_shared()`
```cpp
static Tensor attach_shared(std::string_view name);
```
Attaches to an existing shared-memory tensor.

**Parameters:**
- `name`: Shared memory segment name

**Returns:** Tensor backed by shared memory

**Throws:** `std::runtime_error` if shared memory not found

**Example:**
```cpp
auto attached = TensorDouble::attach_shared("risk_data");
```

#### `detach()`
```cpp
void detach();
```
Unmaps shared-memory tensor (but shared memory persists).

**Example:**
```cpp
shared.detach();  // Unmap, but memory remains accessible to other processes
```

#### `destroy_shared()`
```cpp
static void destroy_shared(std::string_view name);
```
Destroys a shared-memory segment.

**Parameters:**
- `name`: Shared memory segment name

**Example:**
```cpp
TensorDouble::destroy_shared("risk_data");
```

#### `flush()`
```cpp
void flush();
```
Forces write-back for file-backed or memory-mapped tensors.

**Example:**
```cpp
mapped_tensor.flush();  // Ensure writes are visible
```

---

## Buffer Classes

Buffer classes provide memory management abstraction.

### Buffer (Base Class)

```cpp
class Buffer {
 public:
  virtual ~Buffer() = default;
  virtual void* data() = 0;
  virtual const void* data() const = 0;
  virtual size_t size_bytes() const = 0;
  virtual void flush() {}
  virtual void detach() {}
};
```

Base class for all buffer types.

### MemoryBuffer

In-memory buffer using heap allocation.

```cpp
class MemoryBuffer : public Buffer {
 public:
  template <typename T>
  MemoryBuffer(const T* src, size_t count);
  
  template <typename T>
  MemoryBuffer(size_t count);
  
  explicit MemoryBuffer(size_t size_bytes);
  
  void* data() override;
  const void* data() const override;
  size_t size_bytes() const override;
};
```

### MMapBuffer

Memory-mapped file buffer.

```cpp
class MMapBuffer : public Buffer {
 public:
  MMapBuffer(const std::string& path, size_t offset, size_t size_bytes);
  ~MMapBuffer() override;
  
  MMapBuffer(MMapBuffer&& other) noexcept;
  MMapBuffer& operator=(MMapBuffer&& other) noexcept;
  
  void* data() override;
  const void* data() const override;
  size_t size_bytes() const override;
  void flush() override;
  void detach() override;
};
```

### SharedMemoryBuffer

POSIX shared memory buffer.

```cpp
class SharedMemoryBuffer : public Buffer {
 public:
  static std::shared_ptr<SharedMemoryBuffer> create(const std::string& name,
                                                    size_t size_bytes);
  static std::shared_ptr<SharedMemoryBuffer> attach(const std::string& name);
  static void destroy(const std::string& name);
  
  ~SharedMemoryBuffer() override;
  
  void* data() override;
  const void* data() const override;
  size_t size_bytes() const override;
  void detach() override;
  
  const std::string& name() const;
};
```

**Static Methods:**
- `create()`: Creates new shared memory segment
- `attach()`: Attaches to existing shared memory segment
- `destroy()`: Destroys shared memory segment

---

## Storage Types

### TensorHeader

File format header structure:

```cpp
struct TensorHeader {
  static constexpr uint32_t MAGIC = 0x44544E53;  // 'DTNS'
  static constexpr uint32_t VERSION = 1;
  
  uint32_t magic;
  uint32_t version;
  uint32_t ndim;
  uint32_t dtype;
  uint32_t layout;    // 0=RowMajor, 1=ColumnMajor
  uint32_t endian;    // 0=little, 1=big
  uint64_t data_offset;
  uint64_t checksum;  // CRC64 checksum
  // Shape array follows (ndim * uint64_t)
};
```

### Helper Functions

```cpp
bool is_little_endian();
size_t calculate_header_size(uint32_t ndim);
uint32_t calculate_checksum(const void* data, size_t size);
```

---

## Error Handling

Most operations throw `std::runtime_error` on failure, including:
- Invalid shape or indices
- File I/O errors
- Shared memory errors
- Shape mismatches for operations
- Memory allocation failures

Always wrap operations in try-catch blocks for production code:

```cpp
try {
    TensorDouble tensor = TensorDouble::load("data.dt");
    // Use tensor...
} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

---

## Examples

See `examples/basic_example.cpp` for comprehensive C++ usage examples covering all API features.
