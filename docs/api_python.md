# Python API Reference

This document provides comprehensive API reference for the Dragon Tensor Python bindings.

## Table of Contents

- [Module Overview](#module-overview)
- [Tensor Classes](#tensor-classes)
- [Factory Functions](#factory-functions)
- [Tensor Methods](#tensor-methods)
- [Storage Operations](#storage-operations)
- [Examples](#examples)

---

## Module Overview

Import the module:

```python
import dragon_tensor as dt
```

The module provides tensor classes and conversion functions for integration with NumPy, Pandas, and PyTorch.

---

## Tensor Classes

Dragon Tensor provides type-specific tensor classes:

- `TensorFloat` - `Tensor<float>` (32-bit floating point)
- `TensorDouble` - `Tensor<double>` (64-bit floating point)
- `TensorInt` - `Tensor<int32_t>` (32-bit signed integer)
- `TensorLong` - `Tensor<int64_t>` (64-bit signed integer)

All classes have the same interface; only the element type differs.

---

## Factory Functions

### NumPy Conversion

#### `from_numpy_float()`
```python
dt.from_numpy_float(arr: np.ndarray[np.float32]) -> TensorFloat
```
Creates a `TensorFloat` from a NumPy float32 array.

**Parameters:**
- `arr`: NumPy array with `dtype=np.float32`

**Returns:** `TensorFloat` instance

**Note:** This function copies data. For zero-copy, see `to_numpy()` method.

#### `from_numpy_double()`
```python
dt.from_numpy_double(arr: np.ndarray[np.float64]) -> TensorDouble
```
Creates a `TensorDouble` from a NumPy float64 array.

**Parameters:**
- `arr`: NumPy array with `dtype=np.float64`

**Returns:** `TensorDouble` instance

#### `from_numpy_int()`
```python
dt.from_numpy_int(arr: np.ndarray[np.int32]) -> TensorInt
```
Creates a `TensorInt` from a NumPy int32 array.

#### `from_numpy_long()`
```python
dt.from_numpy_long(arr: np.ndarray[np.int64]) -> TensorLong
```
Creates a `TensorLong` from a NumPy int64 array.

### Pandas Conversion

#### `from_pandas_series()`
```python
dt.from_pandas_series(series: pd.Series) -> TensorFloat | TensorDouble | TensorInt | TensorLong
```
Creates a tensor from a Pandas Series.

**Parameters:**
- `series`: Pandas Series

**Returns:** Tensor instance (type depends on Series dtype)

**Supported dtypes:**
- `float32` → `TensorFloat`
- `float64` → `TensorDouble`
- `int32` → `TensorInt`
- `int64` → `TensorLong`

**Example:**
```python
import pandas as pd
series = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
tensor = dt.from_pandas_series(series)
```

#### `from_pandas_dataframe()`
```python
dt.from_pandas_dataframe(df: pd.DataFrame) -> TensorFloat | TensorDouble | TensorInt | TensorLong
```
Creates a tensor from a Pandas DataFrame (uses `.values`).

**Parameters:**
- `df`: Pandas DataFrame

**Returns:** Tensor instance (type depends on DataFrame dtype)

### PyTorch Conversion

#### `from_torch()`
```python
dt.from_torch(torch_tensor: torch.Tensor) -> TensorFloat | TensorDouble | TensorInt | TensorLong
```
Creates a tensor from a PyTorch tensor (zero-copy when PyTorch tensor is CPU and contiguous).

**Parameters:**
- `torch_tensor`: PyTorch tensor

**Returns:** Tensor instance (type depends on PyTorch dtype)

**Note:** PyTorch tensor must be on CPU and contiguous for zero-copy conversion.

**Example:**
```python
import torch
torch_tensor = torch.randn(100, dtype=torch.float64)
dt_tensor = dt.from_torch(torch_tensor)
```

---

## Tensor Methods

All tensor classes (`TensorFloat`, `TensorDouble`, `TensorInt`, `TensorLong`) share the same methods.

### Shape and Size

#### `shape() -> List[int]`
Returns the shape (dimensions) of the tensor.

**Returns:** List of dimension sizes

**Example:**
```python
tensor = dt.TensorDouble([5, 10])  # 5x10 tensor
print(tensor.shape())  # [5, 10]
```

#### `ndim() -> int`
Returns the number of dimensions.

**Returns:** Number of dimensions

#### `size() -> int`
Returns the total number of elements.

**Returns:** Total element count

#### `empty() -> bool`
Checks if the tensor is empty.

**Returns:** `True` if tensor has no elements

### Transformation

#### `reshape(shape: List[int]) -> Tensor`
Reshapes the tensor to a new shape.

**Parameters:**
- `shape`: New shape (total size must match)

**Returns:** Reshaped tensor

**Example:**
```python
tensor = dt.TensorDouble([2, 3])
reshaped = tensor.reshape([6])  # Flatten to 1D
```

#### `flatten() -> Tensor`
Flattens the tensor to 1D.

**Returns:** 1D tensor

### Element Access

#### `__getitem__(index: int) -> float`
Access element by linear index (supports Python indexing).

**Parameters:**
- `index`: Linear index

**Returns:** Element value

**Example:**
```python
tensor = dt.from_numpy_double(np.array([1.0, 2.0, 3.0]))
print(tensor[0])  # 1.0
```

#### `__setitem__(index: int, value: float) -> None`
Set element by linear index.

**Parameters:**
- `index`: Linear index
- `value`: New value

#### `at(index: int) -> float`
Bounds-checked element access.

**Parameters:**
- `index`: Linear index

**Returns:** Element value

**Raises:** `RuntimeError` if index out of bounds

#### `at(indices: List[int]) -> float`
Multi-dimensional element access.

**Parameters:**
- `indices`: List of indices, one per dimension

**Returns:** Element value

**Example:**
```python
tensor = dt.TensorDouble([3, 4, 5])
value = tensor.at([1, 2, 3])  # Access element at [1,2,3]
```

### Data Access

#### `data() -> List[float]`
Returns the underlying data as a Python list.

**Returns:** List of elements

#### `to_numpy() -> np.ndarray`
Converts tensor to NumPy array (zero-copy when possible).

**Returns:** NumPy array sharing memory with tensor

**Example:**
```python
tensor = dt.from_numpy_double(np.array([1.0, 2.0, 3.0]))
np_array = tensor.to_numpy()  # Zero-copy conversion
```

#### `to_torch() -> torch.Tensor`
Converts tensor to PyTorch tensor (zero-copy via NumPy).

**Returns:** PyTorch tensor

**Example:**
```python
tensor = dt.from_numpy_double(np.array([1.0, 2.0, 3.0]))
torch_tensor = tensor.to_torch()  # Zero-copy conversion
```

### Arithmetic Operations

All operations return new tensors (immutable operations):

#### Addition
```python
tensor + other: Tensor
tensor + scalar: Tensor
```

#### Subtraction
```python
tensor - other: Tensor
tensor - scalar: Tensor
scalar - tensor: Tensor  # Reverse subtraction
```

#### Multiplication
```python
tensor * other: Tensor
tensor * scalar: Tensor
scalar * tensor: Tensor  # Reverse multiplication
```

#### Division
```python
tensor / other: Tensor
tensor / scalar: Tensor
scalar / tensor: Tensor  # Reverse division
```

#### In-place Operations
```python
tensor += other: Tensor  # Modifies tensor in place
tensor += scalar: Tensor
tensor -= other: Tensor
tensor -= scalar: Tensor
tensor *= other: Tensor
tensor *= scalar: Tensor
tensor /= other: Tensor
tensor /= scalar: Tensor
```

**Note:** In-place operations return the modified tensor.

### Mathematical Functions

```python
tensor.abs() -> Tensor      # Absolute value
tensor.sqrt() -> Tensor     # Square root
tensor.exp() -> Tensor      # Exponential
tensor.log() -> Tensor      # Natural logarithm
tensor.pow(exponent: float) -> Tensor  # Power function
```

All operations are element-wise.

### Statistical Operations

#### Aggregate Operations (No Axis)

```python
tensor.sum() -> float       # Sum of all elements
tensor.mean() -> float      # Mean of all elements
tensor.max() -> float       # Maximum element
tensor.min() -> float       # Minimum element
tensor.std() -> float       # Standard deviation
tensor.var() -> float       # Variance
```

#### Aggregate Operations (With Axis)

```python
tensor.sum(axis: int) -> Tensor
tensor.mean(axis: int) -> Tensor
tensor.max(axis: int) -> Tensor
tensor.min(axis: int) -> Tensor
tensor.std(axis: int) -> Tensor
tensor.var(axis: int) -> Tensor
```

**Parameters:**
- `axis`: Axis to reduce over (0-based)

**Example:**
```python
tensor = dt.TensorDouble([3, 4])
col_means = tensor.mean(0)  # Mean of each column
row_sums = tensor.sum(1)     # Sum of each row
```

### Financial Operations

#### `returns() -> Tensor`
Calculates percentage returns: `(x[i] - x[i-1]) / x[i-1]`.

**Returns:** Tensor of returns (size = original size - 1)

**Example:**
```python
prices = dt.from_numpy_double(np.array([100, 102, 101, 105, 108]))
returns = prices.returns()
# Returns: [0.02, -0.0098, 0.0396, 0.0286]
```

#### Rolling Window Operations

```python
tensor.rolling_mean(window: int) -> Tensor
tensor.rolling_std(window: int) -> Tensor
tensor.rolling_sum(window: int) -> Tensor
tensor.rolling_max(window: int) -> Tensor
tensor.rolling_min(window: int) -> Tensor
```

**Parameters:**
- `window`: Window size

**Returns:** Tensor with rolling statistics

**Example:**
```python
prices = dt.from_numpy_double(np.array([100, 102, 101, 105, 108, 110]))
rolling_avg = prices.rolling_mean(3)
# 3-element rolling average
```

#### Correlation and Covariance

```python
tensor.correlation(other: Tensor) -> Tensor
tensor.covariance(other: Tensor) -> Tensor
```

**Parameters:**
- `other`: Another tensor (must have matching size)

**Returns:** Tensor containing correlation/covariance values

**Example:**
```python
asset1 = dt.from_numpy_double(np.array([100, 102, 101, 105, 108]))
asset2 = dt.from_numpy_double(np.array([50, 51, 50.5, 52.5, 54]))
corr = asset1.correlation(asset2)
cov = asset1.covariance(asset2)
```

### Slicing Operations

```python
tensor.slice(start: int, end: int) -> Tensor
tensor.slice_row(row: int) -> Tensor      # Extract row (2D only)
tensor.slice_column(col: int) -> Tensor   # Extract column (2D only)
```

**Parameters:**
- `start`, `end`: Range for 1D slicing
- `row`, `col`: Row/column index

**Returns:** New tensor with sliced data

### Matrix Operations (2D Only)

#### `transpose() -> Tensor`
Transposes a 2D matrix.

**Returns:** Transposed tensor

**Example:**
```python
matrix = dt.TensorDouble([2, 3])
transposed = matrix.transpose()  # Now 3x2
```

#### `matmul(other: Tensor) -> Tensor`
Matrix multiplication (2D tensors only).

**Parameters:**
- `other`: Right-hand side matrix

**Returns:** Matrix product

**Note:** Left tensor columns must match right tensor rows.

### Copy Operations

#### `copy() -> Tensor`
Creates a deep copy of the tensor.

**Returns:** New tensor with copied data

---

## Storage Operations

Storage operations enable persistent storage and shared-memory access (v0.2).

### File I/O

#### `save(path: str, layout: str = "row") -> None`
Saves tensor to file with versioned binary format.

**Parameters:**
- `path`: File path to save to
- `layout`: Storage layout - `"row"` or `"column"` (default: `"row"`)

**Raises:** `RuntimeError` on file I/O errors

**Example:**
```python
tensor = dt.from_numpy_double(np.random.randn(100, 1000))
tensor.save("data.dt", layout="column")
```

#### `load(path: str, mmap: bool = True) -> Tensor`
Loads tensor from file (static method).

**Parameters:**
- `path`: File path to load from
- `mmap`: If `True`, use memory-mapped I/O (default: `True`)

**Returns:** Loaded tensor

**Raises:** `RuntimeError` on file I/O or format errors

**Example:**
```python
# On TensorDouble class
loaded = dt.TensorDouble.load("data.dt", mmap=True)
```

### Shared Memory

#### `create_shared(name: str, shape: List[int], dtype: str = "", layout: str = "row") -> Tensor`
Creates a shared-memory tensor (static method).

**Parameters:**
- `name`: Shared memory segment name
- `shape`: Tensor shape
- `dtype`: Data type string (optional, determined by tensor type)
- `layout`: Storage layout - `"row"` or `"column"` (default: `"row"`)

**Returns:** Tensor backed by shared memory

**Raises:** `RuntimeError` if shared memory creation fails

**Example:**
```python
# Process 1: Create shared memory
shared = dt.TensorDouble.create_shared(
    "risk_data", shape=[252, 500], dtype="float64", layout="row"
)
```

#### `attach_shared(name: str) -> Tensor`
Attaches to an existing shared-memory tensor (static method).

**Parameters:**
- `name`: Shared memory segment name

**Returns:** Tensor backed by shared memory

**Raises:** `RuntimeError` if shared memory not found

**Example:**
```python
# Process 2: Attach to shared memory
attached = dt.TensorDouble.attach_shared("risk_data")
```

#### `detach() -> None`
Unmaps shared-memory tensor (but shared memory persists).

**Example:**
```python
shared.detach()  # Unmap, but memory remains
```

#### `destroy_shared(name: str) -> None`
Destroys a shared-memory segment (static method).

**Parameters:**
- `name`: Shared memory segment name

**Example:**
```python
dt.TensorDouble.destroy_shared("risk_data")
```

#### `flush() -> None`
Forces write-back for file-backed or memory-mapped tensors.

**Example:**
```python
mapped_tensor.flush()  # Ensure writes are visible
```

---

## Type Conversion Summary

| From | To | Function | Zero-Copy |
|------|-----|----------|-----------|
| NumPy array | Tensor | `from_numpy_*()` | ❌ (copy) |
| Tensor | NumPy array | `to_numpy()` | ✅ |
| PyTorch tensor | Tensor | `from_torch()` | ✅ (if CPU, contiguous) |
| Tensor | PyTorch tensor | `to_torch()` | ✅ |
| Pandas Series | Tensor | `from_pandas_series()` | ❌ (copy) |
| Pandas DataFrame | Tensor | `from_pandas_dataframe()` | ❌ (copy) |

---

## Error Handling

Python operations raise `RuntimeError` on failure, including:
- Invalid shape or indices
- File I/O errors
- Shared memory errors
- Shape mismatches for operations

Always use try-except blocks for error handling:

```python
try:
    tensor = dt.TensorDouble.load("data.dt")
except RuntimeError as e:
    print(f"Error loading tensor: {e}")
```

---

## Examples

See the following example files:
- `examples/basic_usage.py` - Basic operations
- `examples/financial_analysis.py` - Financial analysis examples
- `examples/integration_examples.py` - Integration with NumPy/Pandas/PyTorch

---

## Performance Notes

- **Zero-copy conversions**: `to_numpy()` and `to_torch()` provide zero-copy views when possible
- **Memory mapping**: Use `mmap=True` when loading large files for on-demand access
- **Shared memory**: Ultra-low latency for inter-process communication
- **Layout optimization**: Use `layout="column"` for column-wise queries, `layout="row"` for row-wise queries

