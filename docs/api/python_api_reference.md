# Python API Reference

This document provides comprehensive API reference for the Dragon Tensor Python bindings.

## Table of Contents

- [Module Overview](#module-overview)
- [Tensor Classes](#tensor-classes)
- [Factory Functions](#factory-functions)
- [Tensor Methods](#tensor-methods)
  - [Shape and Size](#shape-and-size)
  - [Transformation](#transformation)
  - [Element Access](#element-access)
  - [Data Access](#data-access)
  - [Arithmetic Operations](#arithmetic-operations)
  - [Comparison Operations](#comparison-operations)
  - [Mathematical Functions](#mathematical-functions)
  - [Statistical Operations](#statistical-operations)
  - [Financial Operations](#financial-operations)
  - [Slicing Operations](#slicing-operations)
  - [Matrix Operations](#matrix-operations)
  - [Copy Operations](#copy-operations)
- [Storage Operations](#storage-operations)
  - [File I/O](#file-io)
  - [Shared Memory](#shared-memory)
- [Type Conversion Summary](#type-conversion-summary)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Performance Notes](#performance-notes)

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

#### `from_numpy()`
```python
dt.from_numpy(arr: np.ndarray) -> TensorFloat | TensorDouble | TensorInt | TensorLong
```
Creates a tensor from a NumPy array. Automatically detects dtype and returns the appropriate tensor type.

**Parameters:**
- `arr`: NumPy array (any dtype)

**Returns:** Tensor instance (type depends on array dtype)

**Supported dtypes:**
- `float32` → `TensorFloat`
- `float64` → `TensorDouble`
- `int32` → `TensorInt`
- `int64` → `TensorLong`
- Other types → converted to `float64` and returns `TensorDouble`

**Example:**
```python
import numpy as np

# Automatic dtype detection
arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
tensor = dt.from_numpy(arr)  # Returns TensorDouble

arr_int = np.array([1, 2, 3], dtype=np.int32)
tensor_int = dt.from_numpy(arr_int)  # Returns TensorInt

# From Python list (converted to numpy first)
tensor = dt.from_numpy([1.0, 2.0, 3.0])  # Returns TensorDouble
```

**Note:** This function copies data. For zero-copy conversion back to NumPy, use `to_numpy()` method.

#### `to_numpy()`
```python
dt.to_numpy(tensor: Tensor) -> np.ndarray
```
Converts tensor to NumPy array (zero-copy when possible).

**Parameters:**
- `tensor`: Dragon Tensor

**Returns:** NumPy array sharing memory with tensor (zero-copy)

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
np_array = dt.to_numpy(tensor)  # Zero-copy conversion
```

### Pandas Conversion

#### `from_pandas()`
```python
dt.from_pandas(obj: pd.Series | pd.DataFrame) -> TensorFloat | TensorDouble | TensorInt | TensorLong
```
Creates a tensor from a Pandas Series or DataFrame.

**Parameters:**
- `obj`: Pandas Series or DataFrame

**Returns:** Tensor instance (type depends on Series/DataFrame dtype)

**Supported dtypes:**
- `float32` → `TensorFloat`
- `float64` → `TensorDouble`
- `int32` → `TensorInt`
- `int64` → `TensorLong`

**Example:**
```python
import pandas as pd

# From Series
series = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
tensor = dt.from_pandas(series)

# From DataFrame
df = pd.DataFrame({'price': [100, 102, 101, 105]})
tensor = dt.from_pandas(df['price'])
```

#### `to_pandas()`
```python
dt.to_pandas(tensor: Tensor, index=None, columns=None) -> pd.Series | pd.DataFrame
```
Converts tensor to Pandas Series or DataFrame.

**Parameters:**
- `tensor`: Dragon Tensor
- `index`: Optional index for Series/DataFrame
- `columns`: Optional column names for DataFrame

**Returns:** Pandas Series (1D) or DataFrame (2D)

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
series = dt.to_pandas(tensor, index=pd.date_range('2024-01-01', periods=3))
```

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

# GPU tensors must be moved to CPU first
gpu_tensor = torch.randn(100, device='cuda')
cpu_tensor = gpu_tensor.cpu()
dt_tensor = dt.from_torch(cpu_tensor)
```

#### `to_torch()`
```python
dt.to_torch(tensor: Tensor, device=None, dtype=None) -> torch.Tensor
```
Converts tensor to PyTorch tensor (zero-copy via NumPy).

**Parameters:**
- `tensor`: Dragon Tensor
- `device`: Optional device (default: CPU, zero-copy)
- `dtype`: Optional dtype (default: matches tensor dtype, zero-copy)

**Returns:** PyTorch tensor (zero-copy when device=None and dtype matches)

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
torch_tensor = dt.to_torch(tensor)  # Zero-copy conversion

# Convert to GPU (requires copying)
torch_gpu = dt.to_torch(tensor, device='cuda')
```

### Direct Factory Functions

For advanced use cases, you can also use type-specific factory functions:

```python
dt.from_pandas_series(series: pd.Series) -> Tensor
dt.from_pandas_dataframe(df: pd.DataFrame) -> Tensor
dt.from_torch(torch_tensor: torch.Tensor) -> Tensor
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
tensor = dt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
print(tensor.shape())  # [2, 3]
```

#### `ndim() -> int`
Returns the number of dimensions.

**Returns:** Number of dimensions

**Example:**
```python
tensor = dt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
print(tensor.ndim())  # 2
```

#### `size() -> int`
Returns the total number of elements.

**Returns:** Total element count

**Example:**
```python
tensor = dt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
print(tensor.size())  # 6
```

#### `empty() -> bool`
Checks if the tensor is empty.

**Returns:** `True` if tensor has no elements

---

### Transformation

#### `reshape(shape: List[int]) -> Tensor`
Reshapes the tensor to a new shape.

**Parameters:**
- `shape`: New shape (total size must match)

**Returns:** Reshaped tensor

**Example:**
```python
tensor = dt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
reshaped = tensor.reshape([6])  # Flatten to 1D
```

#### `flatten() -> Tensor`
Flattens the tensor to 1D.

**Returns:** 1D tensor

**Example:**
```python
tensor = dt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
flattened = tensor.flatten()  # 1D tensor of size 6
```

---

### Element Access

#### `__getitem__(index: int) -> float`
Access element by linear index (supports Python indexing).

**Parameters:**
- `index`: Linear index

**Returns:** Element value

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
print(tensor[0])  # 1.0
print(tensor[2])  # 3.0
```

#### `__setitem__(index: int, value: float) -> None`
Set element by linear index.

**Parameters:**
- `index`: Linear index
- `value`: New value

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
tensor[0] = 10.0
print(tensor[0])  # 10.0
```

#### `at(index: int) -> float`
Bounds-checked element access.

**Parameters:**
- `index`: Linear index

**Returns:** Element value

**Raises:** `RuntimeError` if index out of bounds

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
value = tensor.at(1)  # 2.0
```

#### `at(indices: List[int]) -> float`
Multi-dimensional element access.

**Parameters:**
- `indices`: List of indices, one per dimension

**Returns:** Element value

**Example:**
```python
tensor = dt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
value = tensor.at([1, 2])  # 6 (element at row 1, column 2)
```

---

### Data Access

#### `data() -> List[float]`
Returns the underlying data as a Python list.

**Returns:** List of elements

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
data_list = tensor.data()  # [1.0, 2.0, 3.0]
```

#### `to_numpy() -> np.ndarray`
Converts tensor to NumPy array (zero-copy when possible).

**Returns:** NumPy array sharing memory with tensor

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
np_array = tensor.to_numpy()  # Zero-copy conversion
```

---

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
```

**Example:**
```python
a = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
b = dt.from_numpy(np.array([4.0, 5.0, 6.0]))

sum_result = a + b        # [5.0, 7.0, 9.0]
scaled = a * 2.0          # [2.0, 4.0, 6.0]
product = a * b           # [4.0, 10.0, 18.0]
quotient = b / a          # [4.0, 2.5, 2.0]
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

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
tensor += 5.0   # tensor becomes [6.0, 7.0, 8.0]
tensor *= 2.0   # tensor becomes [12.0, 14.0, 16.0]
```

---

### Comparison Operations

```python
tensor == other: bool
tensor != other: bool
```

Element-wise comparison.

**Returns:** `True` if all elements match (for `==`)

**Example:**
```python
a = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
b = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
print(a == b)  # True
```

---

### Mathematical Functions

```python
tensor.abs() -> Tensor      # Absolute value
tensor.sqrt() -> Tensor     # Square root
tensor.exp() -> Tensor      # Exponential
tensor.log() -> Tensor      # Natural logarithm
tensor.pow(exponent: float) -> Tensor  # Power function
```

All operations are element-wise.

**Example:**
```python
tensor = dt.from_numpy(np.array([4.0, 9.0, 16.0]))
sqrt_result = tensor.sqrt()    # [2.0, 3.0, 4.0]
pow_result = tensor.pow(2.0)   # [16.0, 81.0, 256.0]
```

---

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

**Example:**
```python
tensor = dt.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
print(tensor.sum())    # 15.0
print(tensor.mean())   # 3.0
print(tensor.max())    # 5.0
print(tensor.min())    # 1.0
print(tensor.std())    # ~1.414
print(tensor.var())    # 2.0
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

**Returns:** Tensor with reduced dimension

**Example:**
```python
tensor = dt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
# [[1, 2, 3],
#  [4, 5, 6]]

col_means = tensor.mean(0)  # Mean of each column: [2.5, 3.5, 4.5]
row_sums = tensor.sum(1)    # Sum of each row: [6.0, 15.0]
col_max = tensor.max(0)     # Max of each column: [4, 5, 6]
```

---

### Financial Operations

#### `returns() -> Tensor`
Calculates percentage returns: `(x[i] - x[i-1]) / x[i-1]`.

**Returns:** Tensor of returns (size = original size - 1)

**Example:**
```python
prices = dt.from_numpy(np.array([100.0, 102.0, 101.0, 105.0, 108.0]))
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
- `window`: Window size (positional argument, not keyword)

**Returns:** Tensor with rolling statistics

**Example:**
```python
prices = dt.from_numpy(np.array([100.0, 102.0, 101.0, 105.0, 108.0, 110.0]))
rolling_avg = prices.rolling_mean(3)  # 3-element rolling average
rolling_vol = prices.rolling_std(3)   # 3-element rolling standard deviation
```

**Note:** Window size must be passed as a positional argument, not a keyword argument.

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
asset1 = dt.from_numpy(np.array([100.0, 102.0, 101.0, 105.0, 108.0]))
asset2 = dt.from_numpy(np.array([50.0, 51.0, 50.5, 52.5, 54.0]))
corr = asset1.correlation(asset2)
cov = asset1.covariance(asset2)
```

---

### Slicing Operations

```python
tensor.slice(start: int, end: int) -> Tensor
tensor.slice_row(row: int) -> Tensor      # Extract row (2D only)
tensor.slice_column(col: int) -> Tensor   # Extract column (2D only)
```

**Parameters:**
- `start`, `end`: Range for 1D slicing (end is exclusive)
- `row`, `col`: Row/column index

**Returns:** New tensor with sliced data

**Example:**
```python
tensor = dt.from_numpy(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
sliced = tensor.slice(1, 4)  # [20.0, 30.0, 40.0]

matrix = dt.from_numpy(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
row = matrix.slice_row(1)     # Extract row 1: [5, 6, 7, 8]
col = matrix.slice_column(2)  # Extract column 2: [3, 7, 11]
```

---

### Matrix Operations (2D Only)

#### `transpose() -> Tensor`
Transposes a 2D matrix.

**Returns:** Transposed tensor

**Example:**
```python
matrix = dt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
transposed = matrix.transpose()
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

#### `matmul(other: Tensor) -> Tensor`
Matrix multiplication (2D tensors only).

**Parameters:**
- `other`: Right-hand side matrix

**Returns:** Matrix product

**Note:** Left tensor columns must match right tensor rows.

**Example:**
```python
a = dt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
b = dt.from_numpy(np.array([[7, 8], [9, 10], [11, 12]]))
result = a.matmul(b)  # 2x2 matrix product
```

---

### Copy Operations

#### `copy() -> Tensor`
Creates a deep copy of the tensor.

**Returns:** New tensor with copied data

**Example:**
```python
original = dt.from_numpy(np.array([1.0, 2.0, 3.0]))
copied = original.copy()
original[0] = 99.0  # Modifying original doesn't affect copy
```

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
tensor = dt.from_numpy(np.random.randn(100, 1000))
dt.save(tensor, "data.dt", layout="column")
```

#### `load(path: str, mmap: bool = True) -> Tensor`
Loads tensor from file.

**Parameters:**
- `path`: File path to load from
- `mmap`: If `True`, use memory-mapped I/O (default: `True`)

**Returns:** Loaded tensor

**Raises:** `RuntimeError` on file I/O or format errors

**Note:** Currently defaults to `TensorDouble`. For other types, use the specific tensor class's load method directly.

**Example:**
```python
# Using convenience function (defaults to TensorDouble)
loaded = dt.load("data.dt", mmap=True)

# Using specific tensor type
loaded = dt.TensorDouble.load("data.dt", mmap=False)
```

#### `open(path: str, mmap: bool = True) -> ContextManager[Tensor]`
Context manager for loading tensors from files. Automatically handles resource cleanup.

**Parameters:**
- `path`: File path to open
- `mmap`: If `True`, use memory-mapped I/O

**Returns:** Context manager yielding a tensor

**Example:**
```python
with dt.open("large_data.dt", mmap=True) as tensor:
    result = tensor.sum()
    # Tensor is automatically detached when exiting context
```

#### `save_parquet(path: str) -> None`
Saves tensor to Parquet file via Arrow.

**Parameters:**
- `path`: Parquet file path

**Raises:** `NotImplementedError` (Parquet support planned for future version)

#### `load_parquet(path: str, mmap: bool = True) -> Tensor`
Loads tensor from Parquet file.

**Parameters:**
- `path`: Parquet file path
- `mmap`: If `True`, use memory-mapped I/O

**Raises:** `NotImplementedError` (Parquet support planned for future version)

---

### Shared Memory

#### `create_shared(name: str, shape: List[int], dtype: str = "float64", layout: str = "row") -> Tensor`
Creates a shared-memory tensor (static method).

**Parameters:**
- `name`: Shared memory segment name
- `shape`: Tensor shape
- `dtype`: Data type string - `"float32"`, `"float64"`, `"int32"`, `"int64"` (default: `"float64"`)
- `layout`: Storage layout - `"row"` or `"column"` (default: `"row"`)

**Returns:** Tensor backed by shared memory

**Raises:** `RuntimeError` if shared memory creation fails

**Example:**
```python
# Process 1: Create shared memory
shared = dt.create_shared(
    "risk_data", shape=[252, 500], dtype="float64", layout="row"
)
```

#### `attach_shared(name: str) -> Tensor`
Attaches to an existing shared-memory tensor.

**Parameters:**
- `name`: Shared memory segment name

**Returns:** Tensor backed by shared memory

**Raises:** `RuntimeError` if shared memory not found

**Example:**
```python
# Process 2: Attach to shared memory
attached = dt.attach_shared("risk_data")
```

#### `detach(tensor: Tensor) -> None`
Unmaps shared-memory tensor (but shared memory persists).

**Parameters:**
- `tensor`: Shared memory tensor

**Example:**
```python
dt.detach(shared)  # Unmap, but memory remains
```

#### `destroy_shared(name: str) -> None`
Destroys a shared-memory segment.

**Parameters:**
- `name`: Shared memory segment name

**Example:**
```python
dt.destroy_shared("risk_data")
```

#### `flush(tensor: Tensor) -> None`
Forces write-back for file-backed or memory-mapped tensors.

**Parameters:**
- `tensor`: File-backed or memory-mapped tensor

**Example:**
```python
dt.flush(mapped_tensor)  # Ensure writes are visible
```

---

## Type Conversion Summary

| From | To | Function | Zero-Copy |
|------|-----|----------|-----------|
| NumPy array | Tensor | `from_numpy()` | ❌ (copy) |
| Tensor | NumPy array | `to_numpy()` | ✅ |
| PyTorch tensor | Tensor | `from_torch()` | ✅ (if CPU, contiguous) |
| Tensor | PyTorch tensor | `to_torch()` | ✅ (if device=None, dtype matches) |
| Pandas Series | Tensor | `from_pandas()` | ❌ (copy) |
| Pandas DataFrame | Tensor | `from_pandas()` | ❌ (copy) |
| Tensor | Pandas Series/DataFrame | `to_pandas()` | ❌ (copy) |

---

## Error Handling

Python operations raise `RuntimeError` on failure, including:
- Invalid shape or indices
- File I/O errors
- Shared memory errors
- Shape mismatches for operations
- Window size exceeds tensor size (for rolling operations)
- Insufficient elements for returns calculation (requires at least 2 elements)

Always use try-except blocks for error handling:

```python
try:
    tensor = dt.load("data.dt")
    returns = tensor.returns()
except RuntimeError as e:
    print(f"Error: {e}")
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
- **Unified API**: `from_numpy()` automatically handles all NumPy dtypes, simplifying code
