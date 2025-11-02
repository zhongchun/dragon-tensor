# Dragon-Tensor Design and Requirements Document (Financial Quantitative Analysis Edition)

## 1. Overview
This document defines the design and requirements for **Dragon-Tensor**, a high-performance, extensible Tensor library implemented in C++. Dragon-Tensor is optimized for **financial data processing** and **quantitative analysis**, including portfolio optimization, time-series modeling, and machine learning. It provides a memory-efficient, high-precision foundation for analytics requiring low latency, zero-copy interoperability, and now supports **persistent and shared-memory storage** for high-performance data pipelines.

---

## 2. Objectives

### Functional Requirements

1. **General Tensor Support**: Handle 1D (vectors), 2D (matrices), and N-dimensional tensors for financial data.
2. **Interoperability**: Enable zero-copy conversion with:
   - **NumPy ndarray** (for analytics and simulation workflows)
   - **PyTorch Tensor** (for model training and GPU acceleration)
3. **Data Persistence and Shared Access**:
   - Store tensors in **files** (row-wise or column-wise layout) to optimize for query patterns.
   - Use **memory-mapped (mmap)** I/O for on-demand retrieval without loading the full dataset.
   - Store tensors in **shared memory**, also supporting both row-wise and column-wise layouts, to enable low-latency multi-process access.
4. **Financial Data Orientation**:
   - Efficient for large dense numeric datasets (e.g., price matrices, volatility surfaces)
   - Time-series slicing for rolling-window analysis
5. **Precision and Stability**: Support `float64` and `float32` for accurate computation.
6. **Device Abstraction**: CPU-first design with GPU extensibility for risk simulations.

### Non-Functional Requirements

- **Zero-Copy** data sharing
- **High Throughput** for time-critical analytics
- **Extensible** modular design
- **Deterministic Memory Control**
- **Persistent and Shared Storage** with predictable access latency
- **Python/Numpy/Torch Compatibility**

---

## 3. Architecture Overview

### 3.1 Layered Structure

| Layer | Role | Description |
|--------|------|-------------|
| **Buffer Layer** | Memory Management | Manages CPU/GPU memory allocation, file-backed and shared-memory buffers. |
| **Tensor Core** | Computation Abstraction | Defines tensor metadata (shape, strides, dtype). Supports slicing and reshaping. |
| **Interop Layer** | Ecosystem Integration | Provides zero-copy conversion with NumPy and PyTorch. |
| **Storage Layer** | Persistence & Shared Access | Manages mmap and shared-memory backed tensors for high-speed analytics. |
| **Python API Layer** | User Interface | Exposes a simple, intuitive API for analysts. |

### 3.2 Data Model

Each `Tensor` object encapsulates:
- `Buffer* buffer`: Underlying memory block (RAM, mmap, or shared memory).
- `std::vector<size_t> shape`: Dimensions (e.g., [days, assets]).
- `std::vector<size_t> strides`: Layout (bytes per step).
- `DType dtype`: Data type.
- `StorageMode storage_mode`: {`InMemory`, `MMap`, `SharedMemory`}.
- `Layout layout`: {`RowMajor`, `ColumnMajor`}.
- `std::shared_ptr<void> owner`: Lifetime management for cross-language safety.

---

## 4. Simplified Python API Design

Dragon-Tensor provides a **minimal, clean Python API**, now extended to handle persistent and shared-memory tensors.

### 4.1 Python API Overview
```python
import dragon_tensor as dt

# From NumPy or Torch
t1 = dt.from_numpy(np_data)
t2 = dt.from_torch(torch_tensor)

# Save tensor to disk (row-wise or column-wise)
t1.save("prices.dt", layout="row")

# Memory-map from file
mapped_t = dt.load("prices.dt", mmap=True)

# Create shared-memory tensor
t_shared = dt.create_shared("shared_prices", shape=(252, 1000), dtype="float64", layout="column")

# Attach to shared-memory tensor from another process
t2 = dt.attach_shared("shared_prices")

# Zero-copy conversion
np_view = mapped_t.to_numpy()
torch_view = t_shared.to_torch()
```

### 4.2 Python API Reference

| Function | Description | Zero-Copy | Return |
|-----------|--------------|------------|---------|
| `from_numpy(array)` | Wrap NumPy ndarray | ✅ | `Tensor` |
| `from_torch(torch_tensor)` | Wrap PyTorch Tensor via DLPack | ✅ | `Tensor` |
| `to_numpy()` | Convert Tensor to NumPy ndarray | ✅ | `np.ndarray` |
| `to_torch()` | Convert Tensor to PyTorch tensor | ✅ | `torch.Tensor` |
| `save(path, layout="row")` | Save tensor to file in row- or column-major layout | ❌ | `None` |
| `load(path, mmap=True)` | Load tensor from file, optionally using mmap | ✅ (if mmap) | `Tensor` |
| `create_shared(name, shape, dtype, layout)` | Create shared-memory tensor | ✅ | `Tensor` |
| `attach_shared(name)` | Attach to existing shared-memory tensor | ✅ | `Tensor` |

---

## 5. Storage Layer Design

### 5.1 File-Based Storage

- **Row-wise layout**: Fast sequential access for time-series data.
- **Column-wise layout**: Optimized for per-asset or per-factor queries.
- **Binary format**: Lightweight header (metadata + shape + dtype) followed by contiguous tensor data.
- **mmap support**: Memory-map large files for partial, on-demand access without full load.

Header structure:
```cpp
struct TensorHeader {
    uint32_t ndim;
    uint32_t dtype;
    uint32_t layout; // 0=row, 1=col
    uint64_t shape[N]; // variable length
};
```

### 5.2 Shared Memory Storage

- Uses POSIX shared memory (`shm_open`, `mmap`) or System V (`shmget`) for cross-process access.
- Shared tensors support both **row-major** and **column-major** layouts.
- Synchronization (optional) via file locks or atomic counters.
- Lifetime managed through reference counting or explicit `detach()`.

```cpp
Tensor Tensor::create_shared(const std::string& name, Shape shape, DType dtype, Layout layout);
Tensor Tensor::attach_shared(const std::string& name);
```

---

## 6. Core API (C++)

### 6.1 Construction

```cpp
Tensor(std::shared_ptr<Buffer> buffer,
       std::vector<size_t> shape,
       DType dtype,
       std::vector<size_t> strides = {},
       Layout layout = Layout::RowMajor,
       StorageMode mode = StorageMode::InMemory);
```

### 6.2 Storage APIs

```cpp
void save(const std::string& path, Layout layout = Layout::RowMajor) const;
static Tensor load(const std::string& path, bool mmap = true);
static Tensor create_shared(const std::string& name, Shape shape, DType dtype, Layout layout);
static Tensor attach_shared(const std::string& name);
```

---

## 7. Data Type System

Supported types for finance and analytics:
```cpp
enum class DType { FLOAT32, FLOAT64, INT32, INT64, UINT8 };
```
Future: `DECIMAL128`, fixed-point arithmetic for precise currency operations.

---

## 8. Example Usage

```python
import dragon_tensor as dt
import numpy as np

# Create tensor and persist
prices = np.random.randn(252, 1000).astype(np.float64)
t = dt.from_numpy(prices)
t.save("prices.dt", layout="column")

# Memory-map file for on-demand queries
mapped_t = dt.load("prices.dt", mmap=True)

# Shared memory tensor for multi-process use
t_shared = dt.create_shared("risk_shared", shape=(252, 500), dtype="float32", layout="row")
```

---

## 9. Financial Analysis Extensions

- **Rolling Window Views** for backtesting
- **Covariance/Correlation Matrices** for portfolio risk models
- **Batch Factor Computation** for cross-asset analysis
- **MMap-based Time-Series Query** for high-frequency analysis
- **Shared Memory Cache** for intra-day model updates across processes

---

## 10. Performance Highlights

| Optimization | Benefit |
|--------------|----------|
| Zero-Copy Interop | Eliminates data transfer overhead |
| MMap-based I/O | On-demand retrieval from large datasets |
| Shared Memory | Ultra-low latency inter-process access |
| Row/Column Layout Control | Query-optimized storage patterns |
| Reference Counting | Safe cross-boundary memory reuse |
| View-based Slicing | O(1) window slicing for time-series |

---

## 11. Future Enhancements

1. **GPU/Device Support** for CUDA/HIP.
2. **Parallelism** (OpenMP or thread pool).
3. **Fixed-Point Arithmetic** for currency-safe precision.
4. **Arrow Interop** for columnar analytics.
5. **Distributed Shared Memory (RDMA)** for cluster-scale analytics.
6. **Schema Versioning and Compression** for persistent storage.

---

## 12. Summary

**Dragon-Tensor** now provides an **end-to-end analytical tensor infrastructure** for finance:
- High-performance in-memory operations.
- Zero-copy integration with NumPy and PyTorch.
- Persistent storage via file and mmap.
- Shared-memory tensors for real-time multi-process access.
- Layered, extensible design ready for GPU, distributed, and streaming extensions.

This architecture enables **low-latency quantitative research**, **risk computation**, and **real-time financial analytics** with unified memory semantics across Python and C++.
