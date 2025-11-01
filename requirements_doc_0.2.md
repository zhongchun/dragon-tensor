# Dragon-Tensor Design and Requirements Document (Financial Quantitative Analysis Edition, v2)

## 1. Overview

This document defines the design and requirements for **Dragon-Tensor**, a high-performance, extensible Tensor library implemented in C++. Dragon-Tensor is optimized for **financial data processing** and **quantitative analysis**, including portfolio optimization, time-series modeling, and machine learning. It provides a memory-efficient, high-precision foundation for analytics requiring low latency, zero-copy interoperability, and now supports **persistent and shared-memory storage** for high-performance data pipelines.

This version adds **robust file format versioning, deterministic memory management, shared memory lifecycle management, and enriched Python ergonomics** for real-world financial workflows.

---

## 2. Objectives

### Functional Requirements

1. **General Tensor Support**: Handle 1D (vectors), 2D (matrices), and N-dimensional tensors for financial data.
2. **Interoperability**: Enable zero-copy conversion with:

   * **NumPy ndarray** (for analytics and simulation workflows)
   * **PyTorch Tensor** (for model training and GPU acceleration)
3. **Data Persistence and Shared Access**:

   * Store tensors in **files** (row-wise or column-wise layout) with **versioning and metadata**.
   * Use **memory-mapped (mmap)** I/O for on-demand retrieval without loading the full dataset.
   * Store tensors in **shared memory**, supporting both row-wise and column-wise layouts, with **explicit lifecycle management**.
4. **Financial Data Orientation**:

   * Efficient for large dense numeric datasets (e.g., price matrices, volatility surfaces)
   * Time-series slicing for rolling-window analysis
5. **Precision and Stability**: Support `float64`, `float32`, and planned fixed-point types (`DECIMAL64/128`) for accurate computation.
6. **Device Abstraction**: CPU-first design with GPU extensibility for risk simulations.

### Non-Functional Requirements

* **Zero-Copy** data sharing
* **High Throughput** for time-critical analytics
* **Extensible** modular design
* **Deterministic Memory Control** via `Buffer` subclasses
* **Persistent and Shared Storage** with predictable access latency
* **Python/Numpy/Torch Compatibility**
* **Safe Shared-Memory Lifetime Management**

---

## 3. Architecture Overview

### 3.1 Layered Structure

| Layer                | Role                        | Description                                                                                                 |
| -------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Buffer Layer**     | Memory Management           | Manages CPU/GPU memory allocation, file-backed and shared-memory buffers with deterministic cleanup.        |
| **Tensor Core**      | Computation Abstraction     | Defines tensor metadata (shape, strides, dtype, layout). Supports slicing and reshaping.                    |
| **Interop Layer**    | Ecosystem Integration       | Provides zero-copy conversion with NumPy and PyTorch.                                                       |
| **Storage Layer**    | Persistence & Shared Access | Manages mmap and shared-memory backed tensors, includes versioning, checksum, and optional synchronization. |
| **Python API Layer** | User Interface              | Exposes a simple, intuitive API for analysts, including context management for file-backed tensors.         |

### 3.2 Data Model

Each `Tensor` object encapsulates:

* `std::shared_ptr<Buffer> buffer`: Underlying memory block (RAM, mmap, or shared memory)
* `std::vector<size_t> shape`: Dimensions (e.g., [days, assets])
* `std::vector<size_t> strides`: Layout strides (bytes per step)
* `DType dtype`: Data type
* `StorageMode storage_mode`: {`InMemory`, `MMap`, `SharedMemory`}
* `Layout layout`: {`RowMajor`, `ColumnMajor`}
* `TensorMeta meta`: Optional descriptive metadata (dim names, labels)
* `std::shared_ptr<void> owner`: Cross-language lifetime management
* Optional **atomic header for shared-memory refcounting and synchronization**

---

## 4. Simplified Python API Design

Dragon-Tensor provides a **minimal, clean Python API**, now extended for persistent and shared-memory tensors with context management and lifecycle methods.

### 4.1 Python API Overview

```python
import dragon_tensor as dt

# From NumPy or Torch
t1 = dt.from_numpy(np_data)
t2 = dt.from_torch(torch_tensor)

# Save tensor to disk (row-wise or column-wise)
t1.save("prices.dt", layout="row")

# Memory-map from file with context manager
with dt.open("prices.dt", mmap=True) as mapped_t:
    np_view = mapped_t.to_numpy()

# Create shared-memory tensor
t_shared = dt.create_shared("shared_prices", shape=(252, 1000), dtype="float64", layout="column")

# Attach to shared-memory tensor from another process
t2 = dt.attach_shared("shared_prices")

# Zero-copy conversion
torch_view = t_shared.to_torch()

# Explicit flush/detach
t_shared.flush()
t_shared.detach()
```

### 4.2 Python API Reference

| Function                                    | Description                                        | Zero-Copy   | Return         |
| ------------------------------------------- | -------------------------------------------------- | ----------- | -------------- |
| `from_numpy(array)`                         | Wrap NumPy ndarray                                 | ✅           | `Tensor`       |
| `from_torch(torch_tensor)`                  | Wrap PyTorch Tensor via DLPack                     | ✅           | `Tensor`       |
| `to_numpy()`                                | Convert Tensor to NumPy ndarray                    | ✅           | `np.ndarray`   |
| `to_torch()`                                | Convert Tensor to PyTorch tensor                   | ✅           | `torch.Tensor` |
| `save(path, layout="row")`                  | Save tensor to file in row- or column-major layout | ❌           | `None`         |
| `open(path, mmap=True)`                     | Context-managed load from file, optionally mmap    | ✅ (if mmap) | `Tensor`       |
| `create_shared(name, shape, dtype, layout)` | Create shared-memory tensor                        | ✅           | `Tensor`       |
| `attach_shared(name)`                       | Attach to existing shared-memory tensor            | ✅           | `Tensor`       |
| `flush()`                                   | Force write-back for file-backed tensors           | ❌           | `None`         |
| `detach()`                                  | Unmap shared-memory tensor                         | ❌           | `None`         |

---

## 5. Storage Layer Design

### 5.1 File-Based Storage

* **Row-wise layout**: Fast sequential access for time-series data
* **Column-wise layout**: Optimized for per-asset queries
* **Binary format with robust header**: includes **magic, version, endian, checksum**
* **mmap support**: On-demand access without full load

```cpp
struct TensorHeader {
    uint32_t magic = 0x44544E53; // 'DTNS'
    uint32_t version = 1;
    uint32_t ndim;
    uint32_t dtype;
    uint32_t layout;  // 0=row, 1=col
    uint32_t endian;   // 0=little, 1=big
    uint64_t shape[N];
    uint64_t data_offset;
    uint64_t checksum; // optional CRC64
};
```

### 5.2 Shared Memory Storage

* Uses POSIX or System V shared memory for cross-process access
* Supports **row-major** and **column-major** layouts
* Optional synchronization using **atomic header or named semaphores**
* Lifecycle management: `detach()`, `destroy_shared()`

```cpp
Tensor Tensor::create_shared(const std::string& name, Shape shape, DType dtype, Layout layout);
Tensor Tensor::attach_shared(const std::string& name);
void Tensor::detach();
static void Tensor::destroy_shared(const std::string& name);
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
       StorageMode mode = StorageMode::InMemory,
       TensorMeta meta = {});
```

### 6.2 Storage APIs

```cpp
void save(const std::string& path, Layout layout = Layout::RowMajor) const;
static Tensor load(const std::string& path, bool mmap = true);
static Tensor create_shared(const std::string& name, Shape shape, DType dtype, Layout layout);
static Tensor attach_shared(const std::string& name);
void flush();  // force write-back for file-backed or mmap tensors
void detach(); // unmap shared memory
static void destroy_shared(const std::string& name);
```

---

## 7. Data Type System

```cpp
enum class DType { FLOAT32, FLOAT64, INT32, INT64, UINT8, DECIMAL64, DECIMAL128 };
```

* Future: precise fixed-point arithmetic for currency operations

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
with dt.open("prices.dt", mmap=True) as mapped_t:
    np_view = mapped_t.to_numpy()

# Shared memory tensor for multi-process use
t_shared = dt.create_shared("risk_shared", shape=(252, 500), dtype="float32", layout="row")
t_shared.flush()
t_shared.detach()
```

---

## 9. Financial Analysis Extensions

* **Rolling Window Views** for backtesting
* **Covariance/Correlation Matrices** for portfolio risk models
* **Batch Factor Computation** for cross-asset analysis
* **MMap-based Time-Series Query** for high-frequency analysis
* **Shared Memory Cache** for intra-day model updates across processes
* **Label-Aware Slicing** with `TensorMeta` dimension names and labels

---

## 10. Performance Highlights

| Optimization                       | Benefit                                 |
| ---------------------------------- | --------------------------------------- |
| Zero-Copy Interop                  | Eliminates data transfer overhead       |
| MMap-based I/O                     | On-demand retrieval from large datasets |
| Shared Memory                      | Ultra-low latency inter-process access  |
| Row/Column Layout Control          | Query-optimized storage patterns        |
| Reference Counting & Atomic Header | Safe cross-boundary memory reuse        |
| View-based Slicing                 | O(1) window slicing for time-series     |

---

## 11. Future Enhancements

1. **GPU/Device Support** for CUDA/HIP
2. **Parallelism** (OpenMP or thread pool)
3. **Fixed-Point Arithmetic** for currency-safe precision
4. **Arrow / Parquet Interop** for columnar analytics
5. **Distributed Shared Memory (RDMA)** for cluster-scale analytics
6. **Schema Versioning, Compression, and Checksum Validation**
7. **Async I/O Backends** for high-throughput ingestion

---

## 12. Summary

**Dragon-Tensor v2** now provides an **end-to-end analytical tensor infrastructure** for finance with:

* Robust, versioned persistent storage
* Deterministic memory and shared-memory management
* Zero-copy integration with NumPy and PyTorch
* Enhanced Python ergonomics (`open()`, context management, `flush()`, `detach()`)
* Rich metadata support for labeled financial datasets
* Layered design ready for GPU, distributed, and streaming extensions

This architecture enables **low-latency quantitative research**, **risk computation**, and **real-time financial analytics** with unified memory semantics across Python and C++.
