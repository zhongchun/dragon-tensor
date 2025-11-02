# Dragon Tensor UML Architecture

This document contains the UML-style architecture diagrams for Dragon Tensor v3.

## Layer Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Python API Layer                                   │
│                                                                              │
│  Python API (Pybind11)                                                       │
│  ───────────────────────                                                    │
│  + from_numpy(np.ndarray) -> Tensor                                          │
│  + from_torch(torch.Tensor) -> Tensor                                        │
│  + from_arrow(arrow.Array) -> Tensor                                         │
│  + from_file(path, layout) -> Tensor                                         │
│  + from_shared(name, layout) -> Tensor                                       │
│  + Tensor operations: reshape(), view(), slice(), transpose()                │
│  + Mathematical ops: sum(), mean(), std(), abs(), sqrt()                     │
│  + Financial ops: rolling_mean(), returns(), correlation(), volatility()     │
│  + Storage ops: save(), load(), flush(), detach()                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ provides high-level API
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Interop Layer                                      │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  NumPyInterop   │  │   TorchInterop  │  │   ArrowInterop  │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                              │
│  Responsibilities:                                                           │
│    - Zero-copy via DLPack, buffer protocol, Arrow memory                    │
│    - Type/stride compatibility checking                                     │
│    - Device/context translation                                             │
│    - Schema and metadata preservation                                       │
│    - View semantics preservation                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ converts formats
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Tensor Core                                        │
│                                                                              │
│  template<typename T, size_t N>                                              │
│  class Tensor                                                                │
│  ───────────────────────                                                    │
│  Attributes:                                                                 │
│    - Shape shape_              // Multi-dimensional shape                    │
│    - DType dtype_              // Data type (FLOAT32, FLOAT64, etc.)       │
│    - Stride strides_           // Memory stride per dimension               │
│    - std::shared_ptr<Buffer> buffer_  // Memory buffer                      │
│    - Layout layout_            // ROW_MAJOR | COLUMN_MAJOR                 │
│    - TensorMeta meta_          // Optional metadata (labels, schema)        │
│                                                                              │
│  Operations:                                                                 │
│    + reshape(new_shape) -> Tensor                                            │
│    + view(dtype) -> Tensor                                                   │
│    + slice(range) -> Tensor                                                  │
│    + transpose() -> Tensor                                                   │
│    + layout() -> Layout                                                      │
│    + backend() -> Backend*                                                   │
│    + mathematical operations (+, -, *, /, abs, sqrt, etc.)                  │
│    + statistical operations (sum, mean, std, var, etc.)                      │
│    + financial operations (returns, rolling, correlation, etc.)             │
│                                                                              │
│  Design:                                                                     │
│    * Template-based with constexpr shapes for compile-time optimization     │
│    * Supports row-major & column-major layouts                              │
│    * View-based slicing (O(1) operations)                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ manages memory
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Buffer Layer                                       │
│                                                                              │
│  class Buffer                                                                 │
│  ───────────────────────                                                    │
│  Attributes:                                                                 │
│    - void* data_                // Raw memory pointer                       │
│    - size_t size_               // Buffer size in bytes                      │
│    - Layout layout_             // Memory layout                            │
│    - std::shared_ptr<Allocator> allocator_  // Allocation strategy          │
│    - std::shared_ptr<Backend> backend_     // Storage backend               │
│                                                                              │
│  Operations:                                                                 │
│    + data() -> void*                                                        │
│    + size() -> size_t                                                       │
│    + layout() -> Layout                                                     │
│    + allocate(size_t, Layout) -> void                                       │
│    + deallocate() -> void                                                   │
│    + map() / unmap()        // For memory-mapped files                      │
│    + attach() / detach()    // For shared memory                            │
│                                                                              │
│  Allocator Types:                                                            │
│    * HeapAllocator          // Standard heap allocation                    │
│    * PoolAllocator          // Pool-based for small tensors                 │
│    * AlignedAllocator       // SIMD-aligned allocation                     │
│                                                                              │
│  Design:                                                                     │
│    * Handles memory ownership, alignment, slicing                          │
│    * RAII-based lifetime management                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ uses storage backends
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Backend Abstraction Layer (Storage Layer)                 │
│                                                                              │
│  interface Backend                                                            │
│  ───────────────────────                                                    │
│  Operations:                                                                 │
│    + allocate(size_t, Layout) -> Buffer                                     │
│    + release(Buffer&) -> void                                                │
│    + name() -> std::string                                                   │
│    + flush() -> void                                                         │
│    + supports_mmap() -> bool                                                 │
│                                                                              │
│  Implementations:                                                             │
│    ┌──────────────────┐                                                     │
│    │ MemoryBackend    │  // Heap-allocated memory                           │
│    └──────────────────┘                                                     │
│    ┌──────────────────┐                                                     │
│    │ MMapBackend      │  // Memory-mapped file I/O                          │
│    └──────────────────┘                                                     │
│    ┌──────────────────┐                                                     │
│    │ SharedMemoryBackend│  // POSIX/System V shared memory                  │
│    └──────────────────┘                                                     │
│    ┌──────────────────┐                                                     │
│    │ ArrowBackend     │  // Apache Arrow/Parquet storage                   │
│    └──────────────────┘                                                     │
│                                                                              │
│  Design:                                                                     │
│    * Configurable layout: ROW_MAJOR / COLUMN_MAJOR                         │
│    * Backend chosen via Tensor creation factory methods                     │
│    * Unified interface regardless of underlying storage mechanism           │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Mermaid Diagram

See the full Mermaid diagram in `DragonTensor_Design_v3.md`.

