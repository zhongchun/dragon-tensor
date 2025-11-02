# Design Documents

This directory contains design specifications and architecture documentation for Dragon Tensor.

## Documents

### Requirements Documents

- **[requirements_doc_0.3.md](./requirements_doc_0.3.md)** (Latest) - Apache Arrow integration, refined 5-layer architecture with Backend Abstraction Layer, and enhanced allocator support
- **[requirements_doc_0.2.md](./requirements_doc_0.2.md)** - Enhanced design with storage layer, buffer abstraction, and shared memory support
- **[requirements_doc_0.1.md](./requirements_doc_0.1.md)** - Initial design and requirements specification

### Architecture Documentation

- **[uml_architecture.md](./uml_architecture.md)** - UML-style architecture diagrams showing the 5-layer structure
- **[backend_abstraction.md](./backend_abstraction.md)** - Backend abstraction layer design and implementation details

## Architecture Overview

Dragon Tensor follows a **5-layer architecture**:

1. **Python API Layer** - High-level Python interface with pybind11
2. **Interop Layer** - Zero-copy integration with NumPy, PyTorch, and Apache Arrow
3. **Tensor Core** - Core tensor operations, shapes, types, and computations
4. **Buffer Layer** - Memory management with allocator abstraction
5. **Backend Abstraction Layer** - Unified storage backends (memory, mmap, shared memory, Arrow/Parquet)

## Version History

- **v0.3** (Current) - Arrow integration, backend abstraction, allocator support, centralized version management
- **v0.2** - Storage layer, buffer abstraction, shared memory
- **v0.1** - Initial design with basic tensor operations

## Version Management

The project uses **`VERSION.txt`** as the single source of truth for version information. Both C++ and Python components read from this file:

- **C++**: CMake generates `version.h` from `VERSION.txt` during build
- **Python**: `setup.py` and `__init__.py` read directly from `VERSION.txt`
- **Build System**: CMake project version is set from `VERSION.txt`

This ensures version consistency across all components.

## Related Documentation

- [API Reference](../api/README.md) - C++ and Python API documentation
- [Performance Guide](../performance/optimizations.md) - Optimization strategies
- [Main README](../README.md) - Documentation index

