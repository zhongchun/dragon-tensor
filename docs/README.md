# Dragon Tensor Documentation

This directory contains comprehensive documentation for Dragon Tensor, including design specifications and API references.

## Documents

### Design & Requirements
- **[Requirements Document v0.1](requirements_doc_0.1.md)** - Initial design and requirements specification
- **[Requirements Document v0.2](requirements_doc_0.2.md)** - Enhanced design with storage layer, buffer abstraction, and shared memory support
- **[Requirements Document v0.3](requirements_doc_0.3.md)** - Apache Arrow integration, refined 5-layer architecture with Backend Abstraction Layer, and enhanced allocator support

### API Reference
- **[C++ API Reference](api_cpp.md)** - Complete C++ API documentation
- **[Python API Reference](api_python.md)** - Complete Python API documentation

### Performance & Optimization
- **[Performance Optimizations](optimizations.md)** - Comprehensive guide to performance optimizations organized by architecture layers

## Overview

The requirements documents describe:
- Architecture and layered design (5-layer architecture: Python API, Interop, Tensor Core, Buffer, Backend Abstraction)
- Functional and non-functional requirements
- API specifications
- Storage layer design (file I/O, memory mapping, shared memory, Arrow/Parquet)
- Python API design
- Data type system
- Performance considerations
- Apache Arrow integration for columnar analytics
- Allocator abstraction and memory management strategies
- Backend abstraction for flexible storage backends

The API reference documents provide:
- Detailed method signatures and parameters
- Return types and error conditions
- Usage examples
- Performance notes and best practices

The optimizations document outlines:
- 14 performance optimization opportunities
- Layer-specific optimizations (Tensor Core, Interop Layer, Buffer Layer, Backend Layer)
- Priority rankings and implementation guidance
- Expected performance gains for each optimization

For user-facing documentation and quick start guides, see the main [README.md](../README.md).

