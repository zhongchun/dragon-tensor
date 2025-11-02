# Dragon Tensor Documentation

This directory contains comprehensive documentation for Dragon Tensor, organized by category.

## Directory Structure

```
docs/
├── design/          # Design documents and architecture specifications
├── api/             # API reference documentation
├── diagrams/        # Architecture diagrams (Mermaid, PlantUML)
└── performance/     # Performance guides and benchmarks
```

## Design Documents

Located in `design/`:

- **[requirements_doc_0.3.md](design/requirements_doc_0.3.md)** - Apache Arrow integration, refined 5-layer architecture with Backend Abstraction Layer, and enhanced allocator support
- **[requirements_doc_0.1.md](design/requirements_doc_0.1.md)** - Initial design and requirements specification
- **[requirements_doc_0.2.md](design/requirements_doc_0.2.md)** - Enhanced design with storage layer, buffer abstraction, and shared memory support
- **[uml_architecture.md](design/uml_architecture.md)** - UML-style architecture diagrams
- **[backend_abstraction.md](design/backend_abstraction.md)** - Backend abstraction layer documentation

## API Reference

Located in `api/`:

- **[C++ API Reference](api/cpp_api_reference.md)** - Complete C++ API documentation
- **[Python API Reference](api/python_api_reference.md)** - Complete Python API documentation

## Performance & Optimization

Located in `performance/`:

- **[Performance Optimizations](performance/optimizations.md)** - Comprehensive guide to performance optimizations organized by architecture layers

## Overview

### Design Documents

The requirements documents describe:
- **Architecture and layered design**: 5-layer architecture (Python API, Interop, Tensor Core, Buffer, Backend Abstraction)
- **Functional and non-functional requirements**: Core features, performance targets, and extensibility goals
- **API specifications**: Detailed interface definitions for both C++ and Python
- **Storage layer design**: File I/O, memory mapping, shared memory, Arrow/Parquet integration
- **Python API design**: High-level interfaces for financial analysis workflows
- **Data type system**: Support for multiple numeric types and Arrow types
- **Performance considerations**: Optimization strategies and expected performance characteristics
- **Apache Arrow integration**: Columnar analytics and Parquet file support
- **Allocator abstraction**: Flexible memory management strategies
- **Backend abstraction**: Unified interface for multiple storage backends

### API Reference

The API reference documents provide:
- **Detailed method signatures and parameters**: Complete function documentation
- **Return types and error conditions**: Type specifications and exception handling
- **Usage examples**: Code samples for common operations
- **Performance notes**: Best practices and performance considerations

### Performance & Optimization

The optimizations document outlines:
- **14 performance optimization opportunities**: Comprehensive list of potential improvements
- **Layer-specific optimizations**: Targeted optimizations for each architecture layer
  - Tensor Core optimizations
  - Interop Layer optimizations  
  - Buffer Layer optimizations
  - Backend Abstraction Layer optimizations
- **Priority rankings**: Implementation guidance based on impact and effort
- **Expected performance gains**: Quantified benefits for each optimization

## Development

- **[Development Guide](DEVELOPMENT.md)** - Guide for developers, including version management, build system details, and contributing guidelines

## Getting Started

For user-facing documentation, installation instructions, and quick start guides, see the main [README.md](../README.md).

## Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── DEVELOPMENT.md         # Development guide (version management, build system)
├── design/                # Design specifications and architecture
│   ├── README.md         # Design documents overview
│   ├── requirements_doc_0.3.md  # Latest requirements (v0.3)
│   ├── requirements_doc_0.2.md   # Requirements (v0.2)
│   ├── requirements_doc_0.1.md   # Requirements (v0.1)
│   ├── uml_architecture.md       # Architecture diagrams
│   └── backend_abstraction.md    # Backend layer design
├── api/                   # API reference documentation
│   ├── README.md         # API documentation overview
│   ├── cpp_api_reference.md      # C++ API reference
│   └── python_api_reference.md   # Python API reference
├── diagrams/              # Visual architecture diagrams (Mermaid, PlantUML)
└── performance/          # Performance guides and benchmarks
    └── optimizations.md   # Performance optimization guide
```

