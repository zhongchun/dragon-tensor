# Development Guide

This document provides information for developers working on Dragon Tensor.

## Version Management

Dragon Tensor uses **`VERSION.txt`** as the single source of truth for version information.

### How It Works

1. **`VERSION.txt`**: Contains the version string (e.g., `0.0.1`)

2. **C++ Build Process**:
   - CMake reads `VERSION.txt` during configuration
   - Generates `build/include/dragon_tensor/version.h` from `include/dragon_tensor/version.h.in` template
   - C++ code includes the generated `version.h` header

3. **Python Build Process**:
   - `setup.py` reads `VERSION.txt` via `get_version()` function
   - Python package metadata uses this version
   - `python/dragon_tensor/__init__.py` reads version from package metadata or falls back to `VERSION.txt`

### Updating the Version

To update the version, simply edit `VERSION.txt`:

```bash
# Change version from 0.0.1 to 0.0.2
echo "0.0.2" > VERSION.txt
```

All build systems will automatically use the new version:
- CMake will generate updated `version.h` on next build
- Python packages will use the new version on next build
- No manual updates needed in other files

## Build System

### CMake Configuration

The CMake build system:
- Reads `VERSION.txt` and generates `version.h`
- Compiles all source files from `src/` directory
- Links backend implementations
- Generates Python bindings via pybind11

### Python Package

The Python package:
- Uses `setup.py` which reads from `VERSION.txt`
- Compiles C++ extension from source (platform-independent)
- Organizes code into submodules: `io`, `finance`, `shared`, `utils`

## Project Structure

See the main [README.md](../README.md) for the complete project structure.

## Code Style

- **C++**: Google C++ Style Guide (enforced via `.clang-format`)
- **Python**: Black formatter

Run `./format.sh` to format all code.

## Testing

Run `./test_build.sh` to verify:
- C++ library builds correctly
- Python module imports successfully
- Basic operations work
- Financial operations work

## Contributing

Before submitting changes:
1. Run `./format.sh` to format code
2. Run `./test_build.sh` to verify build
3. Update documentation if needed
4. Ensure version in `VERSION.txt` is appropriate

