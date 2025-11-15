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

### Optimized Build Architecture

Dragon Tensor uses an **optimized two-stage build process**:

1. **C++ Compilation Stage** (CMake):
   - Compiles all C++ source files
   - Builds the static library (`libdragon_tensor.a`)
   - Generates Python extension module (`.so`/`.dylib`)
   - Uses optimized build tools (Ninja, ccache)

2. **Python Packaging Stage** (`setup.py`):
   - Uses pre-built extension module from CMake
   - No C++ compilation in Python build
   - Faster builds with better caching

This separation provides:
- **Faster rebuilds**: ccache caches C++ compilation results
- **Better parallelization**: CMake handles C++ with optimal job distribution
- **Consistent builds**: Same C++ build process for all targets
- **Cleaner separation**: C++ compilation vs Python packaging

### CMake Configuration

The CMake build system:
- Reads `VERSION.txt` and generates `version.h`
- Compiles all source files from `src/` directory
- Links backend implementations
- Generates Python bindings via pybind11
- **Uses Ninja generator** (if available) for faster builds
- **Applies Release optimizations** (`-O3 -DNDEBUG`) for production builds
- **Auto-detects CPU cores** for parallel compilation

### Build Optimizations

The build system includes several optimizations:

#### ccache (Compiler Cache)
- Automatically enabled if `ccache` is installed
- Caches compilation results for faster rebuilds
- Shows cache statistics after build
- Install: `brew install ccache` (macOS) or `apt-get install ccache` (Linux)

#### Ninja Build System
- Automatically used if `ninja` is available
- Faster than traditional Make
- Better parallel build performance
- Install: `brew install ninja` (macOS) or `apt-get install ninja-build` (Linux)

#### Parallel Builds
- Auto-detects number of CPU cores
- Uses all available cores by default
- Can be overridden with `-j N` flag: `./build.sh -j 4`

#### Release Optimizations
- Automatically applies `-O3 -DNDEBUG` for Release builds
- Optimized for production performance
- Debug builds use `-g -O0` for debugging

### Python Package

The Python package:
- Uses `setup.py` which reads from `VERSION.txt`
- **Uses pre-built extension module** from CMake build directory
- No C++ compilation in Python build (faster)
- Organizes code into submodules: `io`, `finance`, `shared`, `utils`
- Finds and copies the pre-built extension module automatically

**Build Process:**
1. Run `./build.sh` to build C++ with CMake
2. Run `pip install .` or `python -m build --wheel` to create Python package
3. `setup.py` automatically finds and copies the pre-built extension

**Environment Variables:**
- `CMAKE_BUILD_DIR`: Override build directory (default: `build`)
- `USE_CCACHE`: Enable/disable ccache (default: `1`)
- `USE_NINJA`: Enable/disable Ninja (default: `1`)
- `MAX_JOBS`: Override number of parallel jobs (default: auto-detect)

## Project Structure

See the main [README.md](../README.md) for the complete project structure.

## Code Style

- **C++**: Google C++ Style Guide (enforced via `.clang-format`)
- **Python**: Black formatter

Run `./format.sh` to format all code.

## Testing

Run `./scripts/test_build.sh` to verify:
- C++ library builds correctly
- Python module imports successfully
- Basic operations work
- Financial operations work

## Contributing

Before submitting changes:
1. Run `./format.sh` to format code
2. Run `./scripts/test_build.sh` to verify build
3. Update documentation if needed
4. Ensure version in `VERSION.txt` is appropriate

