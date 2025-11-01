# Dragon Tensor

A high-performance tensor library written in C++17, designed specifically for financial data processing and quantitative analysis. Dragon Tensor provides seamless integration with NumPy, Pandas, and PyTorch ecosystems.

**Repository**: [https://github.com/zhongchun/dragon-tensor](https://github.com/zhongchun/dragon-tensor)

## Features

- **High Performance**: C++17 implementation optimized for financial computations
- **Financial Operations**: Built-in support for returns, rolling windows, correlation, and covariance
- **Multi-dimensional Tensors**: Support for 1D, 2D, and higher-dimensional tensors
- **Statistical Operations**: Mean, std, variance, min, max with optional axis operations
- **NumPy Integration**: Seamless conversion to/from NumPy arrays
- **Pandas Integration**: Direct conversion from/to Pandas Series and DataFrames
- **PyTorch Integration**: Convert to/from PyTorch tensors
- **Matrix Operations**: Matrix multiplication, transpose, and more

## Installation

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15+
- Python 3.7+ (for Python bindings)
- pybind11 (will be installed automatically if missing)
- NumPy (will be installed automatically if missing)
- Pandas (optional, for pandas integration)
- PyTorch (optional, for torch integration)

### Quick Build with Script

The easiest way to build Dragon Tensor is using the provided build script:

```bash
# Clone the repository
git clone https://github.com/zhongchun/dragon-tensor.git
cd dragon-tensor

# Standard release build (recommended)
# Builds C++ library AND generates Python wheel automatically
./build.sh

# Debug build (also generates wheel)
./build.sh --debug

# Build and install Python package
./build.sh --install

# Build only C++ library (no Python bindings, no wheel)
./build.sh --no-python

# Build without wheel generation
./build.sh --no-wheel

# Clean build artifacts (wheels, build dir, Python cache)
./build.sh --clean              # Cleans and exits
./build.sh --clean --install    # Cleans, then builds and installs

# Build with tests
./build.sh --with-tests

# See all options
./build.sh --help
```

### Building from Source

#### Using CMake directly:

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=$(which python3) \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

# Build
make -j$(nproc)

# Install Python package
cd ..
pip install .
```

#### CMake Build Options:

```bash
# Build the C++ library only
cmake .. -DBUILD_PYTHON_BINDINGS=OFF

# Build with Python bindings (default)
cmake .. -DBUILD_PYTHON_BINDINGS=ON

# Build with examples (default)
cmake .. -DBUILD_EXAMPLES=ON

# Build with tests
cmake .. -DBUILD_TESTS=ON

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release build with debug info
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## Quick Start

### Python Usage

**Note:** After building, install the Python package:
```bash
pip install .
# Or if you used build.sh with --install, it's already installed
```

```python
import numpy as np
import pandas as pd
import dragon_tensor as dt

# Create a tensor from numpy array
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
tensor = dt.from_numpy_double(arr)

# Basic operations
print(tensor.sum())      # 15.0
print(tensor.mean())    # 3.0
print(tensor.std())     # 1.414...

# Financial operations
returns = tensor.returns()
rolling_mean = tensor.rolling_mean(window=3)

# Create from pandas (if pandas is installed)
try:
    df = pd.DataFrame({'price': [100, 102, 101, 105, 108]})
    tensor_from_pd = dt.from_pandas_series(df['price'])
except ImportError:
    print("Pandas not installed")

# Convert back to numpy
result = tensor.to_numpy()
```

### C++ Usage

After building, the C++ headers are in `include/dragon_tensor/` and the library is in `build/libdragon_tensor.a`.

**Project Structure:**
- `include/dragon_tensor/tensor.h` - Header file with Tensor class declarations
- `src/tensor.cpp` - Implementation file with template instantiations
- `python/bindings.cpp` - Python bindings using pybind11
- `build/` - Build output directory

```cpp
#include <dragon_tensor/tensor.h>
#include <iostream>

using namespace dragon_tensor;

int main() {
    // Create a tensor
    TensorDouble prices({5}, {100.0, 102.0, 101.0, 105.0, 108.0});
    
    // Calculate returns
    auto returns = prices.returns();
    
    // Rolling statistics
    auto rolling_avg = prices.rolling_mean(3);
    auto rolling_std = prices.rolling_std(3);
    
    // Statistical operations
    std::cout << "Mean: " << prices.mean() << std::endl;
    std::cout << "Std: " << prices.std() << std::endl;
    
    // Correlation
    TensorDouble prices2({5}, {50.0, 51.0, 50.5, 52.5, 54.0});
    auto corr = prices.correlation(prices2);
    
    return 0;
}
```

Compile with:
```bash
g++ -std=c++17 -I./include your_program.cpp -L./build -ldragon_tensor -o your_program
```

Or run the example:
```bash
./build/examples/example_basic
```

## Financial Analysis Examples

### Returns Calculation

```python
import dragon_tensor as dt
import numpy as np

# Price series
prices = np.array([100, 102, 101, 105, 108, 110], dtype=np.float64)
tensor = dt.from_numpy_double(prices)

# Calculate returns
returns = tensor.returns()
print(returns.to_numpy())  # [0.02, -0.0098, 0.0396, 0.0286, 0.0185]
```

### Rolling Window Statistics

```python
# Rolling mean and standard deviation
rolling_mean = tensor.rolling_mean(3)
rolling_std = tensor.rolling_std(3)

print("Rolling Mean:", rolling_mean.to_numpy())
print("Rolling Std:", rolling_std.to_numpy())
```

### Correlation Analysis

```python
# Two asset price series
asset1 = dt.from_numpy_double(np.array([100, 102, 101, 105, 108], dtype=np.float64))
asset2 = dt.from_numpy_double(np.array([50, 51, 50.5, 52.5, 54], dtype=np.float64))

# Calculate correlation
corr = asset1.correlation(asset2)
print("Correlation:", corr.to_numpy()[0])

# Covariance
cov = asset1.covariance(asset2)
print("Covariance:", cov.to_numpy()[0])
```

### Integration with Pandas

```python
import pandas as pd
import dragon_tensor as dt

# Load financial data
df = pd.read_csv('prices.csv')
tensor = dt.from_pandas(df['close'])

# Perform calculations
returns = tensor.returns()
rolling_volatility = tensor.rolling_std(window=20)

# Convert back to pandas
returns_series = dt.to_pandas(returns, index=df.index[1:])
```

### Integration with PyTorch

```python
import torch
import dragon_tensor as dt

# Create PyTorch tensor
torch_tensor = torch.randn(100, dtype=torch.float64)

# Convert to Dragon Tensor
dt_tensor = dt.from_torch(torch_tensor)

# Perform calculations
result = dt_tensor.rolling_mean(window=10)

# Convert back to PyTorch
result_torch = dt.to_torch(result)
```

## API Reference

### Core Operations

- **Shape Operations**: `shape()`, `ndim()`, `size()`, `reshape()`, `flatten()`
- **Element Access**: `[]`, `at()`
- **Arithmetic**: `+`, `-`, `*`, `/` (element-wise and with scalars)
- **Mathematical**: `abs()`, `sqrt()`, `exp()`, `log()`, `pow()`

### Statistical Operations

- **Aggregation**: `sum()`, `mean()`, `std()`, `var()`, `max()`, `min()`
- **Axis Operations**: `sum(axis)`, `mean(axis)`, `std(axis)`, etc.

### Financial Operations

- `returns()`: Calculate percentage returns
- `rolling_mean(window)`: Rolling average
- `rolling_std(window)`: Rolling standard deviation
- `rolling_sum(window)`: Rolling sum
- `rolling_max(window)`: Rolling maximum
- `rolling_min(window)`: Rolling minimum
- `correlation(other)`: Correlation coefficient
- `covariance(other)`: Covariance

### Matrix Operations (2D only)

- `transpose()`: Matrix transpose
- `matmul(other)`: Matrix multiplication

## Project Structure

```
dragon-tensor/
├── include/
│   └── dragon_tensor/
│       └── tensor.h              # Tensor class header (declarations)
├── src/
│   └── tensor.cpp                 # Implementation with explicit template instantiations
├── python/
│   ├── bindings.cpp               # Python bindings (pybind11)
│   └── dragon_tensor/
│       ├── __init__.py            # Python package initialization
│       └── wrapper.py             # Convenience wrapper functions
├── examples/                      # Example code
├── CMakeLists.txt                 # CMake build configuration
├── build.sh                       # Build script
└── setup.py                       # Python package setup
```

**Implementation Notes:**
- The Tensor class is a template class with implementations in `src/tensor.cpp`
- Explicit template instantiations are provided for: `float`, `double`, `int32_t`, `int64_t`
- These correspond to `TensorFloat`, `TensorDouble`, `TensorInt`, and `TensorLong` type aliases

## Performance

Dragon Tensor is optimized for financial computations:

- **Memory Efficient**: Minimal overhead compared to raw arrays
- **Fast Operations**: Vectorized operations where possible
- **Zero-copy Conversions**: Efficient conversion between NumPy/Pandas/PyTorch formats

## Building

### Using the Build Script (Recommended)

The `build.sh` script automates the entire build process:

```bash
./build.sh                    # Standard release build (builds C++ + generates wheel)
./build.sh --debug            # Debug build (also generates wheel)
./build.sh --install          # Build and install Python package
./build.sh --no-wheel         # Build C++ without generating wheel
./build.sh --clean            # Clean build artifacts (exits after cleaning)
./build.sh --clean --install  # Clean then build and install
./build.sh --no-python        # Build only C++ library (no Python, no wheel)
./build.sh --with-tests       # Build with tests
./build.sh -j 8               # Use 8 parallel jobs
./build.sh --help             # Show all options
```

The build script automatically:
- Checks for required dependencies (CMake, Python, pybind11, NumPy)
- Installs missing Python packages if needed
- Configures CMake with correct paths
- Builds all components in parallel
- **Generates Python wheel automatically** (unless `--no-wheel` is specified)
- Reports build status

**Default Behavior:**
- By default, `./build.sh` builds the C++ library AND generates a Python wheel
- The wheel is created in `dist/` directory
- Use `--no-wheel` to disable wheel generation
- Use `--no-python` to build only C++ library (automatically disables wheel)

**Clean operation** (`--clean`):
- Removes build directory
- Removes `dist/` directory (Python wheels and source distributions)
- Cleans Python artifacts (`*.egg-info`, `__pycache__`, `.pyc`, `.pyo`)
- If used alone, exits after cleaning (no build)
- If combined with other options, cleans first then proceeds with build

### Using CMake Directly

For manual CMake builds:

```bash
# Basic build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# With specific Python executable
cmake .. \
    -DPython3_EXECUTABLE=$(which python3) \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
```

### Using Makefile

```bash
make                # Build C++ library and generate wheel (default, uses build.sh)
make build          # Build C++ library only (direct cmake, no wheel)
make build-wheel    # Build C++ library and generate wheel (uses build.sh)
make wheel          # Generate Python wheel only (requires build first)
make install        # Install Python package
make install-dev    # Install in development mode
make clean          # Clean build directory only
make clean-all      # Clean build directory, wheels, and Python artifacts
make examples       # Build and run C++ examples
make python-examples # Run Python examples
make help           # Show all available targets
```

**Note:** By default, `make` (or `make all`) uses `build.sh` which automatically generates wheels. Use `make build` for direct CMake builds without wheel generation.

## Running Examples

### C++ Examples

```bash
# Build and run the basic example
./build/examples/example_basic

# Or using make
make examples
```

### Python Examples

After building, you can test the Python bindings:

```bash
# Import and test (requires module in path)
python3 -c "import sys; sys.path.insert(0, './build'); \
import dragon_tensor as dt; import numpy as np; \
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64); \
tensor = dt.from_numpy_double(arr); \
print('Sum:', tensor.sum()); print('Mean:', tensor.mean())"

# Run example scripts (requires proper installation)
python3 examples/basic_usage.py
python3 examples/financial_analysis.py
python3 examples/integration_examples.py
```

## Troubleshooting

### pybind11 not found

If CMake cannot find pybind11:
```bash
pip install pybind11
```

The build script will automatically detect and use pybind11 from Python if installed via pip.

### NumPy not found

The build script will automatically install NumPy if missing. For manual installation:
```bash
pip install numpy
```

### Python module not found after build

The Python extension module is built in the `build/` directory. To use it without installation:

```bash
# Create a symlink for easier import (macOS/Linux)
cd build
ln -sf dragon_tensor*.dylib dragon_tensor.so  # macOS
# or
ln -sf dragon_tensor*.so dragon_tensor.so    # Linux
cd ..

# Then import with build directory in path
python3 -c "import sys; sys.path.insert(0, './build'); import dragon_tensor"
```

For permanent use, install the package:
```bash
pip install .
# or with build script
./build.sh --install
```

### Building Python Wheel

**Note:** The build script automatically generates Python wheels by default when Python bindings are enabled. No additional steps needed!

```bash
# Wheel is automatically generated when running:
./build.sh

# The wheel will be created in dist/ directory
# Example: dist/dragon_tensor-0.0.1-cp312-cp312-macosx_15_0_x86_64.whl
```

To build a wheel manually (if needed):

```bash
# Install build tools (if not already installed)
pip install build wheel

# Build wheel
python3 -m build --wheel

# Install from wheel
pip install dist/dragon_tensor-*.whl
```

**Build Options:**
- `./build.sh` - Builds C++ library and generates wheel automatically
- `./build.sh --no-wheel` - Builds C++ library without generating wheel
- `./build.sh --clean` - Removes dist/ directory with all wheels

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use Dragon Tensor in your research, please cite:

```bibtex
@software{dragon_tensor2024,
  title={Dragon Tensor: High-performance tensor library for financial data analysis},
  author={Dragon Tensor Contributors},
  year={2024},
  url={https://github.com/zhongchun/dragon-tensor}
}
```

