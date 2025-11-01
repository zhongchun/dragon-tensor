#!/bin/bash

# Dragon Tensor Build Script
# This script simplifies building the Dragon Tensor library

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
BUILD_PYTHON_BINDINGS="ON"
BUILD_TESTS="OFF"
BUILD_EXAMPLES="ON"
INSTALL="OFF"
PYTHON_EXECUTABLE=""

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Dragon Tensor library

OPTIONS:
    -h, --help              Show this help message
    -d, --debug             Build in Debug mode (default: Release)
    -t, --build-type TYPE   Build type: Debug, Release, RelWithDebInfo, MinSizeRel
    -b, --build-dir DIR     Build directory (default: build)
    --no-python             Disable Python bindings
    --with-tests            Enable building tests
    --no-examples           Disable building examples
    -i, --install           Install after building
    -p, --python PATH       Python executable path
    --clean                 Clean build directory before building
    -j, --jobs N           Number of parallel jobs (default: auto)

EXAMPLES:
    $0                      # Standard release build
    $0 --debug              # Debug build
    $0 --clean --install    # Clean build and install
    $0 --no-python          # Build only C++ library
    $0 -j 4                 # Build with 4 parallel jobs

EOF
}

# Parse command line arguments
CLEAN_BUILD=false
JOBS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -t|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --no-python)
            BUILD_PYTHON_BINDINGS="OFF"
            shift
            ;;
        --with-tests)
            BUILD_TESTS="ON"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        -i|--install)
            INSTALL="ON"
            shift
            ;;
        -p|--python)
            PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check dependencies
print_info "Checking dependencies..."

# Check for CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake is required but not installed."
    print_info "Install CMake: https://cmake.org/download/"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
print_info "Found CMake version: $CMAKE_VERSION"

# Check for C++ compiler
if ! command -v c++ &> /dev/null && ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    print_error "C++ compiler is required but not found."
    exit 1
fi

# Find Python if not specified
if [ -z "$PYTHON_EXECUTABLE" ] && [ "$BUILD_PYTHON_BINDINGS" = "ON" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_EXECUTABLE=$(which python3)
    elif command -v python &> /dev/null; then
        PYTHON_EXECUTABLE=$(which python)
    else
        print_error "Python is required for building Python bindings but not found."
        exit 1
    fi
    print_info "Using Python: $PYTHON_EXECUTABLE"
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_EXECUTABLE --version 2>&1 | cut -d' ' -f2)
    print_info "Python version: $PYTHON_VERSION"
    
    # Check for pybind11
    if ! $PYTHON_EXECUTABLE -c "import pybind11" 2>/dev/null; then
        print_warning "pybind11 not found. Installing..."
        $PYTHON_EXECUTABLE -m pip install pybind11 --quiet
    fi
    
    # Check for numpy
    if ! $PYTHON_EXECUTABLE -c "import numpy" 2>/dev/null; then
        print_warning "NumPy not found. Installing..."
        $PYTHON_EXECUTABLE -m pip install numpy --quiet
    fi
fi

# Get pybind11 directory if Python bindings are enabled
PYBIND11_DIR=""
if [ "$BUILD_PYTHON_BINDINGS" = "ON" ] && [ -n "$PYTHON_EXECUTABLE" ]; then
    PYBIND11_DIR=$($PYTHON_EXECUTABLE -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
    if [ -z "$PYBIND11_DIR" ]; then
        print_error "Could not find pybind11 CMake directory"
        exit 1
    fi
    print_info "Found pybind11 at: $PYBIND11_DIR"
fi

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_info "Cleaning build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake
print_info "Configuring CMake..."
CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DBUILD_PYTHON_BINDINGS=$BUILD_PYTHON_BINDINGS"
    "-DBUILD_TESTS=$BUILD_TESTS"
    "-DBUILD_EXAMPLES=$BUILD_EXAMPLES"
)

if [ -n "$PYTHON_EXECUTABLE" ]; then
    CMAKE_ARGS+=("-DPython3_EXECUTABLE=$PYTHON_EXECUTABLE")
fi

if [ -n "$PYBIND11_DIR" ]; then
    CMAKE_ARGS+=("-Dpybind11_DIR=$PYBIND11_DIR")
fi

CMAKE_ARGS+=("..")

print_info "CMake command: cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"

# Build
print_info "Building..."
BUILD_ARGS=()
if [ -n "$JOBS" ]; then
    BUILD_ARGS+=("-j$JOBS")
else
    # Auto-detect number of cores
    if command -v nproc &> /dev/null; then
        CORES=$(nproc)
    elif command -v sysctl &> /dev/null; then
        CORES=$(sysctl -n hw.ncpu)
    else
        CORES=4
    fi
    BUILD_ARGS+=("-j$CORES")
fi

cmake --build . "${BUILD_ARGS[@]}"

print_info "Build completed successfully!"

# Show build results
echo ""
print_info "Build results:"
if [ -f "libdragon_tensor.a" ]; then
    echo "  ✓ C++ library: libdragon_tensor.a"
fi

if [ "$BUILD_PYTHON_BINDINGS" = "ON" ]; then
    PYTHON_MODULE=$(find . -name "dragon_tensor*.so" -o -name "dragon_tensor*.dylib" | head -1)
    if [ -n "$PYTHON_MODULE" ]; then
        echo "  ✓ Python module: $(basename $PYTHON_MODULE)"
    fi
fi

if [ "$BUILD_EXAMPLES" = "ON" ]; then
    if [ -f "examples/example_basic" ]; then
        echo "  ✓ Example executable: examples/example_basic"
    fi
fi

# Install if requested
if [ "$INSTALL" = "ON" ]; then
    echo ""
    print_info "Installing..."
    if [ "$BUILD_PYTHON_BINDINGS" = "ON" ]; then
        cd ..
        $PYTHON_EXECUTABLE -m pip install . --quiet
        print_info "Python package installed successfully!"
    else
        cmake --install .
        print_info "Library installed successfully!"
    fi
fi

echo ""
print_info "Build script completed!"
print_info "To run the C++ example: ./$BUILD_DIR/examples/example_basic"
if [ "$BUILD_PYTHON_BINDINGS" = "ON" ]; then
    print_info "To test Python: python examples/basic_usage.py"
fi

