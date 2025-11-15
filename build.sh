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
BUILD_WHEEL="ON"  # Build wheel by default
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
    --no-wheel              Disable wheel building (default: enabled)
    --with-tests            Enable building tests
    --no-examples           Disable building examples
    -i, --install           Install after building
    -p, --python PATH       Python executable path
    --clean                 Clean build directory, wheels, and Python artifacts
                           (exits after cleaning if no other options specified)
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
HAS_BUILD_OPTIONS=false  # Track if any build-related options were specified

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            HAS_BUILD_OPTIONS=true
            shift
            ;;
        -t|--build-type)
            BUILD_TYPE="$2"
            HAS_BUILD_OPTIONS=true
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --no-python)
            BUILD_PYTHON_BINDINGS="OFF"
            BUILD_WHEEL="OFF"
            HAS_BUILD_OPTIONS=true
            shift
            ;;
        --no-wheel)
            BUILD_WHEEL="OFF"
            HAS_BUILD_OPTIONS=true
            shift
            ;;
        --with-tests)
            BUILD_TESTS="ON"
            HAS_BUILD_OPTIONS=true
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            HAS_BUILD_OPTIONS=true
            shift
            ;;
        -i|--install)
            INSTALL="ON"
            HAS_BUILD_OPTIONS=true
            shift
            ;;
        -p|--python)
            PYTHON_EXECUTABLE="$2"
            HAS_BUILD_OPTIONS=true
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            HAS_BUILD_OPTIONS=true
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Clean operation
if [ "$CLEAN_BUILD" = true ]; then
    print_info "Cleaning build artifacts..."
    
    # Clean build directory
    if [ -d "$BUILD_DIR" ]; then
        print_info "Removing build directory: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
    
    # Clean Python build artifacts
    if [ -d "dist" ]; then
        print_info "Removing dist directory (wheels and sdist)"
        rm -rf dist
    fi
    
    if [ -d "build" ] && [ "$BUILD_DIR" != "build" ]; then
        # Additional build directory cleanup
        print_info "Removing additional build artifacts"
        rm -rf build/bdist.* build/lib.* build/temp.*
    fi
    
    # Clean Python egg-info and other artifacts
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Clean .so and .dylib files from Python package directory
    if [ -d "python/dragon_tensor" ]; then
        SO_FILES=$(find python/dragon_tensor \( -name "*.so" -o -name "*.dylib" \) 2>/dev/null | wc -l | tr -d ' ')
        if [ "$SO_FILES" -gt 0 ]; then
            print_info "Removing .so/.dylib files from python/dragon_tensor"
            find python/dragon_tensor \( -name "*.so" -o -name "*.dylib" \) -delete 2>/dev/null || true
        fi
    fi

    print_info "Clean completed successfully!"
    
    # If only --clean was specified (no build options), exit here
    if [ "$HAS_BUILD_OPTIONS" = false ]; then
        echo ""
        print_info "No build options specified. Exiting after clean."
        exit 0
    fi
fi

# Check dependencies (only if we're building)
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

# Setup ccache if available and enabled
USE_CCACHE="${USE_CCACHE:-1}"
if [ "$USE_CCACHE" = "1" ] || [ "$USE_CCACHE" = "true" ] || [ "$USE_CCACHE" = "yes" ]; then
    if command -v ccache &> /dev/null; then
        export CC="ccache $(command -v gcc 2>/dev/null || command -v clang 2>/dev/null || echo 'gcc')"
        export CXX="ccache $(command -v g++ 2>/dev/null || command -v clang++ 2>/dev/null || echo 'g++')"
        print_info "Using ccache for faster rebuilds"
        # Show ccache stats
        if command -v ccache &> /dev/null; then
            ccache -s > /dev/null 2>&1 && ccache -s | head -5
        fi
    else
        print_warning "ccache not found. Install it for faster rebuilds: brew install ccache (macOS) or apt-get install ccache (Linux)"
    fi
fi

# Check for ninja (faster build system)
USE_NINJA="${USE_NINJA:-1}"
NINJA_AVAILABLE=false
if [ "$USE_NINJA" = "1" ] || [ "$USE_NINJA" = "true" ] || [ "$USE_NINJA" = "yes" ]; then
    if command -v ninja &> /dev/null; then
        NINJA_AVAILABLE=true
        print_info "Using Ninja build system (faster than Make)"
    else
        print_warning "Ninja not found. Install it for faster builds: brew install ninja (macOS) or apt-get install ninja-build (Linux)"
    fi
fi

# Auto-detect number of cores for parallel builds
if [ -n "$JOBS" ]; then
    CORES="$JOBS"
else
    if command -v nproc &> /dev/null; then
        CORES=$(nproc)
    elif command -v sysctl &> /dev/null; then
        CORES=$(sysctl -n hw.ncpu)
    else
        CORES=4
    fi
fi
print_info "Using $CORES parallel jobs"

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

# Use Ninja generator if available
if [ "$NINJA_AVAILABLE" = true ]; then
    CMAKE_ARGS+=("-G" "Ninja")
fi

# Add optimization flags for Release builds
if [ "$BUILD_TYPE" = "Release" ]; then
    CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG")
fi

if [ -n "$PYTHON_EXECUTABLE" ]; then
    CMAKE_ARGS+=("-DPython3_EXECUTABLE=$PYTHON_EXECUTABLE")
fi

if [ -n "$PYBIND11_DIR" ]; then
    CMAKE_ARGS+=("-Dpybind11_DIR=$PYBIND11_DIR")
fi

CMAKE_ARGS+=("..")

print_info "CMake command: cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"

# Build with parallel jobs
print_info "Building with $CORES parallel jobs..."
BUILD_ARGS=()
if [ "$NINJA_AVAILABLE" = true ]; then
    # Ninja uses -j flag
    BUILD_ARGS+=("-j$CORES")
else
    # Make uses -j flag
    BUILD_ARGS+=("-j$CORES")
fi

cmake --build . "${BUILD_ARGS[@]}"

print_info "Build completed successfully!"

# Show ccache stats if used
if [ "$USE_CCACHE" = "1" ] || [ "$USE_CCACHE" = "true" ] || [ "$USE_CCACHE" = "yes" ]; then
    if command -v ccache &> /dev/null; then
        echo ""
        print_info "ccache statistics:"
        ccache -s | head -10
    fi
fi

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

# Build wheel if requested and Python bindings are enabled
if [ "$BUILD_WHEEL" = "ON" ] && [ "$BUILD_PYTHON_BINDINGS" = "ON" ]; then
    echo ""
    print_info "Building Python wheel using setup.py..."
    cd ..
    
    # Verify the CMake-built extension module exists
    PYTHON_MODULE=$(find "$BUILD_DIR" -name "dragon_tensor*.so" -o -name "dragon_tensor*.dylib" | head -1)
    if [ -z "$PYTHON_MODULE" ]; then
        print_error "Python extension module not found in $BUILD_DIR"
        print_info "Skipping wheel build"
    else
        print_info "Found extension module: $(basename $PYTHON_MODULE)"
        
        # Install required packages for wheel building
        $PYTHON_EXECUTABLE -m pip install --quiet --upgrade pip setuptools wheel build 2>/dev/null || true
        
        # Set environment variable so setup.py knows where to find the pre-built module
        export CMAKE_BUILD_DIR="$BUILD_DIR"
        
        # Build wheel using setup.py (which will copy the pre-built extension)
        print_info "Running: $PYTHON_EXECUTABLE -m build --wheel"
        $PYTHON_EXECUTABLE -m build --wheel --outdir dist
        
        # Show the built wheel
        WHEEL_FILE=$(ls -t dist/*.whl 2>/dev/null | head -1)
        if [ -n "$WHEEL_FILE" ] && [ -f "$WHEEL_FILE" ]; then
            print_info "Wheel built successfully: $(basename $WHEEL_FILE)"
            ls -lh "$WHEEL_FILE"
        else
            print_error "Failed to create wheel file"
        fi
    fi
    cd "$BUILD_DIR"
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
    if [ "$BUILD_WHEEL" = "ON" ]; then
        WHEEL_FILE=$(ls -t ../dist/*.whl 2>/dev/null | head -1 2>/dev/null)
        if [ -n "$WHEEL_FILE" ]; then
            print_info "Wheel available: dist/$(basename $WHEEL_FILE)"
        fi
    fi
fi

