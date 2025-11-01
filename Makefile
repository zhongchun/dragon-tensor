.PHONY: build clean install install-dev test examples python-examples wheel all clean-all help format format-check format-cpp format-python lint rebuild verify

# Build directory
BUILD_DIR := build

# Default target - uses build.sh for automatic wheel generation
all: build-wheel

# Help target
help:
	@echo "Dragon Tensor Makefile"
	@echo ""
	@echo "Build Targets:"
	@echo "  make all          - Build C++ library and generate wheel (uses build.sh)"
	@echo "  make build        - Build C++ library only (direct cmake)"
	@echo "  make build-wheel  - Build C++ library and generate wheel (uses build.sh)"
	@echo "  make wheel        - Generate Python wheel only (requires build first)"
	@echo "  make rebuild      - Clean and rebuild everything"
	@echo "  make verify       - Build and run basic verification"
	@echo ""
	@echo "Installation:"
	@echo "  make install      - Install Python package"
	@echo "  make install-dev  - Install Python package in development mode"
	@echo ""
	@echo "Testing & Examples:"
	@echo "  make test         - Run tests"
	@echo "  make examples     - Build and run C++ examples"
	@echo "  make python-examples - Run Python examples"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format       - Format all files (C++ and Python)"
	@echo "  make format-check - Check formatting without modifying"
	@echo "  make format-cpp   - Format only C++ files (uses .clang-format)"
	@echo "  make format-python - Format only Python files"
	@echo "  make lint         - Check formatting (alias for format-check)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Clean build directory only"
	@echo "  make clean-all    - Clean build directory, wheels, and Python artifacts"
	@echo ""

# Build with wheel generation (uses build.sh - recommended)
build-wheel:
	@./build.sh

# Build the project directly with cmake (no wheel)
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && cmake --build .

# Generate Python wheel (requires build to be done first)
wheel:
	@echo "Building Python wheel..."
	@if ! python3 -m pip show build > /dev/null 2>&1; then \
		echo "Installing build tools..."; \
		python3 -m pip install build wheel --quiet; \
	fi
	@python3 -m build --wheel
	@echo "Wheel built successfully in dist/"

# Clean build files only
clean:
	@rm -rf $(BUILD_DIR)
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Clean everything (build, wheels, Python artifacts) - matches build.sh --clean
clean-all:
	@rm -rf $(BUILD_DIR)
	@rm -rf dist
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Clean completed: build directory, wheels, and Python artifacts removed"

# Install Python package
install:
	@pip install .

# Install in development mode
install-dev:
	@pip install -e .

# Run tests
test:
	@cd $(BUILD_DIR) && ctest

# Build and run examples
examples: build
	@cd $(BUILD_DIR) && ./examples/example_basic

# Python examples
python-examples:
	@python examples/basic_usage.py
	@python examples/financial_analysis.py
	@python examples/integration_examples.py

# Rebuild everything (clean + build)
rebuild: clean-all
	@echo "Rebuilding from scratch..."
	@./build.sh

# Verify build (build and run basic example)
verify: build
	@echo "Verifying build..."
	@cd $(BUILD_DIR) && ./examples/example_basic
	@echo "✓ Build verification successful"

# Code formatting
format:
	@./format.sh

format-check:
	@./format.sh --check

format-cpp:
	@./format.sh -C

format-python:
	@./format.sh -P

# Lint (alias for format-check)
lint:
	@./format.sh --check

