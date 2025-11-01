#!/bin/bash

# Dragon Tensor Format Script
# Formats C++ files using clang-format (Google style)
# Formats Python files using black

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_section() {
    echo -e "${BLUE}[SECTION]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Format C++ and Python files in Dragon Tensor

OPTIONS:
    -h, --help              Show this help message
    -c, --check             Check formatting without modifying files
    -i, --in-place          Format files in-place (default)
    -C, --cpp-only          Format only C++ files
    -P, --python-only       Format only Python files
    --cpp-style STYLE       C++ style (default: Google)
                           Options: Google, LLVM, Chromium, Mozilla, WebKit

EXAMPLES:
    $0                      # Format all files in-place
    $0 --check              # Check formatting without modifying
    $0 -C                   # Format only C++ files
    $0 -P                   # Format only Python files
    $0 -i -C                # Format C++ files in-place

EOF
}

# Default values
CHECK_ONLY=false
CPP_ONLY=false
PYTHON_ONLY=false
IN_PLACE=true
CPP_STYLE="Google"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--check)
            CHECK_ONLY=true
            IN_PLACE=false
            shift
            ;;
        -i|--in-place)
            IN_PLACE=true
            CHECK_ONLY=false
            shift
            ;;
        -C|--cpp-only)
            CPP_ONLY=true
            shift
            ;;
        -P|--python-only)
            PYTHON_ONLY=true
            shift
            ;;
        --cpp-style)
            CPP_STYLE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check for clang-format
check_clang_format() {
    if ! command -v clang-format &> /dev/null; then
        print_error "clang-format is required but not installed."
        print_info "Install clang-format:"
        print_info "  macOS: brew install clang-format"
        print_info "  Ubuntu/Debian: sudo apt-get install clang-format"
        print_info "  Or download from: https://clang.llvm.org/docs/ClangFormat.html"
        return 1
    fi
    
    CLANG_VERSION=$(clang-format --version | head -n1)
    print_info "Found $CLANG_VERSION"
    return 0
}

# Check for Python formatters
check_python_tools() {
    local missing_tools=()
    
    if ! python3 -m pip show black > /dev/null 2>&1; then
        missing_tools+=("black")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        print_warning "Python formatting tools not found. Installing..."
        python3 -m pip install --quiet "${missing_tools[@]}"
    fi
    
    return 0
}

# Format C++ files
format_cpp() {
    print_section "Formatting C++ files (Google style via .clang-format)..."
    
    if ! check_clang_format; then
        return 1
    fi
    
    # Check if .clang-format exists, if not use Google style flag
    if [ -f ".clang-format" ]; then
        STYLE_FLAG=""
        print_info "Using .clang-format configuration file"
    else
        # Fallback to Google style if no config file
        case "$CPP_STYLE" in
            Google)
                STYLE_FLAG="--style=Google"
                ;;
            LLVM)
                STYLE_FLAG="--style=LLVM"
                ;;
            Chromium)
                STYLE_FLAG="--style=Chromium"
                ;;
            Mozilla)
                STYLE_FLAG="--style=Mozilla"
                ;;
            WebKit)
                STYLE_FLAG="--style=WebKit"
                ;;
            *)
                print_warning "Unknown style '$CPP_STYLE', using Google"
                STYLE_FLAG="--style=Google"
                ;;
        esac
    fi
    
    # Find all C++ source files
    local cpp_files=(
        src/*.cpp
        src/**/*.cpp
        include/**/*.h
        include/**/*.hpp
        python/*.cpp
        examples/*.cpp
        tests/*.cpp
        tests/**/*.cpp
    )
    
    local formatted_count=0
    local error_count=0
    
    # Use find to get actual files (handles missing directories gracefully)
    while IFS= read -r -d '' file; do
        if [ -f "$file" ]; then
            if [ "$CHECK_ONLY" = true ]; then
                if [ -n "$STYLE_FLAG" ]; then
                    cmd="clang-format $STYLE_FLAG"
                else
                    cmd="clang-format"
                fi
                if ! $cmd "$file" | diff -u "$file" - > /dev/null 2>&1; then
                    print_warning "Needs formatting: $file"
                    ((error_count++))
                else
                    print_info "✓ $file"
                fi
            else
                print_info "Formatting: $file"
                if [ -n "$STYLE_FLAG" ]; then
                    clang-format -i $STYLE_FLAG "$file"
                else
                    clang-format -i "$file"
                fi
                ((formatted_count++))
            fi
        fi
    done < <(find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
             -not -path "./build/*" -not -path "./.git/*" -not -path "./dist/*" \
             -not -path "*/__pycache__/*" -print0 2>/dev/null)
    
    if [ "$CHECK_ONLY" = true ]; then
        if [ $error_count -eq 0 ]; then
            print_info "All C++ files are properly formatted!"
            return 0
        else
            print_error "$error_count C++ file(s) need formatting"
            return 1
        fi
    else
        print_info "Formatted $formatted_count C++ file(s)"
    fi
    
    return 0
}

# Format Python files
format_python() {
    print_section "Formatting Python files..."
    
    check_python_tools
    
    # Find all Python files
    local python_files=(
        python/**/*.py
        examples/*.py
        tests/**/*.py
        setup.py
    )
    
    local formatted_count=0
    local error_count=0
    
    while IFS= read -r -d '' file; do
        if [ -f "$file" ]; then
            if [ "$CHECK_ONLY" = true ]; then
                if ! black --check --quiet "$file" 2>/dev/null; then
                    print_warning "Needs formatting: $file"
                    ((error_count++))
                else
                    print_info "✓ $file"
                fi
            else
                print_info "Formatting: $file"
                black --quiet "$file"
                ((formatted_count++))
            fi
        fi
    done < <(find . -type f -name "*.py" \
             -not -path "./build/*" -not -path "./.git/*" -not -path "./dist/*" \
             -not -path "*/__pycache__/*" -not -path "./venv/*" -not -path "./.venv/*" \
             -print0 2>/dev/null)
    
    if [ "$CHECK_ONLY" = true ]; then
        if [ $error_count -eq 0 ]; then
            print_info "All Python files are properly formatted!"
            return 0
        else
            print_error "$error_count Python file(s) need formatting"
            return 1
        fi
    else
        print_info "Formatted $formatted_count Python file(s)"
    fi
    
    return 0
}

# Main execution
main() {
    print_info "Dragon Tensor Format Script"
    echo ""
    
    local cpp_result=0
    local python_result=0
    
    if [ "$PYTHON_ONLY" = true ]; then
        format_python
        python_result=$?
    elif [ "$CPP_ONLY" = true ]; then
        format_cpp
        cpp_result=$?
    else
        # Format both
        format_cpp
        cpp_result=$?
        
        echo ""
        format_python
        python_result=$?
    fi
    
    echo ""
    if [ "$CHECK_ONLY" = true ]; then
        if [ $cpp_result -eq 0 ] && [ $python_result -eq 0 ]; then
            print_info "✓ All files are properly formatted!"
            exit 0
        else
            print_error "Some files need formatting. Run without --check to fix."
            exit 1
        fi
    else
        print_info "Formatting completed!"
        exit 0
    fi
}

# Run main
main
