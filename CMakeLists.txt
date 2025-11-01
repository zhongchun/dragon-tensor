cmake_minimum_required(VERSION 3.15)
project(DragonTensor VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find Python for bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Find NumPy (optional, but recommended)
    find_package(Python3 COMPONENTS NumPy QUIET)
    if(NOT Python3_NumPy_FOUND)
        message(WARNING "NumPy not found via CMake. Will attempt to find it at runtime.")
    endif()
    
    # Try to find pybind11 via CMake, if not found use Python to locate it
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        # Try to get pybind11 cmake dir from Python
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; import os; print(os.path.join(os.path.dirname(pybind11.__file__), 'share', 'cmake', 'pybind11'))"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(pybind11_DIR AND EXISTS "${pybind11_DIR}")
            list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
            find_package(pybind11 REQUIRED)
        else()
            # Try alternative: use pybind11's Python module directly
            execute_process(
                COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE pybind11_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(pybind11_CMAKE_DIR AND EXISTS "${pybind11_CMAKE_DIR}")
                list(APPEND CMAKE_PREFIX_PATH ${pybind11_CMAKE_DIR})
                find_package(pybind11 REQUIRED)
            else()
                message(FATAL_ERROR "pybind11 not found. Please install it: pip install pybind11")
            endif()
        endif()
    endif()
endif()

# Core library - implementation is in src/tensor.cpp
file(GLOB_RECURSE CORE_SOURCES
    "src/*.cpp"
)

add_library(dragon_tensor STATIC
    ${CORE_SOURCES}
)

target_include_directories(dragon_tensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(dragon_tensor PUBLIC cxx_std_17)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(dragon_tensor_py
        python/bindings.cpp
    )
    
    target_link_libraries(dragon_tensor_py PRIVATE
        dragon_tensor
        pybind11::module
        pybind11::lto
    )
    
    # Windows-specific extras
    if(WIN32)
        target_link_libraries(dragon_tensor_py PRIVATE pybind11::windows-extras)
    endif()
    
    # Set output directory for Python module
    set_target_properties(dragon_tensor_py PROPERTIES
        OUTPUT_NAME "dragon_tensor"
        SUFFIX ${Python3_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
    
    if(APPLE)
        set_target_properties(dragon_tensor_py PROPERTIES
            INSTALL_RPATH "@loader_path"
        )
    endif()
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(TARGETS dragon_tensor
    EXPORT DragonTensorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

if(BUILD_PYTHON_BINDINGS)
    install(TARGETS dragon_tensor_py
        LIBRARY DESTINATION ${Python3_SITEARCH}
    )
endif()

