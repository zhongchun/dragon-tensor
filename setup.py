from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
from pybind11 import get_cmake_dir
import pybind11
import os
import sys
import platform


# Read version from VERSION.txt (single source of truth)
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "VERSION.txt")
    with open(version_file, "r") as f:
        return f.read().strip()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# All C++ source files needed to build the extension
# Note: interop sources require pybind11, so they're only compiled with Python bindings
core_sources = [
    "src/tensor.cpp",
    "src/buffer.cpp",
    "src/allocator.cpp",
    "src/backend_factory.cpp",
    "src/io.cpp",
    "src/storage.cpp",
    "src/utils/logging.cpp",
    "src/backends/memory_backend.cpp",
    "src/backends/mmap_backend.cpp",
    "src/backends/sharedmem_backend.cpp",
    "src/backends/arrow_backend.cpp",
    "src/interop/arrow_interop.cpp",
    "src/interop/numpy_interop.cpp",
    "src/interop/torch_interop.cpp",
    "python/bindings.cpp",
]

# Configure compiler and linker flags for filesystem library
extra_compile_args = []
extra_link_args = []

# Handle std::filesystem linking (required for C++17)
if platform.system() == "Darwin":  # macOS
    # macOS 10.15+ has std::filesystem in libc++
    # Need to set deployment target to 10.15+ for std::filesystem support
    deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "10.15")
    extra_compile_args.append(f"-mmacosx-version-min={deployment_target}")
    extra_link_args.append(f"-mmacosx-version-min={deployment_target}")
elif platform.system() == "Linux":
    # Check compiler version for filesystem library requirement
    import subprocess

    try:
        # Try to get compiler version
        result = subprocess.run(
            ["g++", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "g++" in result.stdout:
            # GCC 8 or earlier needs explicit linking
            version_line = result.stdout.split("\n")[0]
            if "g++" in version_line:
                # Extract major version
                import re

                match = re.search(r"g\+\+.*?(\d+)\.(\d+)", version_line)
                if match:
                    major = int(match.group(1))
                    if major < 9:
                        extra_link_args.append("-lstdc++fs")
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        # Fallback: try linking filesystem library
        extra_link_args.append("-lstdc++fs")
elif platform.system() == "Windows":
    # Windows requires no special handling for std::filesystem in MSVC 2017+
    pass

ext_modules = [
    Pybind11Extension(
        "_dragon_tensor_cpp",
        core_sources,
        include_dirs=[
            "include",
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="dragon-tensor",
    version=get_version(),
    author="Dragon Tensor Contributors",
    description="High-performance tensor library for financial data analysis",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["dragon_tensor"],
    package_dir={"": "python"},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Office/Business :: Financial",
    ],
)
