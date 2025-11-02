from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
from pybind11 import get_cmake_dir
import pybind11
import os


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
core_sources = [
    "src/tensor.cpp",
    "src/buffer.cpp",
    "src/backend_factory.cpp",
    "src/io.cpp",
    "src/storage.cpp",
    "src/utils/logging.cpp",
    "src/backends/memory_backend.cpp",
    "src/backends/mmap_backend.cpp",
    "src/backends/sharedmem_backend.cpp",
    "python/bindings.cpp",
]

ext_modules = [
    Pybind11Extension(
        "dragon_tensor",
        core_sources,
        include_dirs=[
            "include",
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=17,
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
