"""
This setup.py does NOT compile C++ code. Instead, it uses the pre-built
extension module from CMake (built via build.sh). This separation allows:
- Faster Python builds (no C++ compilation in setup.py)
- Better build caching (CMake handles C++ compilation with ccache/ninja)
- Consistent builds (same C++ build process for all targets)

The build process:
1. Run './build.sh' to build C++ library and extension module with CMake
2. Run 'pip install .' or 'python -m build --wheel' to create Python package
   (setup.py will find and copy the pre-built extension from build directory)
"""

import glob
import platform
import os
import shutil
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


# Read version from VERSION.txt (single source of truth)
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "VERSION.txt")
    with open(version_file, "r") as f:
        return f.read().strip()


class CopyPrebuiltExtension(build_ext):
    """Custom build_ext that copies pre-built extension from CMake build directory."""

    def build_extensions(self):
        # Get build directory from environment or use default
        # CMAKE_BUILD_DIR is set by build.sh
        cmake_build_dir = os.environ.get("CMAKE_BUILD_DIR", "build")
        cmake_build_dir = os.path.abspath(cmake_build_dir)

        if not os.path.exists(cmake_build_dir):
            raise RuntimeError(
                f"CMake build directory not found: {cmake_build_dir}\n"
                f"Please run './build.sh' first to build the C++ extension module."
            )

        # Find the pre-built extension module
        # CMake builds it as dragon_tensor*.so or dragon_tensor*.dylib
        patterns = [
            os.path.join(cmake_build_dir, "**", "dragon_tensor*.so"),
            os.path.join(cmake_build_dir, "**", "dragon_tensor*.dylib"),
            os.path.join(cmake_build_dir, "dragon_tensor*.so"),
            os.path.join(cmake_build_dir, "dragon_tensor*.dylib"),
        ]

        extension_module = None
        for pattern in patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                extension_module = matches[0]
                break

        if not extension_module:
            raise RuntimeError(
                f"Pre-built extension module not found in {cmake_build_dir}\n"
                f"Please run './build.sh' first to build the C++ extension module."
            )

        print(f"Found pre-built extension: {extension_module}")

        # Build each extension (copy the pre-built module)
        for ext in self.extensions:
            # Determine the output filename
            ext_name = ext.name
            if "." in ext_name:
                # Convert module name like "dragon_tensor.core" to filename
                ext_name = ext_name.replace(".", os.sep)

            # Get the build directory for this extension
            build_lib = self.get_finalized_command("build_py").build_lib
            if not build_lib:
                build_lib = self.build_lib

            # Create output directory - extension goes in dragon_tensor package
            # The extension name is _dragon_tensor_cpp, but it should be in dragon_tensor package
            package_dir = "dragon_tensor"
            output_dir = os.path.join(build_lib, package_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Determine output filename based on platform
            if platform.system() == "Windows":
                ext_suffix = ".pyd"
            elif platform.system() == "Darwin":
                ext_suffix = ".so"  # Python expects .so even on macOS
            else:
                ext_suffix = ".so"

            # Get the base name from the pre-built module
            # CMake builds as dragon_tensorcpython-XX-platform.dylib (no dot between name and ABI)
            # or dragon_tensor.cpython-XX-platform.so (with dot)
            # Python code expects _dragon_tensor_cpp, so we need to rename it
            base_name = os.path.basename(extension_module)

            # Extract the ABI suffix and convert extension to .so for Python
            # Pattern 1: dragon_tensorcpython-312-darwin.dylib (no dot, .dylib extension)
            # Pattern 2: dragon_tensor.cpython-312-darwin.so (with dot, .so extension)
            if base_name.startswith("dragon_tensor"):
                # Remove "dragon_tensor" prefix
                suffix = base_name[len("dragon_tensor") :]

                # Handle both cases: "cpython-..." or ".cpython-..."
                if suffix.startswith("."):
                    # Case: dragon_tensor.cpython-312-darwin.so
                    # Replace .so with .so (already correct) or .dylib with .so
                    if suffix.endswith(".dylib"):
                        suffix = suffix[:-6] + ".so"
                    output_name = f"_dragon_tensor_cpp{suffix}"
                elif suffix.startswith("cpython"):
                    # Case: dragon_tensorcpython-312-darwin.dylib
                    # Need to add dot before cpython and convert .dylib to .so
                    if suffix.endswith(".dylib"):
                        suffix = suffix[:-6] + ".so"
                    output_name = f"_dragon_tensor_cpp.{suffix}"
                else:
                    # Fallback: just replace prefix and ensure .so extension
                    if suffix.endswith(".dylib"):
                        suffix = suffix[:-6] + ".so"
                    output_name = f"_dragon_tensor_cpp{suffix}"
            else:
                # Unknown format, try to preserve ABI info but use .so extension
                if "." in base_name:
                    # Try to extract ABI suffix
                    parts = base_name.split(".")
                    if len(parts) >= 2:
                        # Last part is extension, second-to-last might be ABI tag
                        abi_part = ".".join(parts[1:-1]) if len(parts) > 2 else ""
                        if abi_part:
                            output_name = f"_dragon_tensor_cpp.{abi_part}.so"
                        else:
                            output_name = f"_dragon_tensor_cpp.so"
                    else:
                        output_name = f"_dragon_tensor_cpp.so"
                else:
                    output_name = f"_dragon_tensor_cpp{ext_suffix}"

            output_path = os.path.join(output_dir, output_name)

            # Remove any existing .dylib file in the output directory (we only want .so)
            dylib_path = output_path.replace(".so", ".dylib")
            if os.path.exists(dylib_path) and dylib_path != output_path:
                os.remove(dylib_path)
                print(f"Removed .dylib file: {dylib_path}")

            # Copy the pre-built module
            print(f"Copying {extension_module} -> {output_path}")
            shutil.copy2(extension_module, output_path)

            # Also copy to the package directory for direct installation
            # Clean up any old .dylib files first (they shouldn't be in the package)
            package_dir = os.path.join(
                os.path.dirname(__file__), "python", "dragon_tensor"
            )
            if os.path.exists(package_dir):
                # Remove any old extension files to avoid duplicates
                for old_file in glob.glob(
                    os.path.join(package_dir, "_dragon_tensor_cpp.*")
                ):
                    if os.path.exists(old_file):
                        os.remove(old_file)
                        print(f"Removed old extension file: {old_file}")

                package_output = os.path.join(package_dir, output_name)
                print(f"Copying {extension_module} -> {package_output}")
                shutil.copy2(extension_module, package_output)


# Create a dummy extension - we'll copy the pre-built one in build_ext
# The name should match what Python expects to import (_dragon_tensor_cpp)
ext_modules = [
    Extension(
        "_dragon_tensor_cpp",  # This will become _dragon_tensor_cpp*.so in the package
        sources=[],  # No sources - we use pre-built module from CMake
    ),
]


# Most metadata is now in pyproject.toml
# Only C++ extension-specific configuration remains here
setup(
    version=get_version(),  # Override version from pyproject.toml with VERSION.txt
    ext_modules=ext_modules,
    cmdclass={"build_ext": CopyPrebuiltExtension},
    zip_safe=False,  # Required for C++ extensions
)
