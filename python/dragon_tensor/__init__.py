"""
Dragon Tensor - High-performance tensor library for financial data analysis

This library provides efficient tensor operations optimized for quantitative finance,
with seamless integration with NumPy, Pandas, PyTorch, and Apache Arrow.
"""

# Import the C++ module directly (built by pybind11)
try:
    # First try to import the compiled C++ module
    import sys
    import os

    # The C++ module should be in the same directory as this __init__.py
    # Add the package directory to sys.path to ensure we can import it
    package_dir = os.path.dirname(__file__)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    # Also try adding build directory to path if the module isn't installed
    build_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "build"
    )
    # Try build/lib.<platform>/dragon_tensor directory
    if os.path.exists(build_dir):
        import platform
        import sysconfig

        plat_specifier = sysconfig.get_platform()
        build_lib_dir = os.path.join(
            build_dir, f"lib.{plat_specifier}", "dragon_tensor"
        )
        if os.path.exists(build_lib_dir) and build_lib_dir not in sys.path:
            sys.path.insert(0, build_lib_dir)
        # Also try the build directory itself
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)

    # Import the C++ extension module
    # The C++ module is now compiled as '_dragon_tensor_cpp' to avoid name conflict
    # with the Python package 'dragon_tensor'
    import _dragon_tensor_cpp as _dt_core

    # Import symbols from the C++ module
    TensorFloat = _dt_core.TensorFloat
    TensorDouble = _dt_core.TensorDouble
    TensorInt = _dt_core.TensorInt
    TensorLong = _dt_core.TensorLong
    # Keep type-specific functions private (used internally by utils.from_numpy)
    _from_numpy_float = _dt_core.from_numpy_float
    _from_numpy_double = _dt_core.from_numpy_double
    _from_numpy_int = _dt_core.from_numpy_int
    _from_numpy_long = _dt_core.from_numpy_long
    from_pandas_series = _dt_core.from_pandas_series
    from_pandas_dataframe = _dt_core.from_pandas_dataframe
    from_torch = _dt_core.from_torch

    # Export Layout and other enums early so submodules can import them
    # Try to get Layout from the C++ module
    try:
        Layout = _dt_core.Layout
    except AttributeError:
        # Layout is not exported from C++ module, create a simple enum-like object
        # This is a fallback - submodules should handle this gracefully
        class Layout:
            RowMajor = "row"
            ColumnMajor = "column"

    try:
        DType = _dt_core.DType
    except AttributeError:
        DType = None
    try:
        StorageMode = _dt_core.StorageMode
    except AttributeError:
        StorageMode = None

    # Import submodules using relative imports (now that Layout is available)
    from . import io
    from . import finance
    from . import shared
    from . import utils

    # Read version from VERSION.txt (single source of truth)
    try:
        # Try to get from installed package first
        from importlib.metadata import version

        __version__ = version("dragon-tensor")
    except ImportError:
        # Fallback for older Python versions
        try:
            import pkg_resources

            __version__ = pkg_resources.get_distribution("dragon-tensor").version
        except:
            # Final fallback - read directly from VERSION.txt
            import os

            version_file = os.path.join(
                os.path.dirname(__file__), "..", "..", "VERSION.txt"
            )
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    __version__ = f.read().strip()
            else:
                raise RuntimeError("VERSION.txt not found - cannot determine version")
    __all__ = [
        # Core types
        "TensorFloat",
        "TensorDouble",
        "TensorInt",
        "TensorLong",
        # Factory functions
        "from_pandas_series",
        "from_pandas_dataframe",
        "from_torch",
        # Submodules
        "io",
        "finance",
        "shared",
        "utils",
    ]

    # Re-export commonly used functions from submodules (using already imported modules)
    from_numpy = utils.from_numpy
    to_numpy = utils.to_numpy
    from_pandas = utils.from_pandas
    to_pandas = utils.to_pandas
    from_torch = utils.from_torch
    to_torch = utils.to_torch
    save = io.save
    load = io.load
    open = io.open
    save_parquet = io.save_parquet
    load_parquet = io.load_parquet
    returns = finance.returns
    rolling_mean = finance.rolling_mean
    rolling_std = finance.rolling_std
    correlation = finance.correlation
    covariance = finance.covariance
    create_shared = shared.create_shared
    attach_shared = shared.attach_shared
    detach = shared.detach
    destroy_shared = shared.destroy_shared
    flush = shared.flush

    __all__.extend(
        [
            "from_numpy",
            "to_numpy",
            "from_pandas",
            "to_pandas",
            "from_torch",
            "to_torch",
            "save",
            "load",
            "open",
            "save_parquet",
            "load_parquet",
            "returns",
            "rolling_mean",
            "rolling_std",
            "correlation",
            "covariance",
            "create_shared",
            "attach_shared",
            "detach",
            "destroy_shared",
            "flush",
        ]
    )

except ImportError as e:
    import warnings

    warnings.warn(f"Could not import dragon_tensor C++ module: {e}")
    __all__ = []
