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

    # Add build directory to path if the module isn't installed
    build_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "build"
    )
    if build_dir not in sys.path and os.path.exists(build_dir):
        sys.path.insert(0, build_dir)

    # Import the C++ extension module
    import dragon_tensor as _dt_core
    from dragon_tensor import (
        TensorFloat,
        TensorDouble,
        TensorInt,
        TensorLong,
        from_numpy_float,
        from_numpy_double,
        from_numpy_int,
        from_numpy_long,
        from_pandas_series,
        from_pandas_dataframe,
        from_torch,
    )

    # Import submodules
    from . import io, finance, shared, utils

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
            version_file = os.path.join(os.path.dirname(__file__), "..", "..", "VERSION.txt")
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
        "from_numpy_float",
        "from_numpy_double",
        "from_numpy_int",
        "from_numpy_long",
        "from_pandas_series",
        "from_pandas_dataframe",
        "from_torch",
        # Submodules
        "io",
        "finance",
        "shared",
        "utils",
    ]

    # Re-export commonly used functions from submodules
    from .utils import from_numpy, to_numpy, from_pandas, to_pandas, from_torch, to_torch
    from .io import save, load, open, save_parquet, load_parquet
    from .finance import (
        returns,
        rolling_mean,
        rolling_std,
        correlation,
        covariance,
    )
    from .shared import create_shared, attach_shared, detach, destroy_shared, flush

    __all__.extend([
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
    ])

except ImportError as e:
    import warnings

    warnings.warn(f"Could not import dragon_tensor C++ module: {e}")
    __all__ = []
