"""
Dragon Tensor - High-performance tensor library for financial data analysis

This library provides efficient tensor operations optimized for quantitative finance,
with seamless integration with NumPy, Pandas, and PyTorch.
"""

# Import the C++ module directly (built by pybind11)
try:
    # First try to import the compiled C++ module
    import sys
    import os
    
    # Add build directory to path if the module isn't installed
    build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'build')
    if build_dir not in sys.path and os.path.exists(build_dir):
        sys.path.insert(0, build_dir)
    
    # Import the C++ extension module
    import dragon_tensor as _dt_core
    from dragon_tensor import (
        TensorFloat, TensorDouble, TensorInt, TensorLong,
        from_numpy_float, from_numpy_double, from_numpy_int, from_numpy_long,
        from_pandas_series, from_pandas_dataframe, from_torch
    )
    
    # Import wrapper functions
    try:
        from .wrapper import (
            from_numpy, to_numpy, from_pandas, to_pandas, from_torch as wrapper_from_torch, to_torch
        )
    except ImportError:
        pass
    
    __version__ = "1.0.0"
    __all__ = [
        "TensorFloat",
        "TensorDouble", 
        "TensorInt",
        "TensorLong",
        "from_numpy_float",
        "from_numpy_double",
        "from_numpy_int",
        "from_numpy_long",
        "from_pandas_series",
        "from_pandas_dataframe",
        "from_torch",
    ]
    
    # Add wrapper functions if available
    try:
        __all__.extend(["from_numpy", "to_numpy", "from_pandas", "to_pandas", "to_torch"])
    except:
        pass
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import dragon_tensor C++ module: {e}")
    __all__ = []

