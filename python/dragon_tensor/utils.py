"""
Utility functions for dtype conversion, device checks, and interoperability
"""

import numpy as np

# Lazy imports to reduce memory footprint
# Only check availability, don't import until needed

try:
    import importlib.util
    import sys

    # Check if pandas is available without importing
    spec = importlib.util.find_spec("pandas")
    HAS_PANDAS = spec is not None
except (ImportError, AttributeError):
    HAS_PANDAS = False

try:
    import importlib.util

    # Check if torch is available without importing
    spec = importlib.util.find_spec("torch")
    HAS_TORCH = spec is not None
except (ImportError, AttributeError):
    HAS_TORCH = False

try:
    import importlib.util

    # Check if pyarrow is available without importing
    spec = importlib.util.find_spec("pyarrow")
    HAS_ARROW = spec is not None
except (ImportError, AttributeError):
    HAS_ARROW = False


# Lazy import functions for torch and pyarrow
def _lazy_import_torch():
    """Lazy import torch only when needed"""
    if not HAS_TORCH:
        raise ImportError("torch is not installed")
    if "torch" not in sys.modules:
        import warnings
        import os

        # Suppress NumPy compatibility warnings during torch import
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            old_stderr = sys.stderr
            try:
                sys.stderr = open(os.devnull, "w")
                import torch
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr
    return sys.modules["torch"]


def _lazy_import_pyarrow():
    """Lazy import pyarrow only when needed"""
    if not HAS_ARROW:
        raise ImportError("pyarrow is not installed")
    if "pyarrow" not in sys.modules:
        import pyarrow
    return sys.modules["pyarrow"]


try:
    import dragon_tensor
except ImportError:
    # Absolute import fallback
    import sys
    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import dragon_tensor


def from_numpy(arr):
    """Convert numpy array to Dragon Tensor

    Args:
        arr: numpy array

    Returns:
        Dragon Tensor with appropriate dtype
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    dtype = arr.dtype
    if dtype == np.float32:
        return dragon_tensor._from_numpy_float(arr)
    elif dtype == np.float64:
        return dragon_tensor._from_numpy_double(arr)
    elif dtype == np.int32:
        return dragon_tensor._from_numpy_int(arr)
    elif dtype == np.int64:
        return dragon_tensor._from_numpy_long(arr)
    else:
        # Try to convert to float64
        arr = arr.astype(np.float64)
        return dragon_tensor._from_numpy_double(arr)


def to_numpy(tensor):
    """Convert Dragon Tensor to numpy array

    Args:
        tensor: Dragon Tensor

    Returns:
        numpy array
    """
    return tensor.to_numpy()


def from_pandas(obj):
    """Convert pandas Series or DataFrame to Dragon Tensor

    Args:
        obj: pandas Series or DataFrame

    Returns:
        Dragon Tensor
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for from_pandas")

    # Lazy import pandas only when this function is called
    import pandas as pd

    if isinstance(obj, pd.Series):
        return dragon_tensor.from_pandas_series(obj)
    elif isinstance(obj, pd.DataFrame):
        return dragon_tensor.from_pandas_dataframe(obj)
    else:
        raise TypeError(f"Expected pandas Series or DataFrame, got {type(obj)}")


def to_pandas(tensor, index=None, columns=None):
    """Convert Dragon Tensor to pandas DataFrame or Series

    Args:
        tensor: Dragon Tensor
        index: Optional index for DataFrame/Series
        columns: Optional column names for DataFrame

    Returns:
        pandas DataFrame or Series
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for to_pandas")

    # Lazy import pandas only when this function is called
    import pandas as pd

    arr = tensor.to_numpy()

    if tensor.ndim() == 1:
        return pd.Series(arr, index=index)
    elif tensor.ndim() == 2:
        return pd.DataFrame(arr, index=index, columns=columns)
    else:
        raise ValueError("Can only convert 1D or 2D tensors to pandas")


def from_torch(tensor):
    """Convert PyTorch tensor to Dragon Tensor

    Args:
        tensor: PyTorch tensor

    Returns:
        Dragon Tensor
    """
    # Lazy import torch only when this function is called
    torch = _lazy_import_torch()

    # Convert torch tensor to numpy first (handles NumPy compatibility issues)
    # This is more reliable than using the C++ from_torch which may have NumPy issues
    try:
        # Move to CPU if on GPU
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        # Try to convert to numpy (zero-copy when possible)
        # Handle NumPy compatibility issues by using tolist() as fallback
        try:
            np_array = tensor.detach().numpy()
        except (RuntimeError, AttributeError):
            # Fallback: use tolist() if numpy() fails (NumPy compatibility issues)
            data = tensor.detach().cpu().tolist()
            np_array = np.array(
                data, dtype=np.float64 if tensor.dtype == torch.float64 else np.float32
            )

        # Convert numpy to dragon tensor
        return from_numpy(np_array)
    except Exception as e:
        # Fallback: try C++ module directly
        try:
            import sys

            if "dragon_tensor" in sys.modules:
                dt_module = sys.modules["dragon_tensor"]
                if hasattr(dt_module, "_dt_core"):
                    return dt_module._dt_core.from_torch(tensor)

            import _dragon_tensor_cpp as _dt_core

            return _dt_core.from_torch(tensor)
        except (AttributeError, ImportError, RuntimeError):
            raise RuntimeError(f"Failed to convert torch tensor: {e}")


def to_torch(tensor, device=None, dtype=None):
    """Convert Dragon Tensor to PyTorch tensor (zero-copy)

    Args:
        tensor: Dragon Tensor
        device: Optional device for PyTorch tensor (will copy if not CPU)
        dtype: Optional dtype for PyTorch tensor (will copy if different)

    Returns:
        PyTorch tensor (zero-copy when device=None and dtype matches)
    """
    # Lazy import torch only when this function is called
    torch = _lazy_import_torch()

    # Get NumPy array
    arr = tensor.to_numpy()

    # Try to create PyTorch tensor from NumPy (zero-copy when possible)
    # Handle NumPy compatibility issues
    try:
        torch_tensor = torch.from_numpy(arr)
    except (RuntimeError, AttributeError):
        # Fallback: create tensor from list if numpy() fails (NumPy compatibility issues)
        data = arr.tolist()
        torch_dtype = torch.float64 if arr.dtype == np.float64 else torch.float32
        torch_tensor = torch.tensor(data, dtype=torch_dtype)

    # Device/dtype conversions require copying
    if device is not None:
        torch_tensor = torch_tensor.to(device)
    if dtype is not None:
        torch_tensor = torch_tensor.to(dtype)

    return torch_tensor


def from_arrow(arrow_array):
    """Convert Arrow Array to Dragon Tensor (zero-copy when possible)

    Args:
        arrow_array: pyarrow.Array

    Returns:
        Dragon Tensor
    """
    # Lazy import pyarrow only when this function is called
    pa = _lazy_import_pyarrow()

    if not isinstance(arrow_array, pa.Array):
        raise TypeError(f"Expected pyarrow.Array, got {type(arrow_array)}")

    # Convert Arrow array to numpy (zero-copy when memory layout is compatible)
    # Arrow's to_numpy() method can do zero-copy conversion for compatible types
    np_arr = arrow_array.to_numpy(zero_copy_only=False)

    # Convert numpy array to Dragon Tensor
    # This uses the existing from_numpy which handles dtype conversion
    return from_numpy(np_arr)


def to_arrow(tensor):
    """Convert Dragon Tensor to Arrow Array (zero-copy when possible)

    Args:
        tensor: Dragon Tensor

    Returns:
        pyarrow.Array
    """
    # Lazy import pyarrow only when this function is called
    pa = _lazy_import_pyarrow()

    # Convert tensor to numpy (zero-copy)
    np_arr = to_numpy(tensor)

    # Convert numpy array to Arrow array (zero-copy when memory layout is compatible)
    # pa.array() can do zero-copy conversion for compatible numpy arrays
    return pa.array(np_arr)


__all__ = [
    "from_numpy",
    "to_numpy",
    "from_pandas",
    "to_pandas",
    "from_torch",
    "to_torch",
    "from_arrow",
    "to_arrow",
    "HAS_PANDAS",
    "HAS_TORCH",
    "HAS_ARROW",
]
