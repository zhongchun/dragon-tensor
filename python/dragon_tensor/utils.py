"""
Utility functions for dtype conversion, device checks, and interoperability
"""

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pyarrow as pa

    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

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
        return dragon_tensor.from_numpy_float(arr)
    elif dtype == np.float64:
        return dragon_tensor.from_numpy_double(arr)
    elif dtype == np.int32:
        return dragon_tensor.from_numpy_int(arr)
    elif dtype == np.int64:
        return dragon_tensor.from_numpy_long(arr)
    else:
        # Try to convert to float64
        arr = arr.astype(np.float64)
        return dragon_tensor.from_numpy_double(arr)


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
    if not HAS_TORCH:
        raise ImportError("torch is required for from_torch")

    return dragon_tensor.from_torch(tensor)


def to_torch(tensor, device=None, dtype=None):
    """Convert Dragon Tensor to PyTorch tensor (zero-copy)

    Args:
        tensor: Dragon Tensor
        device: Optional device for PyTorch tensor (will copy if not CPU)
        dtype: Optional dtype for PyTorch tensor (will copy if different)

    Returns:
        PyTorch tensor (zero-copy when device=None and dtype matches)
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for to_torch")

    # Get zero-copy NumPy array
    arr = tensor.to_numpy()

    # torch.from_numpy() creates zero-copy view
    torch_tensor = torch.from_numpy(arr)

    # Device/dtype conversions require copying
    if device is not None:
        torch_tensor = torch_tensor.to(device)
    if dtype is not None:
        torch_tensor = torch_tensor.to(dtype)

    return torch_tensor


def from_arrow(arrow_array):
    """Convert Arrow Array to Dragon Tensor

    Args:
        arrow_array: pyarrow.Array

    Returns:
        Dragon Tensor
    """
    if not HAS_ARROW:
        raise ImportError("pyarrow is required for from_arrow")

    # Placeholder - will be implemented when Arrow interop is added
    raise NotImplementedError("Arrow interop not yet implemented")


def to_arrow(tensor):
    """Convert Dragon Tensor to Arrow Array

    Args:
        tensor: Dragon Tensor

    Returns:
        pyarrow.Array
    """
    if not HAS_ARROW:
        raise ImportError("pyarrow is required for to_arrow")

    # Placeholder - will be implemented when Arrow interop is added
    raise NotImplementedError("Arrow interop not yet implemented")


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
