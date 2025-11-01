"""
Convenience wrappers for integration with NumPy, Pandas, and PyTorch
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
    import dragon_tensor
except ImportError:
    # If direct import fails, try relative import
    from .. import dragon_tensor


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

    Note:
        torch.from_numpy() creates a zero-copy view when the NumPy array
        is C-contiguous. Since tensor.to_numpy() returns a zero-copy view,
        this conversion is also zero-copy (unless device/dtype conversion is needed).
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for to_torch")

    # Get zero-copy NumPy array
    arr = tensor.to_numpy()

    # torch.from_numpy() creates zero-copy view (shares memory with NumPy)
    # This works because:
    # 1. tensor.to_numpy() returns zero-copy view of Tensor's data
    # 2. torch.from_numpy() creates zero-copy view of NumPy array
    # 3. Result: Tensor -> NumPy -> PyTorch all share the same memory!
    torch_tensor = torch.from_numpy(arr)

    # Device/dtype conversions require copying
    if device is not None:
        torch_tensor = torch_tensor.to(device)
    if dtype is not None:
        torch_tensor = torch_tensor.to(dtype)

    return torch_tensor


__all__ = [
    "from_numpy",
    "to_numpy",
    "from_pandas",
    "to_pandas",
    "from_torch",
    "to_torch",
]
