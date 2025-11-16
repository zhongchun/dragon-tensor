"""
File I/O operations for Dragon Tensor

Provides Pythonic wrappers for tensor persistence, including:
- save() and load() for file I/O
- open() context manager for memory-mapped files
- save_parquet() and load_parquet() for Arrow/Parquet integration
"""

import contextlib
from pathlib import Path
from typing import Union

try:
    import dragon_tensor
except ImportError:
    # Absolute import fallback
    import sys
    import os

    # Add parent directory to path if needed
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import dragon_tensor


def save(tensor, path: Union[str, Path], layout: str = "row"):
    """Save tensor to disk

    Args:
        tensor: Dragon Tensor to save
        path: File path to save to
        layout: Memory layout ("row" or "column")

    Example:
        >>> tensor = dt.from_numpy(np.array([1, 2, 3]))
        >>> dt.save(tensor, "data.dt", layout="row")
    """
    from dragon_tensor import Layout

    layout_enum = Layout.RowMajor if layout == "row" else Layout.ColumnMajor
    tensor.save(str(path), layout_enum)


def load(path: Union[str, Path], mmap: bool = True):
    """Load tensor from disk

    Automatically detects the tensor dtype from the file header and loads
    using the appropriate tensor type.

    Args:
        path: File path to load from
        mmap: If True, use memory-mapped I/O (zero-copy for large files)

    Returns:
        Dragon Tensor (type depends on dtype stored in file)

    Example:
        >>> tensor = dt.load("data.dt", mmap=True)

    Raises:
        RuntimeError: If file format is invalid or dtype is unsupported
    """
    # Read dtype from file header
    try:
        dtype_int = dragon_tensor._dt_core.read_dtype_from_file(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read dtype from file: {e}")

    # Map DType enum to tensor class
    # DType enum values: FLOAT32=0, FLOAT64=1, INT32=2, INT64=3, UINT8=4
    if dtype_int == 0:  # FLOAT32
        return dragon_tensor.TensorFloat.load(str(path), mmap)
    elif dtype_int == 1:  # FLOAT64
        return dragon_tensor.TensorDouble.load(str(path), mmap)
    elif dtype_int == 2:  # INT32
        return dragon_tensor.TensorInt.load(str(path), mmap)
    elif dtype_int == 3:  # INT64
        return dragon_tensor.TensorLong.load(str(path), mmap)
    else:
        raise RuntimeError(
            f"Unsupported dtype in file: {dtype_int}. "
            "Supported types: float32, float64, int32, int64"
        )


@contextlib.contextmanager
def open(path: Union[str, Path], mmap: bool = True):
    """Context manager for loading tensors from files

    Automatically handles resource cleanup.

    Args:
        path: File path to open
        mmap: If True, use memory-mapped I/O

    Yields:
        Dragon Tensor

    Example:
        >>> with dt.open("large_data.dt", mmap=True) as tensor:
        ...     result = tensor.sum()
    """
    tensor = load(path, mmap)
    try:
        yield tensor
    finally:
        if mmap:
            tensor.detach()


def save_parquet(tensor, path: Union[str, Path]):
    """Save tensor to Parquet file via Arrow

    Args:
        tensor: Dragon Tensor to save
        path: Parquet file path

    Example:
        >>> tensor = dt.from_numpy(np.random.randn(100, 50))
        >>> dt.save_parquet(tensor, "data.parquet")
    """
    # Placeholder - will be implemented when Arrow interop is added
    raise NotImplementedError("Parquet support not yet implemented")


def load_parquet(path: Union[str, Path], mmap: bool = True):
    """Load tensor from Parquet file via Arrow

    Args:
        path: Parquet file path
        mmap: If True, use memory-mapped I/O

    Returns:
        Dragon Tensor

    Example:
        >>> tensor = dt.load_parquet("data.parquet", mmap=True)
    """
    # Placeholder - will be implemented when Arrow interop is added
    raise NotImplementedError("Parquet support not yet implemented")


__all__ = ["save", "load", "open", "save_parquet", "load_parquet"]
