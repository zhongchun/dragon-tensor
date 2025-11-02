"""
Shared memory operations for Dragon Tensor

Provides functions for creating and managing shared-memory tensors
for cross-process access.
"""

from typing import Optional

try:
    import dragon_tensor
    from dragon_tensor import Layout
except ImportError:
    from .. import dragon_tensor
    from ..dragon_tensor import Layout


def create_shared(
    name: str,
    shape,
    dtype: str = "float64",
    layout: str = "row",
):
    """Create a shared-memory tensor

    Args:
        name: Shared memory name (must be unique)
        shape: Tensor shape (tuple or list)
        dtype: Data type ("float32", "float64", "int32", "int64")
        layout: Memory layout ("row" or "column")

    Returns:
        Dragon Tensor

    Example:
        >>> tensor = dt.create_shared(
        ...     "shared_prices", shape=(252, 1000), dtype="float64"
        ... )
    """
    layout_enum = Layout.RowMajor if layout == "row" else Layout.ColumnMajor

    # Map dtype string to Tensor class
    dtype_map = {
        "float32": dragon_tensor.TensorFloat,
        "float64": dragon_tensor.TensorDouble,
        "int32": dragon_tensor.TensorInt,
        "int64": dragon_tensor.TensorLong,
    }

    tensor_class = dtype_map.get(dtype)
    if tensor_class is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return tensor_class.create_shared(str(name), shape, layout_enum)


def attach_shared(name: str):
    """Attach to an existing shared-memory tensor

    Args:
        name: Shared memory name

    Returns:
        Dragon Tensor

    Example:
        >>> tensor = dt.attach_shared("shared_prices")
    """
    # This will need to be implemented in bindings
    # For now, use a generic approach
    return dragon_tensor.TensorDouble.attach_shared(str(name))


def detach(tensor):
    """Detach from shared-memory tensor

    Args:
        tensor: Shared-memory Dragon Tensor

    Example:
        >>> dt.detach(tensor)
    """
    tensor.detach()


def destroy_shared(name: str):
    """Destroy shared-memory tensor

    Args:
        name: Shared memory name

    Example:
        >>> dt.destroy_shared("shared_prices")
    """
    dragon_tensor.TensorDouble.destroy_shared(str(name))


def flush(tensor):
    """Force write-back for file-backed or shared-memory tensors

    Args:
        tensor: Dragon Tensor

    Example:
        >>> dt.flush(tensor)
    """
    tensor.flush()


__all__ = [
    "create_shared",
    "attach_shared",
    "detach",
    "destroy_shared",
    "flush",
]

