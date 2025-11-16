"""
Basic usage examples for Dragon Tensor
"""

import numpy as np
import dragon_tensor as dt


def example_basic_operations():
    """Basic tensor operations"""
    print("=== Basic Operations ===")

    # Create tensor from numpy
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    tensor = dt.from_numpy(data)

    print(f"Tensor shape: {tensor.shape()}")
    print(f"Tensor size: {tensor.size()}")
    print(f"Sum: {tensor.sum()}")
    print(f"Mean: {tensor.mean()}")
    print(f"Std: {tensor.std()}")
    print(f"Max: {tensor.max()}")
    print(f"Min: {tensor.min()}\n")


def example_arithmetic():
    """Arithmetic operations"""
    print("=== Arithmetic Operations ===")

    a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))

    print(f"a + b: {(a + b).to_numpy()}")
    print(f"a * 2: {(a * 2).to_numpy()}")
    print(f"a * b: {(a * b).to_numpy()}")
    print()


def example_financial():
    """Financial operations"""
    print("=== Financial Operations ===")

    # Simulate price series
    prices = np.array([100.0, 102.0, 101.0, 105.0, 108.0, 110.0], dtype=np.float64)
    tensor = dt.from_numpy(prices)

    # Calculate returns
    returns = tensor.returns()
    print(f"Prices: {prices}")
    print(f"Returns: {returns.to_numpy()}\n")

    # Rolling statistics
    rolling_mean = tensor.rolling_mean(3)
    rolling_std = tensor.rolling_std(3)

    print(f"Rolling Mean (window=3): {rolling_mean.to_numpy()}")
    print(f"Rolling Std (window=3): {rolling_std.to_numpy()}\n")


def example_correlation():
    """Correlation and covariance"""
    print("=== Correlation & Covariance ===")

    # Two correlated asset prices
    asset1 = dt.from_numpy(
        np.array([100.0, 102.0, 101.0, 105.0, 108.0], dtype=np.float64)
    )
    asset2 = dt.from_numpy(np.array([50.0, 51.0, 50.5, 52.5, 54.0], dtype=np.float64))

    corr = asset1.correlation(asset2)
    cov = asset1.covariance(asset2)

    print(f"Correlation: {corr.to_numpy()[0]:.4f}")
    print(f"Covariance: {cov.to_numpy()[0]:.4f}\n")


def example_2d_operations():
    """2D tensor operations"""
    print("=== 2D Tensor Operations ===")

    # Create 2D tensor
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    tensor = dt.from_numpy(data)

    print(f"Shape: {tensor.shape()}")
    print(f"Sum along axis 0 (columns): {tensor.sum(0).to_numpy()}")
    print(f"Mean along axis 1 (rows): {tensor.mean(1).to_numpy()}")

    # Transpose
    transposed = tensor.transpose()
    print(f"Transposed shape: {transposed.shape()}")
    print(f"Transposed data:\n{transposed.to_numpy()}\n")


def example_mathematical_functions():
    """Mathematical functions"""
    print("=== Mathematical Functions ===")

    data = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float64)
    tensor = dt.from_numpy(data)

    print(f"Original: {tensor.to_numpy()}")
    print(f"Abs: {tensor.abs().to_numpy()}")
    print(f"Sqrt: {tensor.sqrt().to_numpy()}")
    print(f"Exp: {tensor.exp().to_numpy()}")
    print(f"Log: {tensor.log().to_numpy()}")
    print(f"Pow(2): {tensor.pow(2.0).to_numpy()}\n")


def example_comparison_operations():
    """Comparison operations"""
    print("=== Comparison Operations ===")

    a = dt.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
    b = dt.from_numpy(np.array([2.0, 2.0, 2.0, 5.0], dtype=np.float64))

    print(f"a: {a.to_numpy()}")
    print(f"b: {b.to_numpy()}")
    print(f"a == b: {a == b}")
    print(f"a != b: {a != b}\n")


def example_slicing_and_indexing():
    """Slicing and indexing operations"""
    print("=== Slicing & Indexing ===")

    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    tensor = dt.from_numpy(data)

    print(f"Original: {tensor.to_numpy()}")
    print(f"First element: {tensor[0]}")
    print(f"Last element: {tensor[tensor.size() - 1]}")
    print(f"Slice [1:4]: {tensor.slice(1, 4).to_numpy()}")

    # 2D slicing
    data_2d = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
    )
    tensor_2d = dt.from_numpy(data_2d)
    print(f"\n2D tensor:\n{tensor_2d.to_numpy()}")
    print(f"Row 1: {tensor_2d.slice_row(1).to_numpy()}")
    print(f"Column 0: {tensor_2d.slice_column(0).to_numpy()}\n")


def example_reshaping():
    """Reshaping operations"""
    print("=== Reshaping Operations ===")

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    tensor = dt.from_numpy(data)

    print(f"Original shape: {tensor.shape()}")
    print(f"Original: {tensor.to_numpy()}")

    # Reshape to 2x3
    reshaped = tensor.reshape([2, 3])
    print(f"Reshaped to [2, 3]:\n{reshaped.to_numpy()}")

    # Flatten
    flattened = reshaped.flatten()
    print(f"Flattened: {flattened.to_numpy()}\n")


def example_matrix_operations():
    """Matrix operations"""
    print("=== Matrix Operations ===")

    # Matrix multiplication
    a = dt.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    b = dt.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))

    print(f"Matrix A:\n{a.to_numpy()}")
    print(f"Matrix B:\n{b.to_numpy()}")

    result = a.matmul(b)
    print(f"A @ B:\n{result.to_numpy()}")

    # Transpose
    a_t = a.transpose()
    print(f"A^T:\n{a_t.to_numpy()}\n")


def example_statistical_operations():
    """Statistical operations with axis"""
    print("=== Statistical Operations (with axis) ===")

    data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
    )
    tensor = dt.from_numpy(data)

    print(f"Data:\n{tensor.to_numpy()}")
    print(f"Sum along axis 0 (columns): {tensor.sum(0).to_numpy()}")
    print(f"Sum along axis 1 (rows): {tensor.sum(1).to_numpy()}")
    print(f"Mean along axis 0: {tensor.mean(0).to_numpy()}")
    print(f"Max along axis 1: {tensor.max(1).to_numpy()}")
    print(f"Min along axis 0: {tensor.min(0).to_numpy()}")
    print(f"Var along axis 1: {tensor.var(1).to_numpy()}")
    print(f"Std along axis 0: {tensor.std(0).to_numpy()}\n")


def example_in_place_operations():
    """In-place operations"""
    print("=== In-Place Operations ===")

    a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))

    print(f"a: {a.to_numpy()}")
    print(f"b: {b.to_numpy()}")

    a += b
    print(f"After a += b: {a.to_numpy()}")

    a *= 2.0
    print(f"After a *= 2: {a.to_numpy()}")

    a -= 1.0
    print(f"After a -= 1: {a.to_numpy()}\n")


def example_3d_tensor():
    """3D tensor operations"""
    print("=== 3D Tensor Operations ===")

    data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    tensor = dt.from_numpy(data)

    print(f"3D tensor shape: {tensor.shape()}")
    print(f"3D tensor:\n{tensor}")
    print(f"Total size: {tensor.size()}")
    print(f"Sum: {tensor.sum()}\n")


def example_numpy_integration():
    """NumPy integration"""
    print("=== NumPy Integration ===")

    # Create from NumPy
    np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    tensor = dt.from_numpy(np_arr)
    print(f"From NumPy: {tensor.to_numpy()}")

    # Convert back to NumPy (zero-copy when possible)
    back_to_numpy = tensor.to_numpy()
    print(f"Back to NumPy: {back_to_numpy}")
    print(
        f"Same memory: {np.shares_memory(np_arr, back_to_numpy) if np_arr.shape == back_to_numpy.shape else False}\n"
    )


def example_io_operations():
    """File I/O operations"""
    print("=== File I/O Operations ===")

    import tempfile
    import os

    # Create a tensor
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    tensor = dt.from_numpy(data)

    # Save to file
    with tempfile.NamedTemporaryFile(suffix=".dt", delete=False) as f:
        temp_path = f.name

    try:
        tensor.save(temp_path)
        print(f"Saved tensor to: {temp_path}")

        # Load from file (with auto dtype detection)
        loaded = dt.load(temp_path, mmap=False)
        print(f"Loaded tensor: {loaded.to_numpy()}")
        print(f"Shape matches: {tensor.shape() == loaded.shape()}")

        # Load with memory mapping
        loaded_mmap = dt.load(temp_path, mmap=True)
        print(f"Loaded with mmap: {loaded_mmap.to_numpy()}")
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print()


def example_advanced_financial():
    """Advanced financial operations"""
    print("=== Advanced Financial Operations ===")

    # Multiple assets
    prices1 = np.array([100.0, 102.0, 101.0, 105.0, 108.0], dtype=np.float64)
    prices2 = np.array([50.0, 51.0, 50.5, 52.5, 54.0], dtype=np.float64)

    asset1 = dt.from_numpy(prices1)
    asset2 = dt.from_numpy(prices2)

    returns1 = asset1.returns()
    returns2 = asset2.returns()

    print(f"Asset 1 returns: {returns1.to_numpy()}")
    print(f"Asset 2 returns: {returns2.to_numpy()}")

    # Portfolio (equal weighted)
    portfolio_returns = (returns1 + returns2) / 2.0
    print(f"Portfolio returns: {portfolio_returns.to_numpy()}")

    # Correlation and covariance
    corr = asset1.correlation(asset2)
    cov = asset1.covariance(asset2)
    print(f"Correlation: {corr.to_numpy()[0]:.4f}")
    print(f"Covariance: {cov.to_numpy()[0]:.4f}")

    # Rolling statistics
    rolling_mean = asset1.rolling_mean(3)
    rolling_std = asset1.rolling_std(3)
    rolling_max = asset1.rolling_max(3)
    rolling_min = asset1.rolling_min(3)

    print(f"Rolling mean (window=3): {rolling_mean.to_numpy()}")
    print(f"Rolling std (window=3): {rolling_std.to_numpy()}")
    print(f"Rolling max (window=3): {rolling_max.to_numpy()}")
    print(f"Rolling min (window=3): {rolling_min.to_numpy()}\n")


def example_type_conversions():
    """Type conversions"""
    print("=== Type Conversions ===")

    # Create float64 tensor
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    tensor_double = dt.from_numpy(data)
    print(f"TensorDouble: {tensor_double}")

    # Create float32 tensor
    data_float = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor_float = dt.from_numpy(data_float)
    print(f"TensorFloat: {tensor_float}")

    # Create int64 tensor
    data_int = np.array([1, 2, 3], dtype=np.int64)
    tensor_int = dt.from_numpy(data_int)
    print(f"TensorLong: {tensor_int}")

    # Create int32 tensor
    data_int32 = np.array([1, 2, 3], dtype=np.int32)
    tensor_int32 = dt.from_numpy(data_int32)
    print(f"TensorInt: {tensor_int32}\n")


if __name__ == "__main__":
    example_basic_operations()
    example_arithmetic()
    example_financial()
    example_correlation()
    example_2d_operations()
    example_mathematical_functions()
    example_comparison_operations()
    example_slicing_and_indexing()
    example_reshaping()
    example_matrix_operations()
    example_statistical_operations()
    example_in_place_operations()
    example_3d_tensor()
    example_numpy_integration()
    example_io_operations()
    example_advanced_financial()
    example_type_conversions()
