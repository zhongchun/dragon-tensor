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


if __name__ == "__main__":
    example_basic_operations()
    example_arithmetic()
    example_financial()
    example_correlation()
    example_2d_operations()
