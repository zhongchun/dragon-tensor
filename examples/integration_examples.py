"""
Examples of integration with NumPy, Pandas, and PyTorch
"""

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Pandas not available, skipping pandas examples")

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available, skipping torch examples")

import dragon_tensor as dt

# Use functions directly from dragon_tensor
from_numpy = dt.from_numpy
to_numpy = lambda t: t.to_numpy()
from_pandas = dt.from_pandas_series
to_pandas = lambda t, **kwargs: pd.Series(t.to_numpy()) if HAS_PANDAS else None
from_torch = dt.from_torch
to_torch = lambda t, **kwargs: torch.from_numpy(t.to_numpy()) if HAS_TORCH else None


def example_numpy_integration():
    """NumPy integration examples"""
    print("=== NumPy Integration ===")

    # Create numpy array
    arr = np.random.randn(100).astype(np.float64)
    print(f"Original numpy array shape: {arr.shape}")

    # Convert to Dragon Tensor
    tensor = from_numpy(arr)
    print(f"Dragon Tensor shape: {tensor.shape()}")

    # Perform operations
    result = tensor.rolling_mean(10)

    # Convert back to numpy
    result_arr = to_numpy(result)
    print(f"Result numpy array shape: {result_arr.shape}")
    print()


def example_pandas_integration():
    """Pandas integration examples"""
    if not HAS_PANDAS:
        return

    print("=== Pandas Integration ===")

    # Create pandas Series
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    series = pd.Series(prices, index=dates, name="price")

    print(f"Original Series:\n{series.head()}\n")

    # Convert to Dragon Tensor
    tensor = from_pandas(series)
    print(f"Dragon Tensor shape: {tensor.shape()}")

    # Calculate returns
    returns = tensor.returns()

    # Convert back to pandas
    returns_series = to_pandas(returns, index=dates[1:])
    print(f"Returns Series:\n{returns_series.head()}\n")

    # DataFrame example
    df = pd.DataFrame(
        {
            "asset1": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "asset2": 50 + np.cumsum(np.random.randn(100) * 0.3),
        },
        index=dates,
    )

    print(f"DataFrame:\n{df.head()}\n")

    # Process each column
    for col in df.columns:
        col_tensor = from_pandas(df[col])
        rolling_vol = col_tensor.rolling_std(20)
        print(f"{col} rolling volatility: {rolling_vol.to_numpy()[-1]:.4f}")
    print()


def example_torch_integration():
    """PyTorch integration examples"""
    if not HAS_TORCH:
        return

    print("=== PyTorch Integration ===")

    # Create PyTorch tensor
    torch_tensor = torch.randn(100, dtype=torch.float64)
    print(f"PyTorch tensor shape: {torch_tensor.shape}")

    # Convert to Dragon Tensor
    dt_tensor = from_torch(torch_tensor)
    print(f"Dragon Tensor shape: {dt_tensor.shape()}")

    # Perform calculations
    rolling_mean = dt_tensor.rolling_mean(10)
    rolling_std = dt_tensor.rolling_std(10)

    # Convert back to PyTorch
    result_torch = to_torch(rolling_mean)
    print(f"Result PyTorch tensor shape: {result_torch.shape}")
    print(f"Result dtype: {result_torch.dtype}")
    print()

    # GPU example (if available)
    if torch.cuda.is_available():
        print("=== GPU Example ===")
        gpu_tensor = torch.randn(100, dtype=torch.float64, device="cuda")

        # Convert to CPU for Dragon Tensor (Dragon Tensor is CPU-only)
        cpu_tensor = gpu_tensor.cpu()
        dt_tensor = from_torch(cpu_tensor)

        # Process
        result = dt_tensor.rolling_mean(10)

        # Convert back to GPU
        result_gpu = to_torch(result, device="cuda")
        print(f"GPU tensor shape: {result_gpu.shape}")
        print(f"GPU tensor device: {result_gpu.device}\n")


def example_workflow():
    """Complete workflow example"""
    print("=== Complete Workflow ===")

    # Start with numpy
    prices = 100 + np.cumsum(np.random.randn(252) * 0.5)

    # Convert to Dragon Tensor
    tensor = from_numpy(prices)

    # Calculate returns and metrics
    returns = tensor.returns()
    rolling_vol = returns.rolling_std(20)

    # Convert back to numpy for further processing
    returns_arr = to_numpy(returns)
    vol_arr = to_numpy(rolling_vol)

    # Use with other libraries
    if HAS_PANDAS:
        df = pd.DataFrame({"returns": returns_arr, "volatility": vol_arr})
        print(f"Final DataFrame:\n{df.head()}\n")


if __name__ == "__main__":
    example_numpy_integration()
    example_pandas_integration()
    example_torch_integration()
    example_workflow()
