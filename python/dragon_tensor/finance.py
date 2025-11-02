"""
Financial analysis operations for Dragon Tensor

Provides financial-specific functions optimized for quantitative analysis,
including rolling window statistics, correlation, covariance, and returns.
"""

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


def returns(tensor):
    """Calculate returns: (x[i] - x[i-1]) / x[i-1]

    Args:
        tensor: Dragon Tensor (1D or 2D)

    Returns:
        Dragon Tensor of returns

    Example:
        >>> prices = dt.from_numpy(np.array([100.0, 102.0, 101.0, 105.0]))
        >>> rets = dt.finance.returns(prices)
    """
    return tensor.returns()


def rolling_mean(tensor, window: int):
    """Calculate rolling mean

    Args:
        tensor: Dragon Tensor
        window: Window size

    Returns:
        Dragon Tensor with rolling means

    Example:
        >>> data = dt.from_numpy(np.random.randn(100))
        >>> rolling_avg = dt.finance.rolling_mean(data, window=20)
    """
    return tensor.rolling_mean(window)


def rolling_std(tensor, window: int):
    """Calculate rolling standard deviation

    Args:
        tensor: Dragon Tensor
        window: Window size

    Returns:
        Dragon Tensor with rolling standard deviations

    Example:
        >>> data = dt.from_numpy(np.random.randn(100))
        >>> rolling_vol = dt.finance.rolling_std(data, window=20)
    """
    return tensor.rolling_std(window)


def rolling_sum(tensor, window: int):
    """Calculate rolling sum

    Args:
        tensor: Dragon Tensor
        window: Window size

    Returns:
        Dragon Tensor with rolling sums
    """
    return tensor.rolling_sum(window)


def rolling_max(tensor, window: int):
    """Calculate rolling maximum

    Args:
        tensor: Dragon Tensor
        window: Window size

    Returns:
        Dragon Tensor with rolling maxima
    """
    return tensor.rolling_max(window)


def rolling_min(tensor, window: int):
    """Calculate rolling minimum

    Args:
        tensor: Dragon Tensor
        window: Window size

    Returns:
        Dragon Tensor with rolling minima
    """
    return tensor.rolling_min(window)


def correlation(tensor1, tensor2):
    """Calculate correlation between two tensors

    Args:
        tensor1: First Dragon Tensor
        tensor2: Second Dragon Tensor

    Returns:
        Correlation value or matrix

    Example:
        >>> prices1 = dt.from_numpy(np.random.randn(100))
        >>> prices2 = dt.from_numpy(np.random.randn(100))
        >>> corr = dt.finance.correlation(prices1, prices2)
    """
    return tensor1.correlation(tensor2)


def covariance(tensor1, tensor2):
    """Calculate covariance between two tensors

    Args:
        tensor1: First Dragon Tensor
        tensor2: Second Dragon Tensor

    Returns:
        Covariance value or matrix

    Example:
        >>> prices1 = dt.from_numpy(np.random.randn(100))
        >>> prices2 = dt.from_numpy(np.random.randn(100))
        >>> cov = dt.finance.covariance(prices1, prices2)
    """
    return tensor1.covariance(tensor2)


__all__ = [
    "returns",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "rolling_max",
    "rolling_min",
    "correlation",
    "covariance",
]
