"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np
import dragon_tensor as dt


@pytest.fixture
def sample_1d_array():
    """Sample 1D numpy array for testing"""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)


@pytest.fixture
def sample_2d_array():
    """Sample 2D numpy array for testing"""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)


@pytest.fixture
def sample_1d_tensor(sample_1d_array):
    """Sample 1D Dragon Tensor"""
    return dt.from_numpy(sample_1d_array)


@pytest.fixture
def sample_2d_tensor(sample_2d_array):
    """Sample 2D Dragon Tensor"""
    return dt.from_numpy(sample_2d_array)


@pytest.fixture
def price_series():
    """Sample price series for financial tests"""
    return np.array([100.0, 102.0, 101.0, 105.0, 108.0, 110.0], dtype=np.float64)


@pytest.fixture
def price_tensor(price_series):
    """Sample price tensor for financial tests"""
    return dt.from_numpy(price_series)


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for I/O tests"""
    return tmp_path
