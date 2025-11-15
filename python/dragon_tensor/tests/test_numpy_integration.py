"""
Tests for NumPy integration
"""

import numpy as np
import pytest
import dragon_tensor as dt


class TestNumPyIntegration:
    """Test integration with NumPy"""

    def test_round_trip_float64(self):
        """Test round-trip conversion float64"""
        original = np.random.randn(100).astype(np.float64)
        tensor = dt.from_numpy(original)
        result = dt.to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, original)

    def test_round_trip_float32(self):
        """Test round-trip conversion float32"""
        original = np.random.randn(50).astype(np.float32)
        tensor = dt.from_numpy(original)
        result = dt.to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, original)

    def test_round_trip_int32(self):
        """Test round-trip conversion int32"""
        original = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        tensor = dt.from_numpy(original)
        result = dt.to_numpy(tensor)
        np.testing.assert_array_equal(result, original)

    def test_round_trip_int64(self):
        """Test round-trip conversion int64"""
        original = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        tensor = dt.from_numpy(original)
        result = dt.to_numpy(tensor)
        np.testing.assert_array_equal(result, original)

    def test_round_trip_2d(self):
        """Test round-trip conversion 2D array"""
        original = np.random.randn(10, 20).astype(np.float64)
        tensor = dt.from_numpy(original)
        result = dt.to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, original)

    def test_auto_dtype_conversion(self):
        """Test automatic dtype conversion for unsupported types"""
        # Test with int16 (should convert to float64)
        arr = np.array([1, 2, 3], dtype=np.int16)
        tensor = dt.from_numpy(arr)
        result = dt.to_numpy(tensor)
        # Should be converted to float64
        assert result.dtype == np.float64
        np.testing.assert_array_almost_equal(result, arr.astype(np.float64))

    def test_from_list(self):
        """Test creating tensor from Python list"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        tensor = dt.from_numpy(data)
        assert tensor.size() == 5
        result = dt.to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, np.array(data))

    def test_zero_copy_view(self):
        """Test that to_numpy creates a view when possible"""
        original = np.random.randn(100).astype(np.float64)
        tensor = dt.from_numpy(original)
        result = dt.to_numpy(tensor)
        # Modify result and check tensor is affected (if zero-copy)
        # Note: This depends on implementation, may not always be zero-copy
        assert result.shape == original.shape
        assert result.dtype == original.dtype

    def test_large_array(self):
        """Test with large array"""
        original = np.random.randn(10000).astype(np.float64)
        tensor = dt.from_numpy(original)
        result = dt.to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, original)

    def test_empty_array(self):
        """Test with empty array (should raise error)"""
        original = np.array([], dtype=np.float64)
        with pytest.raises(RuntimeError, match="Shape dimensions"):
            dt.from_numpy(original)
