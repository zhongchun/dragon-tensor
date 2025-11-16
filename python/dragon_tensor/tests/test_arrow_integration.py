"""
Tests for Apache Arrow integration
"""

import numpy as np
import pytest
import dragon_tensor as dt

# Check if pyarrow is available
try:
    import pyarrow as pa

    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False


@pytest.mark.skipif(not HAS_ARROW, reason="pyarrow is not installed")
class TestArrowIntegration:
    """Test integration with Apache Arrow"""

    def test_round_trip_float64(self):
        """Test round-trip conversion float64"""
        original = pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float64())
        tensor = dt.from_arrow(original)
        result = dt.to_arrow(tensor)
        np.testing.assert_array_almost_equal(original.to_numpy(), result.to_numpy())

    def test_round_trip_float32(self):
        """Test round-trip conversion float32"""
        original = pa.array([1.5, 2.5, 3.5, 4.5], type=pa.float32())
        tensor = dt.from_arrow(original)
        result = dt.to_arrow(tensor)
        np.testing.assert_array_almost_equal(original.to_numpy(), result.to_numpy())

    def test_round_trip_int32(self):
        """Test round-trip conversion int32"""
        original = pa.array([1, 2, 3, 4, 5], type=pa.int32())
        tensor = dt.from_arrow(original)
        result = dt.to_arrow(tensor)
        np.testing.assert_array_equal(original.to_numpy(), result.to_numpy())

    def test_round_trip_int64(self):
        """Test round-trip conversion int64"""
        original = pa.array([10, 20, 30, 40, 50], type=pa.int64())
        tensor = dt.from_arrow(original)
        result = dt.to_arrow(tensor)
        np.testing.assert_array_equal(original.to_numpy(), result.to_numpy())

    def test_round_trip_2d(self):
        """Test round-trip conversion 2D array via flattening"""
        # Arrow arrays are 1D, so we test with flattened 2D data
        data_2d = np.random.randn(10, 20).astype(np.float64)
        original = pa.array(data_2d.flatten(), type=pa.float64())
        tensor = dt.from_arrow(original)
        result = dt.to_arrow(tensor)
        np.testing.assert_array_almost_equal(original.to_numpy(), result.to_numpy())

    def test_from_arrow_method(self):
        """Test using from_arrow as module function"""
        original = pa.array([1.0, 2.0, 3.0], type=pa.float64())
        tensor = dt.from_arrow(original)
        assert tensor.size() == 3
        result = dt.to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, original.to_numpy())

    def test_to_arrow_method(self):
        """Test using to_arrow as tensor method"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        tensor = dt.from_numpy(data)
        result_arrow = tensor.to_arrow()
        assert isinstance(result_arrow, pa.Array)
        np.testing.assert_array_almost_equal(data, result_arrow.to_numpy())

    def test_to_arrow_module_function(self):
        """Test using to_arrow as module function"""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        tensor = dt.from_numpy(data)
        result_arrow = dt.to_arrow(tensor)
        assert isinstance(result_arrow, pa.Array)
        np.testing.assert_array_almost_equal(data, result_arrow.to_numpy())

    def test_large_array(self):
        """Test with large Arrow array"""
        data = np.random.randn(10000).astype(np.float64)
        original = pa.array(data, type=pa.float64())
        tensor = dt.from_arrow(original)
        result = dt.to_arrow(tensor)
        np.testing.assert_array_almost_equal(original.to_numpy(), result.to_numpy())

    def test_empty_array_error(self):
        """Test with empty Arrow array (should raise error)"""
        original = pa.array([], type=pa.float64())
        with pytest.raises(RuntimeError, match="Shape dimensions"):
            dt.from_arrow(original)

    def test_invalid_type_error(self):
        """Test that non-Arrow array raises TypeError"""
        with pytest.raises(TypeError, match="Expected pyarrow.Array"):
            dt.from_arrow([1.0, 2.0, 3.0])

    def test_arrow_to_numpy_to_arrow(self):
        """Test Arrow -> Tensor -> NumPy -> Tensor -> Arrow"""
        original = pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float64())
        tensor1 = dt.from_arrow(original)
        numpy_array = dt.to_numpy(tensor1)
        tensor2 = dt.from_numpy(numpy_array)
        result = dt.to_arrow(tensor2)
        np.testing.assert_array_almost_equal(original.to_numpy(), result.to_numpy())

    def test_numpy_to_arrow_to_numpy(self):
        """Test NumPy -> Tensor -> Arrow -> NumPy"""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        tensor = dt.from_numpy(original)
        arrow_array = dt.to_arrow(tensor)
        result = arrow_array.to_numpy()
        np.testing.assert_array_almost_equal(original, result)

    def test_negative_values(self):
        """Test with negative values"""
        original = pa.array([-1.0, -2.0, 0.0, 2.0, 1.0], type=pa.float64())
        tensor = dt.from_arrow(original)
        result = dt.to_arrow(tensor)
        np.testing.assert_array_almost_equal(original.to_numpy(), result.to_numpy())

    def test_all_tensor_types_to_arrow(self):
        """Test that all tensor types can be converted to Arrow"""
        # Test TensorDouble
        data_double = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        tensor_double = dt.from_numpy(data_double)
        arrow_double = tensor_double.to_arrow()
        assert isinstance(arrow_double, pa.Array)

        # Test TensorFloat
        data_float = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor_float = dt.from_numpy(data_float)
        arrow_float = tensor_float.to_arrow()
        assert isinstance(arrow_float, pa.Array)

        # Test TensorInt
        data_int = np.array([1, 2, 3], dtype=np.int32)
        tensor_int = dt.from_numpy(data_int)
        arrow_int = tensor_int.to_arrow()
        assert isinstance(arrow_int, pa.Array)

        # Test TensorLong
        data_long = np.array([1, 2, 3], dtype=np.int64)
        tensor_long = dt.from_numpy(data_long)
        arrow_long = tensor_long.to_arrow()
        assert isinstance(arrow_long, pa.Array)

    @pytest.mark.requires_arrow
    def test_arrow_marker(self):
        """Test that requires_arrow marker works"""
        original = pa.array([1.0, 2.0, 3.0], type=pa.float64())
        tensor = dt.from_arrow(original)
        assert tensor.size() == 3
