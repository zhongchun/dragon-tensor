"""
Tests for basic tensor operations
"""

import numpy as np
import pytest
import dragon_tensor as dt


class TestTensorCreation:
    """Test tensor creation from numpy arrays"""

    def test_from_numpy_float64(self, sample_1d_array):
        """Test creating tensor from float64 array"""
        tensor = dt.from_numpy(sample_1d_array)
        assert tensor is not None
        assert tensor.size() == 5

    def test_from_numpy_float32(self):
        """Test creating tensor from float32 array"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = dt.from_numpy(arr)
        assert tensor is not None
        assert tensor.size() == 3

    def test_from_numpy_int32(self):
        """Test creating tensor from int32 array"""
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        tensor = dt.from_numpy(arr)
        assert tensor is not None
        assert tensor.size() == 4

    def test_from_numpy_int64(self):
        """Test creating tensor from int64 array"""
        arr = np.array([1, 2, 3, 4], dtype=np.int64)
        tensor = dt.from_numpy(arr)
        assert tensor is not None
        assert tensor.size() == 4

    def test_from_numpy_2d(self, sample_2d_array):
        """Test creating 2D tensor"""
        tensor = dt.from_numpy(sample_2d_array)
        assert tensor is not None
        shape = tensor.shape()
        assert len(shape) == 2
        assert shape[0] == 2
        assert shape[1] == 3

    def test_to_numpy(self, sample_1d_array, sample_1d_tensor):
        """Test converting tensor back to numpy"""
        result = dt.to_numpy(sample_1d_tensor)
        np.testing.assert_array_almost_equal(result, sample_1d_array)

    def test_shape(self, sample_1d_tensor, sample_2d_tensor):
        """Test tensor shape"""
        assert sample_1d_tensor.shape() == [5]
        assert sample_2d_tensor.shape() == [2, 3]

    def test_size(self, sample_1d_tensor, sample_2d_tensor):
        """Test tensor size"""
        assert sample_1d_tensor.size() == 5
        assert sample_2d_tensor.size() == 6


class TestBasicOperations:
    """Test basic mathematical operations"""

    def test_sum(self, sample_1d_tensor):
        """Test sum operation"""
        assert sample_1d_tensor.sum() == 15.0

    def test_mean(self, sample_1d_tensor):
        """Test mean operation"""
        assert sample_1d_tensor.mean() == 3.0

    def test_std(self, sample_1d_tensor):
        """Test standard deviation"""
        std = sample_1d_tensor.std()
        expected = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(std - expected) < 1e-10

    def test_max(self, sample_1d_tensor):
        """Test max operation"""
        assert sample_1d_tensor.max() == 5.0

    def test_min(self, sample_1d_tensor):
        """Test min operation"""
        assert sample_1d_tensor.min() == 1.0

    def test_sum_axis_0(self, sample_2d_tensor):
        """Test sum along axis 0"""
        result = sample_2d_tensor.sum(0)
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_mean_axis_1(self, sample_2d_tensor):
        """Test mean along axis 1"""
        result = sample_2d_tensor.mean(1)
        expected = np.array([2.0, 5.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)


class TestArithmeticOperations:
    """Test arithmetic operations"""

    def test_add_tensors(self):
        """Test adding two tensors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))
        result = a + b
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_multiply_scalar(self):
        """Test multiplying tensor by scalar"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        result = a * 2
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_multiply_tensors(self):
        """Test multiplying two tensors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))
        result = a * b
        expected = np.array([4.0, 10.0, 18.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_subtract_tensors(self):
        """Test subtracting two tensors"""
        a = dt.from_numpy(np.array([5.0, 7.0, 9.0], dtype=np.float64))
        b = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        result = a - b
        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_divide_tensors(self):
        """Test dividing two tensors"""
        a = dt.from_numpy(np.array([4.0, 10.0, 18.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))
        result = a / b
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)


class TestMatrixOperations:
    """Test matrix operations"""

    def test_transpose(self, sample_2d_tensor):
        """Test matrix transpose"""
        transposed = sample_2d_tensor.transpose()
        assert transposed.shape() == [3, 2]
        result = dt.to_numpy(transposed)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul(self):
        """Test matrix multiplication"""
        a = dt.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        b = dt.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))
        result = a.matmul(b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)


class TestTensorTypes:
    """Test different tensor types"""

    def test_tensor_float(self):
        """Test TensorFloat creation"""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        tensor = dt.from_numpy(arr)
        assert isinstance(tensor, dt.TensorFloat)

    def test_tensor_double(self, sample_1d_tensor):
        """Test TensorDouble creation"""
        assert isinstance(sample_1d_tensor, dt.TensorDouble)

    def test_tensor_int(self):
        """Test TensorInt creation"""
        arr = np.array([1, 2, 3], dtype=np.int32)
        tensor = dt.from_numpy(arr)
        assert isinstance(tensor, dt.TensorInt)

    def test_tensor_long(self):
        """Test TensorLong creation"""
        arr = np.array([1, 2, 3], dtype=np.int64)
        tensor = dt.from_numpy(arr)
        assert isinstance(tensor, dt.TensorLong)
