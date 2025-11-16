"""
Tests for multiple tensor operations
"""

import numpy as np
import pytest
import dragon_tensor as dt


class TestMultipleTensorArithmetic:
    """Test arithmetic operations with multiple tensors"""

    def test_add_multiple_tensors(self):
        """Test adding multiple tensors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))
        c = dt.from_numpy(np.array([7.0, 8.0, 9.0], dtype=np.float64))

        result = a + b + c
        expected = np.array([12.0, 15.0, 18.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_multiply_multiple_tensors(self):
        """Test multiplying multiple tensors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([2.0, 3.0, 4.0], dtype=np.float64))
        c = dt.from_numpy(np.array([3.0, 4.0, 5.0], dtype=np.float64))

        result = a * b * c
        expected = np.array([6.0, 24.0, 60.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_chain_operations(self):
        """Test chaining multiple operations"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))
        c = dt.from_numpy(np.array([2.0, 2.0, 2.0], dtype=np.float64))

        result = (a + b) * c - a
        expected = np.array([9.0, 12.0, 15.0])  # (a+b)*c - a
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_arithmetic_with_different_types(self):
        """Test arithmetic operations with different tensor types"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float32))

        # Convert to same type first (tensors of different types can't be added directly)
        b_double = dt.from_numpy(dt.to_numpy(b).astype(np.float64))
        result = a + b_double
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_shape_mismatch_error(self):
        """Test that shape mismatch raises error"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0], dtype=np.float64))

        with pytest.raises(RuntimeError, match="Shape mismatch"):
            _ = a + b

    def test_2d_tensor_operations(self):
        """Test operations with multiple 2D tensors"""
        a = dt.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        b = dt.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))

        result = a + b
        expected = np.array([[6.0, 8.0], [10.0, 12.0]])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

        result = a * b
        expected = np.array([[5.0, 12.0], [21.0, 32.0]])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)


class TestMultipleTensorInPlace:
    """Test in-place operations with multiple tensors"""

    def test_inplace_add_multiple(self):
        """Test in-place addition with multiple operations"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))

        a += b
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(a), expected)

        a += b
        expected = np.array([9.0, 12.0, 15.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(a), expected)

    def test_inplace_multiply_multiple(self):
        """Test in-place multiplication with multiple operations"""
        a = dt.from_numpy(np.array([2.0, 3.0, 4.0], dtype=np.float64))
        b = dt.from_numpy(np.array([2.0, 2.0, 2.0], dtype=np.float64))

        a *= b
        expected = np.array([4.0, 6.0, 8.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(a), expected)

        a *= b
        expected = np.array([8.0, 12.0, 16.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(a), expected)

    def test_inplace_chain_operations(self):
        """Test chaining in-place operations"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))

        a += b  # a becomes [5.0, 7.0, 9.0]
        a *= 2.0  # a becomes [10.0, 14.0, 18.0]
        a -= b  # a becomes [6.0, 9.0, 12.0]

        expected = np.array([6.0, 9.0, 12.0])  # ((a+b)*2 - b)
        np.testing.assert_array_almost_equal(dt.to_numpy(a), expected)


class TestMultipleTensorComparison:
    """Test comparison operations with multiple tensors"""

    def test_equality_multiple_tensors(self):
        """Test equality comparison with multiple tensors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        c = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))

        assert a == b
        assert b == c
        assert a == c

    def test_inequality_multiple_tensors(self):
        """Test inequality comparison with multiple tensors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))
        c = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))

        assert a != b
        assert a == c
        assert b != c

    def test_comparison_chain(self):
        """Test chaining comparison operations"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        c = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))

        # All should be equal
        assert (a == b) == (b == c)


class TestMultipleTensorMatrixOperations:
    """Test matrix operations with multiple tensors"""

    def test_matmul_multiple(self):
        """Test matrix multiplication with multiple tensors"""
        a = dt.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        b = dt.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))
        c = dt.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64))

        # (A * B) * C should equal A * (B * C) for identity matrix C
        ab = a.matmul(b)
        abc = ab.matmul(c)
        bc = b.matmul(c)
        abc2 = a.matmul(bc)

        np.testing.assert_array_almost_equal(dt.to_numpy(abc), dt.to_numpy(abc2))

    def test_matmul_chain(self):
        """Test chaining matrix multiplications"""
        a = dt.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        b = dt.from_numpy(np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64))

        # A * B * B should equal A * (B * B)
        ab = a.matmul(b)
        abb = ab.matmul(b)
        bb = b.matmul(b)
        abb2 = a.matmul(bb)

        np.testing.assert_array_almost_equal(dt.to_numpy(abb), dt.to_numpy(abb2))

    def test_transpose_multiple(self):
        """Test transpose operations with multiple tensors"""
        a = dt.from_numpy(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        )
        b = a.transpose()

        # Double transpose should return original
        c = b.transpose()
        np.testing.assert_array_almost_equal(dt.to_numpy(a), dt.to_numpy(c))


class TestMultipleTensorStatistical:
    """Test statistical operations with multiple tensors"""

    def test_statistical_comparison(self):
        """Test comparing statistics of multiple tensors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        b = dt.from_numpy(np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64))

        mean_a = a.mean()
        mean_b = b.mean()

        assert mean_b == 2.0 * mean_a

    def test_statistical_operations_on_results(self):
        """Test statistical operations on results of tensor operations"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        b = dt.from_numpy(np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64))

        sum_tensor = a + b
        mean_sum = sum_tensor.mean()

        # Mean of sum should equal sum of means
        expected = a.mean() + b.mean()
        assert abs(mean_sum - expected) < 1e-10

    def test_std_of_sum(self):
        """Test standard deviation of sum of tensors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        b = dt.from_numpy(np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64))

        sum_tensor = a + b
        std_sum = sum_tensor.std()

        # Std of (a + constant) should equal std of a
        std_a = a.std()
        assert abs(std_sum - std_a) < 1e-10


class TestMultipleTensorFinancial:
    """Test financial operations with multiple tensors"""

    def test_returns_comparison(self):
        """Test comparing returns of multiple price series"""
        prices1 = dt.from_numpy(
            np.array([100.0, 102.0, 101.0, 105.0, 108.0], dtype=np.float64)
        )
        prices2 = dt.from_numpy(
            np.array([50.0, 51.0, 50.5, 52.5, 54.0], dtype=np.float64)
        )

        returns1 = prices1.returns()
        returns2 = prices2.returns()

        # Both should have same length
        assert returns1.size() == returns2.size()

        # Calculate correlation between returns
        corr = dt.correlation(returns1, returns2)
        corr_value = dt.to_numpy(corr)[0] if corr.size() == 1 else dt.to_numpy(corr)
        assert -1.0 <= corr_value <= 1.0

    def test_rolling_statistics_multiple(self):
        """Test rolling statistics on multiple tensors"""
        prices1 = dt.from_numpy(
            np.array([100.0, 102.0, 101.0, 105.0, 108.0, 110.0], dtype=np.float64)
        )
        prices2 = dt.from_numpy(
            np.array([50.0, 51.0, 50.5, 52.5, 54.0, 55.0], dtype=np.float64)
        )

        rolling_mean1 = prices1.rolling_mean(3)
        rolling_mean2 = prices2.rolling_mean(3)

        # Both should have same length
        assert rolling_mean1.size() == rolling_mean2.size()

        # Calculate correlation between rolling means
        corr = dt.correlation(rolling_mean1, rolling_mean2)
        corr_value = dt.to_numpy(corr)[0] if corr.size() == 1 else dt.to_numpy(corr)
        # Handle floating point precision issues
        assert -1.0 - 1e-10 <= corr_value <= 1.0 + 1e-10

    def test_portfolio_returns(self):
        """Test calculating portfolio returns from multiple assets"""
        asset1 = dt.from_numpy(
            np.array([100.0, 102.0, 101.0, 105.0, 108.0], dtype=np.float64)
        )
        asset2 = dt.from_numpy(
            np.array([50.0, 51.0, 50.5, 52.5, 54.0], dtype=np.float64)
        )

        returns1 = asset1.returns()
        returns2 = asset2.returns()

        # Equal-weighted portfolio
        portfolio_returns = (returns1 + returns2) / 2.0

        assert portfolio_returns.size() == returns1.size()
        assert portfolio_returns.size() == returns2.size()

    def test_correlation_matrix_style(self):
        """Test correlation calculations in matrix style"""
        # Create multiple asset returns
        np.random.seed(42)
        asset1 = dt.from_numpy(np.random.randn(100).astype(np.float64))
        asset2 = dt.from_numpy(np.random.randn(100).astype(np.float64))
        asset3 = dt.from_numpy(np.random.randn(100).astype(np.float64))

        # Calculate pairwise correlations
        corr12 = dt.correlation(asset1, asset2)
        corr13 = dt.correlation(asset1, asset3)
        corr23 = dt.correlation(asset2, asset3)

        corr12_val = (
            dt.to_numpy(corr12)[0] if corr12.size() == 1 else dt.to_numpy(corr12)
        )
        corr13_val = (
            dt.to_numpy(corr13)[0] if corr13.size() == 1 else dt.to_numpy(corr13)
        )
        corr23_val = (
            dt.to_numpy(corr23)[0] if corr23.size() == 1 else dt.to_numpy(corr23)
        )

        # All correlations should be in valid range
        assert -1.0 <= corr12_val <= 1.0
        assert -1.0 <= corr13_val <= 1.0
        assert -1.0 <= corr23_val <= 1.0


class TestMultipleTensorEdgeCases:
    """Test edge cases with multiple tensors"""

    def test_zero_tensors(self):
        """Test operations with zero tensors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        zero = dt.from_numpy(np.array([0.0, 0.0, 0.0], dtype=np.float64))

        result = a + zero
        np.testing.assert_array_almost_equal(dt.to_numpy(result), dt.to_numpy(a))

        result = a * zero
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_division_by_zero_error(self):
        """Test that division by zero raises error"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        zero = dt.from_numpy(np.array([0.0, 0.0, 0.0], dtype=np.float64))

        with pytest.raises(RuntimeError, match="Division by zero"):
            _ = a / zero

    def test_large_number_of_tensors(self):
        """Test operations with many tensors"""
        tensors = [
            dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
            for _ in range(10)
        ]

        # Sum all tensors
        result = tensors[0]
        for t in tensors[1:]:
            result = result + t

        expected = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(dt.to_numpy(result), expected)

    def test_different_sized_operations_error(self):
        """Test that operations with incompatible sizes raise errors"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0], dtype=np.float64))
        c = dt.from_numpy(np.array([6.0, 7.0, 8.0, 9.0], dtype=np.float64))

        with pytest.raises(RuntimeError, match="Shape mismatch"):
            _ = a + b

        with pytest.raises(RuntimeError, match="Shape mismatch"):
            _ = a + c

    def test_2d_shape_mismatch(self):
        """Test 2D tensor shape mismatch"""
        a = dt.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        b = dt.from_numpy(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        )

        with pytest.raises(RuntimeError, match="Shape mismatch"):
            _ = a + b


class TestMultipleTensorTypeConversions:
    """Test type conversions with multiple tensors"""

    def test_convert_multiple_to_numpy(self):
        """Test converting multiple tensors to numpy"""
        a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))

        np_a = a.to_numpy()
        np_b = b.to_numpy()

        result = np_a + np_b
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_multiple_to_torch(self):
        """Test converting multiple tensors to torch"""
        try:
            import torch

            a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
            b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))

            torch_a = a.to_torch()
            torch_b = b.to_torch()

            result = torch_a + torch_b
            expected = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float64)

            # Handle NumPy compatibility issues
            try:
                result_np = result.numpy()
                expected_np = expected.numpy()
            except RuntimeError:
                # Fallback to tolist() if numpy() fails
                result_np = np.array(result.tolist())
                expected_np = np.array(expected.tolist())

            np.testing.assert_array_almost_equal(result_np, expected_np)
        except ImportError:
            pytest.skip("torch not available")

    def test_convert_multiple_to_arrow(self):
        """Test converting multiple tensors to arrow"""
        try:
            import pyarrow as pa

            a = dt.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
            b = dt.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))

            arrow_a = a.to_arrow()
            arrow_b = b.to_arrow()

            # Verify both are Arrow arrays
            assert isinstance(arrow_a, pa.Array)
            assert isinstance(arrow_b, pa.Array)

            # Can create RecordBatch from multiple arrays
            batch = pa.RecordBatch.from_arrays([arrow_a, arrow_b], ["a", "b"])
            assert batch.num_columns == 2
            assert batch.num_rows == 3
        except ImportError:
            pytest.skip("pyarrow not available")
