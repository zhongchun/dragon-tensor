"""
Tests for financial operations
"""

import numpy as np
import pytest
import dragon_tensor as dt


class TestReturns:
    """Test returns calculation"""

    def test_returns_basic(self, price_tensor):
        """Test basic returns calculation"""
        returns = dt.returns(price_tensor)
        result = dt.to_numpy(returns)
        # Returns should have one less element than prices
        assert len(result) == 5
        # First return should be (102 - 100) / 100 = 0.02
        assert abs(result[0] - 0.02) < 1e-10

    def test_returns_values(self, price_tensor):
        """Test returns values are correct"""
        prices = np.array([100.0, 102.0, 101.0, 105.0, 108.0, 110.0], dtype=np.float64)
        returns = dt.returns(price_tensor)
        result = dt.to_numpy(returns)
        # Manual calculation
        expected = np.diff(prices) / prices[:-1]
        np.testing.assert_array_almost_equal(result, expected)

    def test_returns_single_price(self):
        """Test returns with single price (should raise error)"""
        prices = np.array([100.0], dtype=np.float64)
        tensor = dt.from_numpy(prices)
        with pytest.raises(RuntimeError, match="at least 2 elements"):
            dt.returns(tensor)


class TestRollingStatistics:
    """Test rolling statistics"""

    def test_rolling_mean(self, price_tensor):
        """Test rolling mean"""
        rolling_mean = dt.rolling_mean(price_tensor, 3)
        result = dt.to_numpy(rolling_mean)
        # Rolling mean with window 3 should have len(prices) - 2 elements
        assert len(result) == 4
        # First value should be mean of first 3 prices
        expected_first = (100.0 + 102.0 + 101.0) / 3.0
        assert abs(result[0] - expected_first) < 1e-10

    def test_rolling_std(self, price_tensor):
        """Test rolling standard deviation"""
        rolling_std = dt.rolling_std(price_tensor, 3)
        result = dt.to_numpy(rolling_std)
        assert len(result) == 4
        # Check that values are positive
        assert np.all(result >= 0)

    def test_rolling_mean_window_1(self, price_tensor):
        """Test rolling mean with window size 1"""
        rolling_mean = dt.rolling_mean(price_tensor, 1)
        result = dt.to_numpy(rolling_mean)
        # With window 1, should return original values
        prices = dt.to_numpy(price_tensor)
        np.testing.assert_array_almost_equal(result, prices)

    def test_rolling_std_window_1(self, price_tensor):
        """Test rolling std with window size 1"""
        rolling_std = dt.rolling_std(price_tensor, 1)
        result = dt.to_numpy(rolling_std)
        # With window 1, std should be 0
        np.testing.assert_array_almost_equal(result, np.zeros(len(result)))

    def test_rolling_mean_large_window(self):
        """Test rolling mean with window larger than data (should raise error)"""
        prices = np.array([100.0, 102.0, 101.0], dtype=np.float64)
        tensor = dt.from_numpy(prices)
        with pytest.raises(RuntimeError, match="Window size"):
            dt.rolling_mean(tensor, 5)


class TestCorrelationCovariance:
    """Test correlation and covariance"""

    def test_correlation(self):
        """Test correlation calculation"""
        # Create two correlated series
        np.random.seed(42)
        base = np.random.randn(100)
        series1 = dt.from_numpy(base.astype(np.float64))
        series2 = dt.from_numpy(
            (base * 0.5 + np.random.randn(100) * 0.1).astype(np.float64)
        )

        corr = dt.correlation(series1, series2)
        result = dt.to_numpy(corr)
        # Correlation should be a scalar or 1-element array
        assert len(result) == 1 or np.isscalar(result)
        corr_value = result[0] if len(result) == 1 else result
        # Should be positive correlation
        assert corr_value > 0
        # Should be less than 1
        assert corr_value <= 1.0

    def test_covariance(self):
        """Test covariance calculation"""
        # Create two series
        np.random.seed(42)
        series1 = dt.from_numpy(np.random.randn(100).astype(np.float64))
        series2 = dt.from_numpy(np.random.randn(100).astype(np.float64))

        cov = dt.covariance(series1, series2)
        result = dt.to_numpy(cov)
        # Covariance should be a scalar or 1-element array
        assert len(result) == 1 or np.isscalar(result)

    def test_correlation_identical(self):
        """Test correlation of identical series (should be 1.0)"""
        series = dt.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        corr = dt.correlation(series, series)
        result = dt.to_numpy(corr)
        corr_value = result[0] if len(result) == 1 else result
        assert abs(corr_value - 1.0) < 1e-10

    def test_correlation_opposite(self):
        """Test correlation of opposite series (should be -1.0)"""
        series1 = dt.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        series2 = dt.from_numpy(
            np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64)
        )
        corr = dt.correlation(series1, series2)
        result = dt.to_numpy(corr)
        corr_value = result[0] if len(result) == 1 else result
        assert abs(corr_value - (-1.0)) < 1e-10


class TestFinancialWorkflow:
    """Test complete financial workflows"""

    def test_returns_volatility_workflow(self, price_tensor):
        """Test calculating returns and volatility"""
        returns = dt.returns(price_tensor)
        rolling_vol = dt.rolling_std(returns, 3)
        vol_result = dt.to_numpy(rolling_vol)
        # Volatility should be positive
        assert np.all(vol_result >= 0)

    def test_moving_average_crossover(self):
        """Test moving average crossover strategy"""
        prices = np.cumsum(np.random.randn(100) * 0.02 + 0.001) + 100
        tensor = dt.from_numpy(prices.astype(np.float64))

        short_ma = dt.rolling_mean(tensor, 5)
        long_ma = dt.rolling_mean(tensor, 20)

        short_vals = dt.to_numpy(short_ma)
        long_vals = dt.to_numpy(long_ma)

        # Check that we can compare them
        assert len(short_vals) > 0
        assert len(long_vals) > 0
