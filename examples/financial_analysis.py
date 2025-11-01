"""
Financial analysis examples using Dragon Tensor
"""

import numpy as np
import dragon_tensor as dt


def example_portfolio_returns():
    """Calculate portfolio returns"""
    print("=== Portfolio Returns ===")

    # Simulate daily prices for multiple assets
    np.random.seed(42)
    days = 100

    # Generate correlated price series
    base_prices = np.cumsum(np.random.randn(days) * 0.02 + 0.001) + 100
    asset1 = dt.from_numpy(base_prices)
    asset2 = dt.from_numpy(base_prices * 0.5 + np.random.randn(days) * 0.1)

    # Calculate returns
    returns1 = asset1.returns()
    returns2 = asset2.returns()

    print(f"Asset 1 mean return: {returns1.mean():.4f}")
    print(f"Asset 1 std return: {returns1.std():.4f}")
    print(f"Asset 2 mean return: {returns2.mean():.4f}")
    print(f"Asset 2 std return: {returns2.std():.4f}")

    # Correlation
    corr = returns1.correlation(returns2)
    print(f"Correlation: {corr.to_numpy()[0]:.4f}\n")


def example_volatility_analysis():
    """Volatility analysis using rolling windows"""
    print("=== Volatility Analysis ===")

    # Simulate price series
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(252) * 0.015 + 0.0005) + 100
    tensor = dt.from_numpy(prices)

    # Calculate returns
    returns = tensor.returns()

    # Rolling volatility (annualized)
    rolling_vol = returns.rolling_std(window=20) * np.sqrt(252)  # Annualized

    print(f"Full period volatility: {returns.std() * np.sqrt(252):.4f}")
    print(f"Rolling volatility (last 5): {rolling_vol.to_numpy()[-5:]}")
    print()


def example_moving_averages():
    """Moving average crossover strategy signals"""
    print("=== Moving Averages ===")

    # Simulate price series
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100) * 0.02 + 0.001) + 100
    tensor = dt.from_numpy(prices)

    # Short and long moving averages
    short_ma = tensor.rolling_mean(window=5)
    long_ma = tensor.rolling_mean(window=20)

    # Generate signals (simplified)
    short_vals = short_ma.to_numpy()
    long_vals = long_ma.to_numpy()

    # Compare last values
    print(f"Short MA (5): {short_vals[-1]:.2f}")
    print(f"Long MA (20): {long_vals[-1]:.2f}")
    if short_vals[-1] > long_vals[-1]:
        print("Signal: Bullish (Short MA > Long MA)")
    else:
        print("Signal: Bearish (Short MA < Long MA)")
    print()


def example_risk_metrics():
    """Calculate risk metrics"""
    print("=== Risk Metrics ===")

    # Simulate portfolio returns
    np.random.seed(42)
    returns_data = np.random.randn(252) * 0.02
    returns = dt.from_numpy(returns_data)

    # Basic statistics
    mean_return = returns.mean()
    std_return = returns.std()

    # Rolling metrics
    rolling_std = returns.rolling_std(window=20)

    print(f"Mean Return: {mean_return:.4f}")
    print(f"Std Return: {std_return:.4f}")
    print(f"Sharpe Ratio (assuming 0% risk-free): {mean_return / std_return:.4f}")
    print(f"Current Rolling Std: {rolling_std.to_numpy()[-1]:.4f}")
    print()


def example_price_levels():
    """Support and resistance levels"""
    print("=== Price Levels ===")

    # Simulate price series with trends
    np.random.seed(42)
    trend = np.linspace(100, 110, 50)
    noise = np.random.randn(50) * 2
    prices = trend + noise
    tensor = dt.from_numpy(prices)

    # Rolling max and min (support/resistance)
    rolling_max = tensor.rolling_max(window=10)
    rolling_min = tensor.rolling_min(window=10)

    print(f"Current Price: {prices[-1]:.2f}")
    print(f"Recent High (10-day): {rolling_max.to_numpy()[-1]:.2f}")
    print(f"Recent Low (10-day): {rolling_min.to_numpy()[-1]:.2f}")
    print()


if __name__ == "__main__":
    example_portfolio_returns()
    example_volatility_analysis()
    example_moving_averages()
    example_risk_metrics()
    example_price_levels()
