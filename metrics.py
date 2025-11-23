"""
Metrics Module - Portfolio Performance Metrics

This module contains all functions for computing portfolio performance metrics
including returns, risk measures, Sharpe ratio, drawdown, and risk contributions.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
from scipy import stats

# ============================================================================
# BASIC PORTFOLIO METRICS
# ============================================================================

def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """
    Calculate expected portfolio return.

    Args:
        weights: Array of portfolio weights
        mean_returns: Array of mean returns for each asset

    Returns:
        Expected portfolio return (annualized)
    """
    return np.sum(weights * mean_returns)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio volatility (standard deviation).

    Args:
        weights: Array of portfolio weights
        cov_matrix: Covariance matrix of asset returns

    Returns:
        Portfolio volatility (annualized)
    """
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)


def sharpe_ratio(
    portfolio_ret: float,
    portfolio_vol: float,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        portfolio_ret: Portfolio return (annualized)
        portfolio_vol: Portfolio volatility (annualized)
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if portfolio_vol == 0:
        return 0
    return (portfolio_ret - risk_free_rate) / portfolio_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (similar to Sharpe but only considers downside deviation).

    Args:
        returns: Series of portfolio returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year (252 for daily)

    Returns:
        Sortino ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf

    downside_std = np.sqrt(np.mean(downside_returns ** 2))

    if downside_std == 0:
        return np.inf

    annualized_return = np.mean(returns) * periods_per_year
    annualized_downside = downside_std * np.sqrt(periods_per_year)

    return (annualized_return - risk_free_rate) / annualized_downside


def maximum_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        returns: Series of returns

    Returns:
        Tuple of (max_drawdown, start_date, end_date)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()
    end_date = drawdown.idxmin()

    # Find start of drawdown (the peak before the trough)
    start_date = cumulative[:end_date].idxmax()

    return max_dd, start_date, end_date


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).

    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year

    Returns:
        Calmar ratio
    """
    annual_return = returns.mean() * periods_per_year
    max_dd, _, _ = maximum_drawdown(returns)

    if max_dd == 0:
        return np.inf

    return annual_return / abs(max_dd)


def value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.

    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR value (negative number indicating potential loss)
    """
    return np.percentile(returns, (1 - confidence_level) * 100)


def conditional_value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)

    Returns:
        CVaR value (expected loss beyond VaR)
    """
    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate Information Ratio (active return / tracking error).

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Information ratio
    """
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()

    if tracking_error == 0:
        return 0

    return active_returns.mean() / tracking_error


def beta_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Tuple[float, float]:
    """
    Calculate portfolio beta and alpha relative to benchmark.

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year

    Returns:
        Tuple of (beta, alpha)
    """
    # Align the series
    aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
    port_ret = aligned_returns.iloc[:, 0]
    bench_ret = aligned_returns.iloc[:, 1]

    # Calculate beta
    covariance = np.cov(port_ret, bench_ret)[0, 1]
    benchmark_variance = np.var(bench_ret)

    if benchmark_variance == 0:
        beta = 0
    else:
        beta = covariance / benchmark_variance

    # Calculate alpha
    rf_period = risk_free_rate / periods_per_year
    alpha_period = port_ret.mean() - (rf_period + beta * (bench_ret.mean() - rf_period))
    alpha = alpha_period * periods_per_year

    return beta, alpha


def marginal_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate marginal contribution to risk for each asset.

    Args:
        weights: Array of portfolio weights
        cov_matrix: Covariance matrix

    Returns:
        Array of marginal risk contributions
    """
    portfolio_vol = portfolio_volatility(weights, cov_matrix)

    if portfolio_vol == 0:
        return np.zeros(len(weights))

    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
    return marginal_contrib


def risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate total risk contribution for each asset.

    Args:
        weights: Array of portfolio weights
        cov_matrix: Covariance matrix

    Returns:
        Array of risk contributions (sum to portfolio variance)
    """
    marginal_contrib = marginal_risk_contribution(weights, cov_matrix)
    risk_contrib = weights * marginal_contrib
    return risk_contrib


def risk_contribution_pct(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate percentage risk contribution for each asset.

    Args:
        weights: Array of portfolio weights
        cov_matrix: Covariance matrix

    Returns:
        Array of percentage risk contributions (sum to 100%)
    """
    total_risk = risk_contribution(weights, cov_matrix)
    return 100 * total_risk / total_risk.sum()


def downside_deviation(
    returns: pd.Series,
    target_return: float = 0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate downside deviation (semi-deviation).

    Args:
        returns: Series of returns
        target_return: Target/minimum acceptable return
        periods_per_year: Number of periods in a year

    Returns:
        Annualized downside deviation
    """
    downside_returns = returns[returns < target_return]
    downside_diff = downside_returns - target_return
    downside_var = np.mean(downside_diff ** 2)
    downside_std = np.sqrt(downside_var)

    return downside_std * np.sqrt(periods_per_year)


def tail_ratio(returns: pd.Series, percentile: float = 5) -> float:
    """
    Calculate tail ratio (right tail / left tail).

    Args:
        returns: Series of returns
        percentile: Percentile for tail calculation (e.g., 5 for 95th/5th)

    Returns:
        Tail ratio (higher is better)
    """
    right_tail = np.abs(np.percentile(returns, 100 - percentile))
    left_tail = np.abs(np.percentile(returns, percentile))

    if left_tail == 0:
        return np.inf

    return right_tail / left_tail


def omega_ratio(
    returns: pd.Series,
    target_return: float = 0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Omega ratio (probability-weighted gains / losses).

    Args:
        returns: Series of returns
        target_return: Target return threshold (annualized)
        periods_per_year: Number of periods in a year

    Returns:
        Omega ratio
    """
    target_daily = target_return / periods_per_year
    excess = returns - target_daily

    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()

    if losses == 0:
        return np.inf

    return gains / losses


def annualize_returns(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Annualize returns from a series.

    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year

    Returns:
        Annualized return
    """
    return returns.mean() * periods_per_year


def annualize_volatility(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Annualize volatility from a series.

    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year

    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def portfolio_statistics(
    weights: np.ndarray,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate comprehensive portfolio statistics.

    Args:
        weights: Portfolio weights
        returns: DataFrame of asset returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year

    Returns:
        Dictionary of portfolio statistics
    """
    # Calculate portfolio returns
    port_returns = (returns * weights).sum(axis=1)

    # Basic metrics
    mean_returns = returns.mean() * periods_per_year
    cov_matrix = returns.cov() * periods_per_year

    port_return = portfolio_return(weights, mean_returns.values)
    port_vol = portfolio_volatility(weights, cov_matrix.values)
    sharpe = sharpe_ratio(port_return, port_vol, risk_free_rate)

    # Risk metrics
    max_dd, dd_start, dd_end = maximum_drawdown(port_returns)
    var_95 = value_at_risk(port_returns, 0.95)
    cvar_95 = conditional_value_at_risk(port_returns, 0.95)

    # Risk contributions
    risk_contrib = risk_contribution_pct(weights, cov_matrix.values)

    portfolio_stats = {
        'Expected Annual Return': port_return,
        'Annual Volatility': port_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino_ratio(port_returns, risk_free_rate, periods_per_year),
        'Calmar Ratio': calmar_ratio(port_returns, periods_per_year),
        'Maximum Drawdown': max_dd,
        'Drawdown Start': dd_start,
        'Drawdown End': dd_end,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Skewness': stats.skew(port_returns),
        'Kurtosis': stats.kurtosis(port_returns),
        'Risk Contributions': dict(zip(returns.columns, risk_contrib))
    }

    return portfolio_stats
