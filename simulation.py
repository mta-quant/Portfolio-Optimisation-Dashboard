"""
Simulation Module - Monte Carlo and Scenario Analysis

This module provides Monte Carlo simulation capabilities for portfolio analysis,
including price simulations, scenario analysis, and stress testing.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy import stats

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_portfolio_simulation(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    initial_value: float = 10000,
    num_simulations: int = 1000,
    num_days: int = 252,
    periods_per_year: int = 252
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Monte Carlo simulation of portfolio value evolution.

    Args:
        weights: Portfolio weights
        mean_returns: Mean returns (annualized)
        cov_matrix: Covariance matrix (annualized)
        initial_value: Initial portfolio value
        num_simulations: Number of simulation paths
        num_days: Number of days to simulate
        periods_per_year: Periods per year for scaling

    Returns:
        Tuple of (final_values, simulation_paths_df)
    """
    # Convert annual parameters to daily
    daily_returns = mean_returns / periods_per_year
    daily_cov = cov_matrix / periods_per_year

    # Check for and handle assets with zero variance (like CASH)
    variances = np.diag(daily_cov)
    zero_var_mask = variances < 1e-10

    if np.any(zero_var_mask):
        # Separate zero-variance and risky assets
        risky_mask = ~zero_var_mask
        risky_indices = np.where(risky_mask)[0]
        zero_var_indices = np.where(zero_var_mask)[0]

        # Extract risky asset covariance matrix
        risky_cov = daily_cov[np.ix_(risky_indices, risky_indices)]
        risky_returns = daily_returns[risky_indices]
        risky_weights = weights[risky_indices]

        # Zero-variance asset returns (deterministic)
        zero_var_returns = daily_returns[zero_var_indices]
        zero_var_weights = weights[zero_var_indices]

        # Generate correlated returns for risky assets only
        try:
            L = np.linalg.cholesky(risky_cov)
        except np.linalg.LinAlgError:
            # If still fails, add small regularization
            epsilon = 1e-8
            risky_cov_reg = risky_cov + epsilon * np.eye(len(risky_cov))
            L = np.linalg.cholesky(risky_cov_reg)
    else:
        # All assets are risky, proceed normally
        try:
            L = np.linalg.cholesky(daily_cov)
        except np.linalg.LinAlgError:
            # Add small regularization to ensure positive definiteness
            epsilon = 1e-8
            daily_cov_reg = daily_cov + epsilon * np.eye(len(daily_cov))
            L = np.linalg.cholesky(daily_cov_reg)

        risky_mask = np.ones(len(weights), dtype=bool)
        risky_indices = np.arange(len(weights))
        zero_var_mask = np.zeros(len(weights), dtype=bool)
        zero_var_indices = np.array([], dtype=int)
        risky_returns = daily_returns
        risky_weights = weights
        zero_var_returns = np.array([])
        zero_var_weights = np.array([])

    # Storage for simulation results
    all_paths = np.zeros((num_simulations, num_days + 1))
    all_paths[:, 0] = initial_value

    for sim in range(num_simulations):
        portfolio_values = [initial_value]

        for day in range(num_days):
            if np.any(zero_var_mask):
                # Generate correlated random returns for risky assets only
                random_returns = np.random.normal(0, 1, len(risky_indices))
                risky_correlated_returns = risky_returns + L @ random_returns

                # Calculate contribution from risky assets
                risky_contribution = np.dot(risky_weights, risky_correlated_returns)

                # Add deterministic contribution from zero-variance assets
                zero_var_contribution = np.dot(zero_var_weights, zero_var_returns)

                # Total portfolio return
                portfolio_return = risky_contribution + zero_var_contribution
            else:
                # All assets are risky
                random_returns = np.random.normal(0, 1, len(weights))
                correlated_returns = daily_returns + L @ random_returns
                portfolio_return = np.dot(weights, correlated_returns)

            # Update portfolio value
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)

        all_paths[sim, :] = portfolio_values

    final_values = all_paths[:, -1]

    # Create DataFrame with sample paths for visualization
    sample_indices = np.random.choice(num_simulations, min(100, num_simulations), replace=False)
    sample_paths = all_paths[sample_indices, :]

    paths_df = pd.DataFrame(
        sample_paths.T,
        columns=[f'Path_{i}' for i in range(len(sample_indices))]
    )

    return final_values, paths_df


def var_cvar_simulation(
    final_values: np.ndarray,
    initial_value: float,
    confidence_levels: List[float] = [0.95, 0.99]
) -> dict:
    """
    Calculate VaR and CVaR from Monte Carlo simulation results.

    Args:
        final_values: Array of final portfolio values from simulation
        initial_value: Initial portfolio value
        confidence_levels: List of confidence levels for VaR/CVaR

    Returns:
        Dictionary with VaR and CVaR values
    """
    returns = (final_values - initial_value) / initial_value

    results = {}

    for cl in confidence_levels:
        var = np.percentile(returns, (1 - cl) * 100)
        cvar = returns[returns <= var].mean()

        results[f'VaR_{int(cl*100)}'] = var
        results[f'CVaR_{int(cl*100)}'] = cvar

    return results


def scenario_analysis(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    price_shocks: List[float],
    volatility_shocks: List[float],
    correlation_shocks: List[float] = [1.0]
) -> pd.DataFrame:
    """
    Perform scenario analysis with various market shocks.

    Args:
        weights: Portfolio weights
        mean_returns: Mean returns
        cov_matrix: Covariance matrix
        price_shocks: List of price shock multipliers (e.g., [0.9, 1.0, 1.1])
        volatility_shocks: List of volatility shock multipliers
        correlation_shocks: List of correlation shock multipliers

    Returns:
        DataFrame with scenario analysis results
    """
    from metrics import portfolio_return, portfolio_volatility

    scenarios = []

    for price_shock in price_shocks:
        for vol_shock in volatility_shocks:
            for corr_shock in correlation_shocks:
                # Adjust returns and covariance
                shocked_returns = mean_returns * price_shock

                # Adjust covariance matrix
                # Extract correlation matrix
                stds = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(stds, stds)

                # Shock correlations towards 1 (higher systemic risk)
                if corr_shock != 1.0:
                    identity = np.eye(len(corr_matrix))
                    shocked_corr = corr_shock * corr_matrix + (1 - corr_shock) * identity
                    np.fill_diagonal(shocked_corr, 1.0)
                else:
                    shocked_corr = corr_matrix

                # Reconstruct covariance with shocked volatilities
                shocked_stds = stds * vol_shock
                shocked_cov = np.outer(shocked_stds, shocked_stds) * shocked_corr

                # Calculate portfolio metrics under shock
                port_ret = portfolio_return(weights, shocked_returns)
                port_vol = portfolio_volatility(weights, shocked_cov)

                scenarios.append({
                    'Price Shock': f'{(price_shock - 1) * 100:.1f}%',
                    'Volatility Shock': f'{(vol_shock - 1) * 100:.1f}%',
                    'Correlation Shock': f'{(corr_shock - 1) * 100:.1f}%',
                    'Expected Return': port_ret,
                    'Volatility': port_vol,
                    'Sharpe': port_ret / port_vol if port_vol > 0 else 0
                })

    return pd.DataFrame(scenarios)


def stress_test(
    weights: np.ndarray,
    returns: pd.DataFrame,
    historical_scenarios: dict
) -> pd.DataFrame:
    """
    Stress test portfolio using historical crisis scenarios.

    Args:
        weights: Portfolio weights
        returns: DataFrame of asset returns
        historical_scenarios: Dict of scenario names and date ranges

    Returns:
        DataFrame with stress test results
    """
    results = []

    for scenario_name, (start_date, end_date) in historical_scenarios.items():
        # Get returns during crisis period
        crisis_returns = returns.loc[start_date:end_date]

        if crisis_returns.empty:
            continue

        # Calculate portfolio returns during crisis
        portfolio_returns = (crisis_returns * weights).sum(axis=1)

        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        avg_daily_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        max_drawdown = (portfolio_returns.cumsum().cummax() - portfolio_returns.cumsum()).max()

        results.append({
            'Scenario': scenario_name,
            'Start Date': start_date,
            'End Date': end_date,
            'Total Return': total_return,
            'Avg Daily Return': avg_daily_return,
            'Volatility': volatility,
            'Max Drawdown': max_drawdown
        })

    return pd.DataFrame(results)


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    num_paths: int = 1000
) -> np.ndarray:
    """
    Simulate asset prices using Geometric Brownian Motion.

    Args:
        S0: Initial price
        mu: Drift (expected return)
        sigma: Volatility
        T: Time horizon (years)
        N: Number of time steps
        num_paths: Number of simulation paths

    Returns:
        Array of simulated price paths (num_paths x N+1)
    """
    dt = T / N
    paths = np.zeros((num_paths, N + 1))
    paths[:, 0] = S0

    for t in range(1, N + 1):
        z = np.random.standard_normal(num_paths)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return paths


def bootstrap_simulation(
    returns: pd.DataFrame,
    weights: np.ndarray,
    num_simulations: int = 1000,
    block_size: int = 20,
    num_days: int = 252
) -> np.ndarray:
    """
    Bootstrap simulation using historical returns.

    Args:
        returns: Historical returns DataFrame
        weights: Portfolio weights
        num_simulations: Number of bootstrap samples
        block_size: Size of blocks for block bootstrap (preserves autocorrelation)
        num_days: Number of days to simulate

    Returns:
        Array of simulated portfolio returns
    """
    n_returns = len(returns)
    simulated_returns = []

    for _ in range(num_simulations):
        # Block bootstrap to preserve time-series structure
        sampled_returns = []

        while len(sampled_returns) < num_days:
            start_idx = np.random.randint(0, n_returns - block_size)
            block = returns.iloc[start_idx:start_idx + block_size]
            sampled_returns.extend(block.values.tolist())

        sampled_returns = sampled_returns[:num_days]
        sampled_df = pd.DataFrame(sampled_returns, columns=returns.columns)

        # Calculate portfolio return
        portfolio_return = (sampled_df * weights).sum(axis=1).sum()
        simulated_returns.append(portfolio_return)

    return np.array(simulated_returns)


def regime_switching_simulation(
    weights: np.ndarray,
    bull_params: Tuple[np.ndarray, np.ndarray],
    bear_params: Tuple[np.ndarray, np.ndarray],
    transition_probs: np.ndarray,
    initial_regime: int = 0,
    num_days: int = 252,
    num_simulations: int = 1000,
    initial_value: float = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation with regime switching.

    Args:
        weights: Portfolio weights
        bull_params: (mean_returns, cov_matrix) for bull market
        bear_params: (mean_returns, cov_matrix) for bear market
        transition_probs: 2x2 transition probability matrix [[P(bull|bull), P(bear|bull)],
                                                              [P(bull|bear), P(bear|bear)]]
        initial_regime: Starting regime (0=bull, 1=bear)
        num_days: Number of days to simulate
        num_simulations: Number of simulation paths
        initial_value: Initial portfolio value

    Returns:
        Tuple of (final_values, regimes_history)
    """
    final_values = []
    regimes_history = []

    bull_mean, bull_cov = bull_params
    bear_mean, bear_cov = bear_params

    for _ in range(num_simulations):
        portfolio_value = initial_value
        regime = initial_regime
        regimes = [regime]

        for day in range(num_days):
            # Select parameters based on current regime
            if regime == 0:  # Bull market
                mean_ret = bull_mean / 252
                cov = bull_cov / 252
            else:  # Bear market
                mean_ret = bear_mean / 252
                cov = bear_cov / 252

            # Generate returns
            asset_returns = np.random.multivariate_normal(mean_ret, cov)
            portfolio_return = np.dot(weights, asset_returns)

            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)

            # Regime switching
            if np.random.random() < transition_probs[regime, 1 - regime]:
                regime = 1 - regime

            regimes.append(regime)

        final_values.append(portfolio_value)
        regimes_history.append(regimes)

    return np.array(final_values), np.array(regimes_history)


def calculate_simulation_statistics(
    final_values: np.ndarray,
    initial_value: float
) -> dict:
    """
    Calculate statistics from Monte Carlo simulation results.

    Args:
        final_values: Array of final portfolio values
        initial_value: Initial portfolio value

    Returns:
        Dictionary of statistics
    """
    returns = (final_values - initial_value) / initial_value

    stats_dict = {
        'Mean Final Value': final_values.mean(),
        'Median Final Value': np.median(final_values),
        'Std Final Value': final_values.std(),
        'Min Final Value': final_values.min(),
        'Max Final Value': final_values.max(),
        'Mean Return': returns.mean(),
        'Median Return': np.median(returns),
        'Std Return': returns.std(),
        'Probability of Loss': (returns < 0).sum() / len(returns),
        'Probability of Gain > 10%': (returns > 0.10).sum() / len(returns),
        'Skewness': stats.skew(returns),
        'Kurtosis': stats.kurtosis(returns)
    }

    return stats_dict


def tail_risk_analysis(
    returns: np.ndarray,
    confidence_levels: List[float] = [0.90, 0.95, 0.99]
) -> pd.DataFrame:
    """
    Analyze tail risk characteristics.

    Args:
        returns: Array of returns
        confidence_levels: List of confidence levels

    Returns:
        DataFrame with tail risk metrics
    """
    results = []

    for cl in confidence_levels:
        var = np.percentile(returns, (1 - cl) * 100)
        cvar = returns[returns <= var].mean()

        tail_returns = returns[returns <= var]

        results.append({
            'Confidence Level': f'{cl*100:.0f}%',
            'VaR': var,
            'CVaR': cvar,
            'Tail Mean': tail_returns.mean(),
            'Tail Std': tail_returns.std(),
            'Worst Return': returns.min()
        })

    return pd.DataFrame(results)
