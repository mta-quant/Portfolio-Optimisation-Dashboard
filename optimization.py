"""
Optimization Module - Portfolio Optimization Methods

This module implements various portfolio optimization techniques including
mean-variance optimization, risk parity, maximum Sharpe ratio, and efficient frontier.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
from typing import Tuple, Optional, List
from metrics import portfolio_return, portfolio_volatility, sharpe_ratio

# ============================================================================
# OPTIMIZATION ALGORITHMS
# ============================================================================

def mean_variance_optimization(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: Optional[float] = None,
    allow_short: bool = False,
    weight_bounds: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, dict]:
    """
    Perform mean-variance (Markowitz) optimization.

    Args:
        mean_returns: Array of expected returns
        cov_matrix: Covariance matrix
        target_return: Target return (if None, minimize risk for any return)
        allow_short: Allow short positions
        weight_bounds: Min/max weight per asset

    Returns:
        Tuple of (optimal_weights, optimization_info)
    """
    n_assets = len(mean_returns)

    # Define optimization variables
    w = cp.Variable(n_assets)

    # Objective: minimize portfolio variance
    portfolio_variance = cp.quad_form(w, cov_matrix)
    objective = cp.Minimize(portfolio_variance)

    # Constraints
    constraints = [cp.sum(w) == 1]  # Weights sum to 1

    if not allow_short:
        constraints.append(w >= weight_bounds[0])

    constraints.append(w <= weight_bounds[1])

    if target_return is not None:
        constraints.append(mean_returns @ w >= target_return)

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != 'optimal':
        return np.array([1/n_assets] * n_assets), {'status': 'failed'}

    weights = w.value
    weights = np.array([max(0, w) if not allow_short else w for w in weights])
    weights = weights / weights.sum()  # Normalize

    info = {
        'status': 'success',
        'return': portfolio_return(weights, mean_returns),
        'volatility': portfolio_volatility(weights, cov_matrix),
        'objective_value': problem.value
    }

    return weights, info


def maximum_sharpe_ratio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    allow_short: bool = False,
    weight_bounds: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, dict]:
    """
    Optimize portfolio for maximum Sharpe ratio.

    Args:
        mean_returns: Array of expected returns
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate
        allow_short: Allow short positions
        weight_bounds: Min/max weight per asset

    Returns:
        Tuple of (optimal_weights, optimization_info)
    """
    n_assets = len(mean_returns)

    def neg_sharpe(weights):
        ret = portfolio_return(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        return -sharpe_ratio(ret, vol, risk_free_rate)

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
    ]

    # Bounds
    if allow_short:
        bounds = tuple(weight_bounds for _ in range(n_assets))
    else:
        bounds = tuple((weight_bounds[0], weight_bounds[1]) for _ in range(n_assets))

    # Initial guess
    init_guess = np.array([1/n_assets] * n_assets)

    # Optimize
    result = minimize(
        neg_sharpe,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if not result.success:
        return np.array([1/n_assets] * n_assets), {'status': 'failed'}

    weights = result.x
    weights = weights / weights.sum()  # Normalize

    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)

    info = {
        'status': 'success',
        'return': ret,
        'volatility': vol,
        'sharpe_ratio': sharpe_ratio(ret, vol, risk_free_rate)
    }

    return weights, info


def minimum_volatility(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    allow_short: bool = False,
    weight_bounds: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, dict]:
    """
    Find minimum volatility portfolio.

    Args:
        mean_returns: Array of expected returns
        cov_matrix: Covariance matrix
        allow_short: Allow short positions
        weight_bounds: Min/max weight per asset

    Returns:
        Tuple of (optimal_weights, optimization_info)
    """
    n_assets = len(mean_returns)

    def portfolio_vol(weights):
        return portfolio_volatility(weights, cov_matrix)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    if allow_short:
        bounds = tuple(weight_bounds for _ in range(n_assets))
    else:
        bounds = tuple((weight_bounds[0], weight_bounds[1]) for _ in range(n_assets))

    init_guess = np.array([1/n_assets] * n_assets)

    result = minimize(
        portfolio_vol,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return np.array([1/n_assets] * n_assets), {'status': 'failed'}

    weights = result.x
    weights = weights / weights.sum()

    info = {
        'status': 'success',
        'return': portfolio_return(weights, mean_returns),
        'volatility': portfolio_volatility(weights, cov_matrix)
    }

    return weights, info


def risk_parity(
    cov_matrix: np.ndarray,
    weight_bounds: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, dict]:
    """
    Compute risk parity (equal risk contribution) portfolio.

    Args:
        cov_matrix: Covariance matrix
        weight_bounds: Min/max weight per asset

    Returns:
        Tuple of (optimal_weights, optimization_info)
    """
    n_assets = cov_matrix.shape[0]

    def risk_budget_objective(weights):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib

        # Objective: minimize difference in risk contributions
        target_risk = np.ones(n_assets) / n_assets
        return np.sum((risk_contrib / risk_contrib.sum() - target_risk) ** 2)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    bounds = tuple((weight_bounds[0], weight_bounds[1]) for _ in range(n_assets))

    # Start with equal weights
    init_guess = np.array([1/n_assets] * n_assets)

    result = minimize(
        risk_budget_objective,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if not result.success:
        return np.array([1/n_assets] * n_assets), {'status': 'failed'}

    weights = result.x
    weights = weights / weights.sum()

    info = {
        'status': 'success',
        'volatility': portfolio_volatility(weights, cov_matrix)
    }

    return weights, info


def efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_portfolios: int = 100,
    allow_short: bool = False,
    weight_bounds: Tuple[float, float] = (0, 1),
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Generate efficient frontier portfolios.

    Args:
        mean_returns: Array of expected returns
        cov_matrix: Covariance matrix
        num_portfolios: Number of portfolios to generate
        allow_short: Allow short positions
        weight_bounds: Min/max weight per asset
        risk_free_rate: Risk-free rate

    Returns:
        DataFrame with portfolio returns, volatilities, and Sharpe ratios
    """
    # Find the actual feasible return range by optimizing
    # 1. Find minimum volatility portfolio (lower bound)
    min_vol_weights, min_vol_info = minimum_volatility(
        mean_returns, cov_matrix, allow_short, weight_bounds
    )
    min_feasible_ret = min_vol_info['return'] if min_vol_info['status'] == 'success' else mean_returns.min()

    # 2. Find maximum return portfolio (upper bound)
    max_ret_weights, max_ret_info = maximum_return_portfolio(
        mean_returns, cov_matrix, allow_short, weight_bounds
    )
    max_feasible_ret = max_ret_info['return'] if max_ret_info['status'] == 'success' else mean_returns.max()

    # Use a range slightly beyond the feasible bounds to ensure coverage
    min_ret = min_feasible_ret * 0.95
    max_ret = max_feasible_ret * 1.05

    target_returns = np.linspace(min_ret, max_ret, num_portfolios)

    efficient_portfolios = []

    for target in target_returns:
        weights, info = mean_variance_optimization(
            mean_returns,
            cov_matrix,
            target_return=target,
            allow_short=allow_short,
            weight_bounds=weight_bounds
        )

        if info['status'] == 'success':
            ret = info['return']
            vol = info['volatility']
            sharpe = sharpe_ratio(ret, vol, risk_free_rate)

            efficient_portfolios.append({
                'Return': ret,
                'Volatility': vol,
                'Sharpe': sharpe,
                'Weights': weights
            })

    return pd.DataFrame(efficient_portfolios)


def maximum_return_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    allow_short: bool = False,
    weight_bounds: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, dict]:
    """
    Find the portfolio with maximum return subject to constraints.

    Args:
        mean_returns: Array of expected returns
        cov_matrix: Covariance matrix
        allow_short: Allow short positions
        weight_bounds: Min/max weight per asset

    Returns:
        Tuple of (optimal_weights, optimization_info)
    """
    n_assets = len(mean_returns)

    # Define optimization variables
    w = cp.Variable(n_assets)

    # Objective: maximize return (minimize negative return)
    portfolio_return = w @ mean_returns
    objective = cp.Maximize(portfolio_return)

    # Constraints
    constraints = [cp.sum(w) == 1]

    if not allow_short:
        constraints.append(w >= weight_bounds[0])

    constraints.append(w <= weight_bounds[1])

    # Solve
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve()

        if problem.status in ['optimal', 'optimal_inaccurate']:
            weights = w.value
            ret = portfolio_return.value
            vol = portfolio_volatility(weights, cov_matrix)

            return weights, {
                'status': 'success',
                'return': ret,
                'volatility': vol
            }
        else:
            return np.array([]), {
                'status': 'failed',
                'message': f'Optimization status: {problem.status}',
                'return': 0,
                'volatility': 0
            }
    except Exception as e:
        return np.array([]), {
            'status': 'failed',
            'message': str(e),
            'return': 0,
            'volatility': 0
        }


def random_portfolios(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_portfolios: int = 10000,
    risk_free_rate: float = 0.02,
    allow_short: bool = False,
    weight_bounds: Tuple[float, float] = (0, 1)
) -> pd.DataFrame:
    """
    Generate random portfolios for visualization.

    Args:
        mean_returns: Array of expected returns
        cov_matrix: Covariance matrix
        num_portfolios: Number of random portfolios to generate
        risk_free_rate: Risk-free rate
        allow_short: Allow short positions
        weight_bounds: Min/max weight per asset (same as optimization constraints)

    Returns:
        DataFrame with portfolio returns, volatilities, and Sharpe ratios
    """
    n_assets = len(mean_returns)
    results = []
    min_weight, max_weight = weight_bounds

    # Generate more portfolios to compensate for rejections
    attempts = 0
    max_attempts = num_portfolios * 100  # Prevent infinite loop

    while len(results) < num_portfolios and attempts < max_attempts:
        attempts += 1

        if allow_short:
            weights = np.random.uniform(min_weight, max_weight, n_assets)
        else:
            weights = np.random.uniform(max(0, min_weight), max_weight, n_assets)

        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            continue

        # Check if weights respect constraints after normalization
        if np.any(weights < min_weight - 1e-6) or np.any(weights > max_weight + 1e-6):
            continue  # Skip this portfolio if it violates constraints

        ret = portfolio_return(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        sharpe = sharpe_ratio(ret, vol, risk_free_rate)

        results.append({
            'Return': ret,
            'Volatility': vol,
            'Sharpe': sharpe
        })

    return pd.DataFrame(results)


def black_litterman(
    market_caps: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 2.5,
    views: Optional[pd.DataFrame] = None,
    view_confidences: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Black-Litterman model to incorporate market equilibrium and investor views.

    Args:
        market_caps: Market capitalizations (proxy for market weights)
        cov_matrix: Covariance matrix
        risk_aversion: Market risk aversion parameter
        views: DataFrame with 'Asset' and 'ExpectedReturn' columns
        view_confidences: Confidence in each view (uncertainty)

    Returns:
        Tuple of (posterior_returns, posterior_covariance)
    """
    # Market equilibrium weights
    market_weights = market_caps / market_caps.sum()

    # Implied equilibrium returns (reverse optimization)
    pi = risk_aversion * np.dot(cov_matrix, market_weights)

    if views is None:
        return pi, cov_matrix

    # Process views (simplified implementation)
    # In practice, would need proper P matrix and Q vector
    tau = 0.025  # Scalar representing uncertainty in prior

    # Posterior estimates (simplified)
    posterior_returns = pi  # Would incorporate views with proper math
    posterior_cov = cov_matrix  # Would adjust with tau

    return posterior_returns, posterior_cov


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    method: str = 'single'
) -> np.ndarray:
    """
    Hierarchical Risk Parity (HRP) portfolio optimization.

    Args:
        returns: DataFrame of asset returns
        method: Linkage method for clustering ('single', 'complete', 'average')

    Returns:
        Array of optimal weights
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    # Compute correlation matrix
    corr_matrix = returns.corr()

    # Convert to distance matrix
    distances = np.sqrt(0.5 * (1 - corr_matrix))

    # Hierarchical clustering
    dist_condensed = squareform(distances)
    links = linkage(dist_condensed, method=method)

    # Get quasi-diagonal matrix through clustering
    # Simplified implementation - full HRP requires recursive bisection
    n_assets = len(returns.columns)
    weights = np.ones(n_assets) / n_assets

    # Calculate inverse-variance weights within clusters
    cov_matrix = returns.cov()
    inv_var = 1 / np.diag(cov_matrix)
    weights = inv_var / inv_var.sum()

    return weights


def maximum_diversification(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    weight_bounds: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, dict]:
    """
    Maximum diversification portfolio optimization.

    Args:
        mean_returns: Array of expected returns
        cov_matrix: Covariance matrix
        weight_bounds: Min/max weight per asset

    Returns:
        Tuple of (optimal_weights, optimization_info)
    """
    n_assets = len(mean_returns)
    stds = np.sqrt(np.diag(cov_matrix))

    def diversification_ratio(weights):
        weighted_std = np.dot(weights, stds)
        portfolio_std = portfolio_volatility(weights, cov_matrix)
        return -weighted_std / portfolio_std  # Negative for minimization

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    bounds = tuple((weight_bounds[0], weight_bounds[1]) for _ in range(n_assets))

    init_guess = np.array([1/n_assets] * n_assets)

    result = minimize(
        diversification_ratio,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return np.array([1/n_assets] * n_assets), {'status': 'failed'}

    weights = result.x
    weights = weights / weights.sum()

    info = {
        'status': 'success',
        'return': portfolio_return(weights, mean_returns),
        'volatility': portfolio_volatility(weights, cov_matrix),
        'diversification_ratio': -result.fun
    }

    return weights, info
