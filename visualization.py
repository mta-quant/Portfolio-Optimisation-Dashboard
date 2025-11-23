"""
Visualization Module - Interactive Charts and Plots

This module provides all visualization functions for the portfolio optimization dashboard,
including interactive Plotly charts, correlation heatmaps, and network graphs.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Optional, Tuple

# ============================================================================
# PORTFOLIO VISUALIZATION FUNCTIONS
# ============================================================================

def plot_efficient_frontier(
    efficient_portfolios: pd.DataFrame,
    random_portfolios: Optional[pd.DataFrame] = None,
    highlight_portfolios: Optional[dict] = None,
    current_portfolio: Optional[Tuple[float, float]] = None,
    risk_free_rate: float = 0.02
) -> go.Figure:
    """
    Plot efficient frontier with optional random portfolios and highlights.

    Args:
        efficient_portfolios: DataFrame with 'Return', 'Volatility', 'Sharpe'
        random_portfolios: Optional DataFrame with random portfolio simulations
        highlight_portfolios: Dict of portfolio names and (return, vol, sharpe) tuples
        current_portfolio: Tuple of (return, volatility) for current portfolio
        risk_free_rate: Risk-free rate for Capital Allocation Line

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Plot random portfolios if provided
    if random_portfolios is not None and not random_portfolios.empty:
        fig.add_trace(go.Scatter(
            x=random_portfolios['Volatility'],
            y=random_portfolios['Return'],
            mode='markers',
            marker=dict(
                size=4,
                color=random_portfolios['Sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe<br>Ratio'),
                opacity=0.6
            ),
            name='Random Portfolios',
            hovertemplate='<b>Random Portfolio</b><br>' +
                          'Return: %{y:.2%}<br>' +
                          'Volatility: %{x:.2%}<br>' +
                          '<extra></extra>'
        ))

    # Plot efficient frontier (BLACK line)
    fig.add_trace(go.Scatter(
        x=efficient_portfolios['Volatility'],
        y=efficient_portfolios['Return'],
        mode='lines+markers',
        marker=dict(size=6, color='black'),
        line=dict(color='black', width=3),
        name='Efficient Frontier',
        hovertemplate='<b>Efficient Portfolio</b><br>' +
                      'Return: %{y:.2%}<br>' +
                      'Volatility: %{x:.2%}<br>' +
                      '<extra></extra>'
    ))

    # Draw Capital Allocation Line (CAL) if Max Sharpe portfolio is available
    tangency_portfolio = None
    if highlight_portfolios and 'Max Sharpe' in highlight_portfolios:
        ret, vol, sharpe = highlight_portfolios['Max Sharpe']
        tangency_portfolio = (ret, vol)

        # Get the min volatility from the data to determine where to start the CAL
        min_vol = efficient_portfolios['Volatility'].min()

        # Start CAL slightly before the minimum volatility on the frontier
        cal_start_vol = max(0, min_vol * 0.8)

        # Extend the line beyond the tangency point
        max_vol = efficient_portfolios['Volatility'].max() * 1.1

        # Calculate returns at both ends using CAL equation: R = Rf + Sharpe * σ
        cal_start_ret = risk_free_rate + sharpe * cal_start_vol
        cal_end_ret = risk_free_rate + sharpe * max_vol

        # Add Capital Allocation Line (RED line)
        fig.add_trace(go.Scatter(
            x=[cal_start_vol, max_vol],
            y=[cal_start_ret, cal_end_ret],
            mode='lines',
            line=dict(color='red', width=3, dash='solid'),
            name='Capital Allocation Line',
            hovertemplate='<b>Capital Allocation Line</b><br>' +
                          'Volatility: %{x:.2%}<br>' +
                          'Return: %{y:.2%}<br>' +
                          'Optimal mix of risk-free asset<br>and tangency portfolio<br>' +
                          f'Slope (Sharpe): {sharpe:.2f}<br>' +
                          '<extra></extra>'
        ))

    # Highlight special portfolios
    if highlight_portfolios:
        for name, (ret, vol, sharpe) in highlight_portfolios.items():
            fig.add_trace(go.Scatter(
                x=[vol],
                y=[ret],
                mode='markers',
                marker=dict(size=15, symbol='star', color='gold', line=dict(width=2, color='black')),
                name=name,
                hovertemplate=f'<b>{name}</b><br>' +
                              f'Return: {ret:.2%}<br>' +
                              f'Volatility: {vol:.2%}<br>' +
                              f'Sharpe: {sharpe:.2f}<br>' +
                              '<extra></extra>'
            ))

    # Current portfolio
    if current_portfolio:
        ret, vol = current_portfolio
        fig.add_trace(go.Scatter(
            x=[vol],
            y=[ret],
            mode='markers',
            marker=dict(size=15, symbol='diamond', color='cyan', line=dict(width=2, color='blue')),
            name='Current Portfolio',
            hovertemplate='<b>Current Portfolio</b><br>' +
                          f'Return: {ret:.2%}<br>' +
                          f'Volatility: {vol:.2%}<br>' +
                          '<extra></extra>'
        ))

    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility (Annual)',
        yaxis_title='Expected Return (Annual)',
        hovermode='closest',
        template='plotly_white',
        height=600,
        xaxis=dict(tickformat='.1%'),
        yaxis=dict(tickformat='.1%'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


def plot_portfolio_allocation(
    weights: pd.Series,
    title: str = 'Portfolio Allocation'
) -> go.Figure:
    """
    Plot portfolio allocation as a pie chart.

    Args:
        weights: Series with asset names as index and weights as values
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Filter out very small weights
    weights_filtered = weights[weights > 0.001]

    fig = go.Figure(data=[go.Pie(
        labels=weights_filtered.index,
        values=weights_filtered.values,
        hole=0.3,
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>' +
                      'Weight: %{value:.2%}<br>' +
                      '<extra></extra>'
    )])

    fig.update_layout(
        title=title,
        height=500,
        template='plotly_white'
    )

    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = 'Asset Correlation Matrix'
) -> go.Figure:
    """
    Plot interactive correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title='Correlation'),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        height=600,
        template='plotly_white',
        xaxis={'side': 'bottom'}
    )

    return fig


def plot_returns_distribution(
    returns: pd.Series,
    title: str = 'Portfolio Returns Distribution'
) -> go.Figure:
    """
    Plot returns distribution with histogram and normal fit.

    Args:
        returns: Series of returns
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns',
        histnorm='probability density',
        marker_color='lightblue',
        opacity=0.7
    ))

    # Normal distribution overlay
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Returns',
        yaxis_title='Density',
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = 'Portfolio Drawdown'
) -> go.Figure:
    """
    Plot portfolio drawdown over time.

    Args:
        returns: Series of returns
        title: Chart title

    Returns:
        Plotly figure object
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red'),
        name='Drawdown',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Drawdown',
        template='plotly_white',
        height=400,
        yaxis=dict(tickformat='.1%')
    )

    return fig


def plot_cumulative_returns(
    cumulative_values: pd.DataFrame,
    title: str = 'Cumulative Returns'
) -> go.Figure:
    """
    Plot cumulative returns for multiple assets/portfolios.

    Args:
        cumulative_values: DataFrame with cumulative return values (indexed to 1.0)
                          Columns represent different assets/portfolios
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Data is already cumulative, no need to calculate
    fig = go.Figure()

    for col in cumulative_values.columns:
        fig.add_trace(go.Scatter(
            x=cumulative_values.index,
            y=cumulative_values[col],
            mode='lines',
            name=col,
            hovertemplate=f'<b>{col}</b><br>' +
                          'Date: %{x}<br>' +
                          'Value: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Cumulative Return (Indexed to 1)',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    return fig


def plot_risk_return_scatter(
    assets_df: pd.DataFrame,
    title: str = 'Risk-Return Profile'
) -> go.Figure:
    """
    Plot risk-return scatter for individual assets.

    Args:
        assets_df: DataFrame with 'Return', 'Volatility', and asset names
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=assets_df['Volatility'],
        y=assets_df['Return'],
        mode='markers+text',
        marker=dict(size=12, color='blue'),
        text=assets_df.index,
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>' +
                      'Return: %{y:.2%}<br>' +
                      'Volatility: %{x:.2%}<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Volatility (Annual)',
        yaxis_title='Expected Return (Annual)',
        template='plotly_white',
        height=500,
        xaxis=dict(tickformat='.1%'),
        yaxis=dict(tickformat='.1%')
    )

    return fig


def plot_monte_carlo_simulation(
    paths_df: pd.DataFrame,
    final_values: np.ndarray,
    initial_value: float,
    title: str = 'Monte Carlo Simulation'
) -> go.Figure:
    """
    Plot Monte Carlo simulation paths and distribution.

    Args:
        paths_df: DataFrame with sample simulation paths
        final_values: Array of final portfolio values
        initial_value: Initial portfolio value
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Simulation Paths', 'Final Value Distribution'),
        column_widths=[0.6, 0.4]
    )

    # Plot sample paths
    for col in paths_df.columns:
        fig.add_trace(
            go.Scatter(
                y=paths_df[col],
                mode='lines',
                line=dict(width=1),
                opacity=0.3,
                showlegend=False,
                hovertemplate='Day: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )

    # Add mean path
    mean_path = paths_df.mean(axis=1)
    fig.add_trace(
        go.Scatter(
            y=mean_path,
            mode='lines',
            line=dict(color='red', width=3),
            name='Mean Path',
            hovertemplate='Day: %{x}<br>Mean Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Plot final value distribution
    fig.add_trace(
        go.Histogram(
            x=final_values,
            nbinsx=50,
            name='Final Values',
            marker_color='lightblue',
            showlegend=False,
            hovertemplate='Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=2
    )

    # Add vertical lines for percentiles
    percentiles = [5, 50, 95]
    colors = ['red', 'green', 'red']
    for p, color in zip(percentiles, colors):
        val = np.percentile(final_values, p)
        fig.add_vline(
            x=val,
            line_dash='dash',
            line_color=color,
            annotation_text=f'{p}th: ${val:,.0f}',
            annotation_position='top',
            row=1, col=2
        )

    fig.update_xaxes(title_text='Days', row=1, col=1)
    fig.update_xaxes(title_text='Final Portfolio Value', row=1, col=2)
    fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig


def plot_risk_contribution(
    risk_contributions: pd.Series,
    title: str = 'Risk Contribution by Asset'
) -> go.Figure:
    """
    Plot risk contribution breakdown.

    Args:
        risk_contributions: Series with asset names and percentage risk contributions
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[
        go.Bar(
            x=risk_contributions.index,
            y=risk_contributions.values,
            marker_color='steelblue',
            text=risk_contributions.values,
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Risk Contribution: %{y:.2f}%<extra></extra>'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title='Asset',
        yaxis_title='Risk Contribution (%)',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def plot_correlation_network(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.3,
    title: str = 'Asset Correlation Network'
) -> go.Figure:
    """
    Plot network graph of asset correlations.

    Args:
        corr_matrix: Correlation matrix DataFrame
        threshold: Minimum correlation to show edge
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Create network graph
    G = nx.Graph()

    # Add nodes
    for asset in corr_matrix.columns:
        G.add_node(asset)

    # Add edges for correlations above threshold
    for i, asset1 in enumerate(corr_matrix.columns):
        for j, asset2 in enumerate(corr_matrix.columns):
            if i < j:  # Avoid duplicate edges
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    G.add_edge(asset1, asset2, weight=abs(corr), correlation=corr)

    # Get positions using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        corr = G.edges[edge]['correlation']
        color = 'red' if corr > 0 else 'blue'
        width = abs(corr) * 3

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            opacity=0.5,
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = list(G.nodes())

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        hovertemplate='<b>%{text}</b><extra></extra>'
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        template='plotly_white',
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig


def plot_scenario_analysis(
    scenario_df: pd.DataFrame,
    title: str = 'Scenario Analysis'
) -> go.Figure:
    """
    Plot scenario analysis results.

    Args:
        scenario_df: DataFrame with scenario results
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scenario_df['Volatility'],
        y=scenario_df['Expected Return'],
        mode='markers',
        marker=dict(
            size=10,
            color=scenario_df['Sharpe'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Sharpe')
        ),
        text=[f"Price: {p}<br>Vol: {v}" for p, v in zip(
            scenario_df['Price Shock'],
            scenario_df['Volatility Shock']
        )],
        hovertemplate='<b>Scenario</b><br>' +
                      '%{text}<br>' +
                      'Return: %{y:.2%}<br>' +
                      'Volatility: %{x:.2%}<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Volatility',
        yaxis_title='Expected Return',
        template='plotly_white',
        height=500,
        xaxis=dict(tickformat='.1%'),
        yaxis=dict(tickformat='.1%')
    )

    return fig


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 60,
    metric: str = 'volatility',
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot rolling window metrics.

    Args:
        returns: Series of returns
        window: Rolling window size
        metric: Metric to plot ('volatility', 'sharpe', 'beta')
        title: Chart title

    Returns:
        Plotly figure object
    """
    if metric == 'volatility':
        rolling_metric = returns.rolling(window).std() * np.sqrt(252)
        y_title = 'Rolling Volatility'
        title = title or f'{window}-Day Rolling Volatility'
    elif metric == 'sharpe':
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_metric = rolling_return / rolling_vol
        y_title = 'Rolling Sharpe Ratio'
        title = title or f'{window}-Day Rolling Sharpe Ratio'
    else:
        rolling_metric = returns.rolling(window).mean() * 252
        y_title = 'Rolling Return'
        title = title or f'{window}-Day Rolling Return'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_metric.index,
        y=rolling_metric.values,
        mode='lines',
        line=dict(color='blue'),
        fill='tozeroy',
        fillcolor='rgba(0,0,255,0.1)',
        name=y_title,
        hovertemplate='Date: %{x}<br>Value: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_title,
        template='plotly_white',
        height=400
    )

    return fig
