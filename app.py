"""
Modern Portfolio Theory Dashboard
A professional, interactive Streamlit dashboard for portfolio optimisation and analysis.

This application demonstrates quantitative finance skills including:
- Mean-variance optimisation
- Risk parity
- Monte Carlo simulation
- Scenario analysis
- Advanced risk metrics
"""

# ============================================================================
# IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Import custom modules for portfolio analysis
import data                 # Market data fetching and preprocessing
import metrics             # Portfolio performance metrics
import optimization        # Optimization algorithms (max Sharpe, min vol, etc.)
import visualization       # Interactive Plotly charts
import simulation          # Monte Carlo and scenario analysis
import ticker_search       # Fuzzy ticker search functionality
from streamlit_searchbox import st_searchbox

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Modern Portfolio Theory Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE & URL MANAGEMENT
# ============================================================================

def initialize_session_state():
    """
    Initialize Streamlit session state variables.

    Handles portfolio state, optimization results, and URL parameter persistence.
    Loads tickers from URL on first page load for bookmarking/sharing.
    """
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = None
    if 'optimisation_results' not in st.session_state:
        st.session_state.optimisation_results = {}
    if 'returns_data' not in st.session_state:
        st.session_state.returns_data = None
    if 'price_data' not in st.session_state:
        st.session_state.price_data = None

    # Load tickers from URL query params on first load
    if 'tickers_list' not in st.session_state:
        query_params = st.query_params
        if 'tickers' in query_params:
            # Parse comma-separated tickers from URL
            tickers_param = query_params['tickers']
            if isinstance(tickers_param, list):
                tickers_param = tickers_param[0]
            st.session_state.tickers_list = [t.strip().upper() for t in tickers_param.split(',') if t.strip()]
        else:
            st.session_state.tickers_list = []

    if 'new_ticker' not in st.session_state:
        st.session_state.new_ticker = ""
    if 'preset_message' not in st.session_state:
        st.session_state.preset_message = ""
    if 'last_added_ticker' not in st.session_state:
        st.session_state.last_added_ticker = None
    if 'ticker_add_count' not in st.session_state:
        st.session_state.ticker_add_count = 0


def update_url_params():
    """
    Update URL query parameters with current tickers for persistence.

    Enables bookmarking and sharing of portfolios via URL.
    """
    if st.session_state.tickers_list:
        tickers_str = ','.join(st.session_state.tickers_list)
        st.query_params['tickers'] = tickers_str
    else:
        # Clear tickers param if list is empty
        if 'tickers' in st.query_params:
            del st.query_params['tickers']

# ============================================================================
# SIDEBAR INPUT CONFIGURATION
# ============================================================================

def sidebar_inputs():
    """
    Create sidebar with all user inputs and configuration options.

    Returns:
        dict: Configuration dictionary containing all user-selected parameters
              including tickers, dates, risk-free rate, optimization settings, etc.
    """
    st.sidebar.title("Configuration")
    st.sidebar.markdown("---")

    # Asset Selection (Expanded by default)
    with st.sidebar.expander("**1. Asset Selection**", expanded=True):
        st.markdown("**Current Portfolio:**")

        # Display existing tickers with delete buttons
        if len(st.session_state.tickers_list) > 0:
            # Add CSS to center button text
            st.markdown("""
                <style>
                .stButton button {
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }
                .stButton button p {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                </style>
            """, unsafe_allow_html=True)

            for ticker in st.session_state.tickers_list:
                col1, col2 = st.columns([5, 1], gap="small")
                with col1:
                    # Add vertical padding to center text with button
                    st.markdown(f'<p style="margin-top: 8px; margin-bottom: 0px; font-weight: 600; font-size: 16px;">{ticker}</p>', unsafe_allow_html=True)
                with col2:
                    if st.button("×", key=f"remove_{ticker}", help=f"Remove {ticker}", use_container_width=True):
                        st.session_state.tickers_list.remove(ticker)
                        update_url_params()
                        st.rerun()

        # Add new ticker section with searchbox
        st.markdown("---")
        st.markdown("**Add New Asset:**")

        # Aggressive CSS to completely override searchbox styling
        st.markdown("""
            <style>
            /* Hide ALL SVG elements (dropdown arrows) */
            [data-testid="stSidebar"] svg,
            [data-testid="stSidebar"] [data-baseweb="icon"],
            div[data-baseweb="select"] svg {
                display: none !important;
                visibility: hidden !important;
                width: 0 !important;
                height: 0 !important;
            }

            /* Override ALL border colors - force gray/blue only */
            div[data-baseweb="select"],
            div[data-baseweb="select"] > div,
            div[data-baseweb="popover"] {
                border: 1px solid #d3d3d3 !important;
                outline: none !important;
                box-shadow: none !important;
            }

            /* On focus - blue border only */
            div[data-baseweb="select"]:focus-within,
            div[data-baseweb="select"]:focus-within > div {
                border: 1px solid #1f77b4 !important;
                outline: none !important;
                box-shadow: none !important;
            }

            /* Override ALL error/invalid states */
            div[data-baseweb="select"][aria-invalid="true"],
            div[data-baseweb="select"][aria-invalid="true"] > div,
            div[data-baseweb="select"][data-error="true"],
            div[data-baseweb="select"][data-error="true"] > div,
            div[class*="error"],
            div[class*="Error"],
            [aria-invalid="true"] {
                border: 1px solid #d3d3d3 !important;
                outline: none !important;
                box-shadow: none !important;
            }

            /* Input field inside */
            div[data-baseweb="select"] input,
            div[data-baseweb="select"] input:focus {
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }

            /* Force override any inline styles */
            [data-testid="stSidebar"] * {
                border-color: inherit !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Use dynamic key to force searchbox reset after each addition
        selected = st_searchbox(
            ticker_search.search_callback,
            key=f"ticker_searchbox_{st.session_state.ticker_add_count}",
            placeholder="",
            label="Search for stocks",
            clear_on_submit=True,
            label_visibility="collapsed"
        )

        if selected:
            # Extract ticker from formatted selection
            new_ticker = ticker_search.extract_ticker_from_selection(selected)

            if new_ticker and new_ticker not in st.session_state.tickers_list:
                st.session_state.tickers_list.append(new_ticker)
                # Increment counter to force searchbox recreation
                st.session_state.ticker_add_count += 1
                update_url_params()
                st.rerun()
            elif new_ticker and new_ticker in st.session_state.tickers_list:
                st.warning(f"{new_ticker} is already in your portfolio")

        # Bulk import section
        st.markdown("---")
        st.markdown("**Or paste multiple tickers:**")
        bulk_tickers = st.text_area(
            "Bulk ticker input",
            placeholder="e.g., AAPL, MSFT, GOOGL, AMZN",
            height=60,
            label_visibility="collapsed",
            key="bulk_ticker_input"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Add All", key="add_bulk_tickers", use_container_width=True):
                if bulk_tickers:
                    # Parse comma-separated tickers
                    new_tickers = [t.strip().upper() for t in bulk_tickers.replace('\n', ',').split(',') if t.strip()]

                    if new_tickers:
                        added_count = 0
                        duplicate_count = 0

                        for ticker in new_tickers:
                            if ticker not in st.session_state.tickers_list:
                                st.session_state.tickers_list.append(ticker)
                                added_count += 1
                            else:
                                duplicate_count += 1

                        if added_count > 0:
                            st.session_state.ticker_add_count += 1
                            update_url_params()
                            st.success(f"Added {added_count} ticker(s)")
                            if duplicate_count > 0:
                                st.info(f"{duplicate_count} ticker(s) already in portfolio")
                            st.rerun()
                        else:
                            st.warning("All tickers are already in your portfolio")
                    else:
                        st.warning("No valid tickers found")
                else:
                    st.warning("Please enter tickers separated by commas")

        with col2:
            if st.button("Clear Input", key="clear_bulk_input", use_container_width=True):
                st.rerun()

        st.info(f"Total: {len(st.session_state.tickers_list)} assets")

    tickers = st.session_state.tickers_list

    # Date Range
    with st.sidebar.expander("**2. Date Range**", expanded=False):
        end_date = datetime.now()

        # Date range mode selection
        date_mode = st.radio(
            "Select date range method:",
            options=["Years back", "Custom dates"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if date_mode == "Years back":
            years_back = st.number_input(
                "Years of historical data",
                min_value=0.5,
                max_value=20.0,
                value=3.0,
                step=0.5,
                help="Number of years to look back from today"
            )
            date_start = end_date - timedelta(days=int(365.25 * years_back))
            date_end = end_date

            # Show calculated date range
            st.caption(f"📅 From {date_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        else:
            # Custom date selection
            start_date = end_date - timedelta(days=365*3)

            date_start = st.date_input(
                "Start Date",
                value=start_date,
                max_value=end_date
            )
            date_end = st.date_input(
                "End Date",
                value=end_date,
                max_value=end_date
            )

    # Risk Parameters
    with st.sidebar.expander("**3. Risk Parameters**", expanded=False):
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Annual risk-free rate for Sharpe ratio calculation"
        ) / 100

        include_risk_free = st.checkbox(
            "Include Risk-Free Asset in Portfolio",
            value=True,
            help="Allow optimizer to allocate to cash/bonds at the risk-free rate"
        )

    # Optimisation Settings
    with st.sidebar.expander("**4. Optimisation Settings**", expanded=True):
        optimisation_method = st.selectbox(
            "Optimisation Method",
            options=[
                "Maximum Sharpe Ratio",
                "Minimum Volatility",
                "Risk Parity",
                "Mean-Variance (Target Return)",
                "Maximum Diversification"
            ],
            help="Select portfolio optimisation strategy"
        )

        allow_short = st.checkbox(
            "Allow Short Positions",
            value=False,
            help="Allow negative weights (short selling)"
        )

        max_weight = st.slider(
            "Maximum Weight per Asset (%)",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            help="Maximum allocation to a single asset"
        ) / 100

        min_weight = 0.0 if not allow_short else -0.5

        target_return = None
        if optimisation_method == "Mean-Variance (Target Return)":
            target_return = st.slider(
                "Target Annual Return (%)",
                min_value=0.0,
                max_value=50.0,
                value=15.0,
                step=1.0
            ) / 100

    # Advanced Settings
    with st.sidebar.expander("**5. Advanced Settings**", expanded=False):
        cov_method = st.selectbox(
            "Covariance Estimation",
            options=["sample", "ewma", "ledoit_wolf"],
            index=2,
            help="Method for estimating covariance matrix"
        )

        num_simulations = st.number_input(
            "Monte Carlo Simulations",
            min_value=100,
            max_value=10000,
            value=10000,
            step=100
        )

        num_portfolios_ef = st.number_input(
            "Efficient Frontier Points",
            min_value=20,
            max_value=200,
            value=200,
            step=10
        )

    return {
        'tickers': tickers,
        'start_date': date_start.strftime('%Y-%m-%d'),
        'end_date': date_end.strftime('%Y-%m-%d'),
        'risk_free_rate': risk_free_rate,
        'include_risk_free': include_risk_free,
        'optimisation_method': optimisation_method,
        'allow_short': allow_short,
        'weight_bounds': (min_weight, max_weight),
        'target_return': target_return,
        'cov_method': cov_method,
        'num_simulations': num_simulations,
        'num_portfolios_ef': num_portfolios_ef
    }

# ============================================================================
# DISPLAY FUNCTIONS - HEADER & DATA LOADING
# ============================================================================

def display_header():
    """Display main dashboard header with title and branding."""
    st.markdown('<div class="main-header">Modern Portfolio Theory Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        Advanced portfolio optimisation and risk analysis tool showcasing quantitative finance expertise.
    </div>
    """, unsafe_allow_html=True)


def load_and_process_data(config):
    """
    Load and process market data for selected assets.

    Args:
        config: Configuration dictionary with tickers, dates, etc.

    Returns:
        tuple: (price_data, returns_data, current_prices) DataFrames
    """
    with st.spinner("Loading market data..."):
        # Validate tickers
        valid_tickers, invalid_tickers = data.validate_tickers(config['tickers'])

        if invalid_tickers:
            st.warning(f" Invalid tickers removed: {', '.join(invalid_tickers)}")

        if not valid_tickers:
            st.error(f"""
            **No valid tickers found**

            All {len(config['tickers'])} ticker(s) entered are invalid or have no data:
            {', '.join(config['tickers'])}

            **Please:**
            - Check ticker symbols are correct (e.g., AAPL not APPLE)
            - Ensure tickers are actively traded
            - Try searching in the add asset box for suggestions
            """)
            return None, None, None

        # Fetch price data
        price_data = data.fetch_stock_data(
            valid_tickers,
            config['start_date'],
            config['end_date']
        )

        if price_data.empty:
            st.error(f"""
            **Failed to fetch market data**

            Could not retrieve price data for: {', '.join(valid_tickers)}

            **Possible causes:**
            - Date range is too far in the past or future
            - Market data service is temporarily unavailable
            - Tickers may be delisted or have insufficient history

            **Try:**
            - Selecting a more recent date range (e.g., last 1-3 years)
            - Using different, more actively traded tickers
            - Waiting a moment and trying again
            """)
            return None, None, None

        # Calculate returns
        returns_data = data.calculate_returns(price_data, method='log')

        # Get current prices
        current_prices = data.get_current_prices(valid_tickers)

        st.session_state.price_data = price_data
        st.session_state.returns_data = returns_data

        return price_data, returns_data, current_prices

# ============================================================================
# DISPLAY FUNCTIONS - PORTFOLIO ANALYSIS & VISUALIZATION
# ============================================================================

def display_market_overview(price_data, returns_data, current_prices):
    """
    Display market overview with asset statistics and performance charts.

    Shows current prices, returns, volatility, and Sharpe ratios for all assets.
    """
    st.header("Market Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Assets Analyzed",
            len(price_data.columns),
            help="Number of assets in portfolio universe"
        )

    with col2:
        st.metric(
            "Data Points",
            len(price_data),
            help="Number of historical data points"
        )

    with col3:
        date_range = (price_data.index[-1] - price_data.index[0]).days
        st.metric(
            "Period (days)",
            date_range,
            help="Historical data period"
        )

    # Asset statistics
    with st.expander(" Asset Statistics", expanded=True):
        periods_per_year = 252

        # Fetch company names
        with st.spinner("Fetching company names..."):
            company_names = data.get_company_names(list(returns_data.columns))

        asset_stats = pd.DataFrame({
            'Ticker': returns_data.columns,
            'Company': [company_names.get(ticker, ticker) for ticker in returns_data.columns],
            'Current Price': current_prices,
            'Annual Return': metrics.annualize_returns(returns_data, periods_per_year),
            'Annual Volatility': metrics.annualize_volatility(returns_data, periods_per_year),
            'Sharpe Ratio': [
                metrics.sharpe_ratio(
                    metrics.annualize_returns(returns_data[col], periods_per_year),
                    metrics.annualize_volatility(returns_data[col], periods_per_year),
                    st.session_state.config['risk_free_rate']
                ) for col in returns_data.columns
            ]
        })

        # Sort by Sharpe Ratio in descending order
        asset_stats = asset_stats.sort_values('Sharpe Ratio', ascending=False)

        # Set ticker as index for display but keep it visible
        asset_stats = asset_stats.set_index('Ticker')

        st.dataframe(
            asset_stats.style.format({
                'Current Price': '${:.2f}',
                'Annual Return': '{:.2%}',
                'Annual Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}'
            }).background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn'),
            use_container_width=True
        )

    # Risk-Return Scatter
    st.plotly_chart(
        visualization.plot_risk_return_scatter(
            asset_stats[['Annual Return', 'Annual Volatility']].rename(
                columns={'Annual Return': 'Return', 'Annual Volatility': 'Volatility'}
            ),
            title='Asset Risk-Return Profile'
        ),
        use_container_width=True
    )


def perform_optimisation(returns_data, config):
    """
    Perform portfolio optimization using the selected method.

    Calculates mean returns and covariance matrix, adds risk-free asset if requested,
    then runs the specified optimization algorithm (max Sharpe, min vol, etc.).

    Args:
        returns_data: DataFrame of asset returns
        config: Configuration dict with optimization settings

    Returns:
        dict: Optimization results including weights, metrics, and status
    """
    st.header("Portfolio Optimisation")

    periods_per_year = 252
    mean_returns = returns_data.mean() * periods_per_year
    cov_matrix = data.calculate_covariance_matrix(
        returns_data,
        method=config['cov_method']
    ) * periods_per_year

    # Add risk-free asset if requested
    asset_names = list(returns_data.columns)
    returns_data_extended = returns_data.copy()

    if config['include_risk_free']:
        # Add risk-free asset to mean returns
        mean_returns = pd.concat([
            mean_returns,
            pd.Series([config['risk_free_rate']], index=['CASH'])
        ])

        # Expand covariance matrix with zeros for risk-free asset
        n = len(cov_matrix)
        # Add row of zeros
        cov_matrix = pd.concat([
            cov_matrix,
            pd.DataFrame(np.zeros((1, n)), columns=cov_matrix.columns, index=['CASH'])
        ])
        # Add column of zeros
        cov_matrix['CASH'] = 0.0

        # Add CASH column to returns data (constant daily risk-free rate)
        daily_rf_rate = config['risk_free_rate'] / periods_per_year
        returns_data_extended['CASH'] = daily_rf_rate

        asset_names.append('CASH')

    method = config['optimisation_method']
    allow_short = config['allow_short']
    weight_bounds = config['weight_bounds']

    try:
        with st.spinner(f"Optimizing portfolio using {method}..."):
            if method =="Maximum Sharpe Ratio":
                weights, info = optimization.maximum_sharpe_ratio(
                    mean_returns.values,
                    cov_matrix.values,
                    config['risk_free_rate'],
                    allow_short,
                    weight_bounds
                )

            elif method =="Minimum Volatility":
                weights, info = optimization.minimum_volatility(
                    mean_returns.values,
                    cov_matrix.values,
                    allow_short,
                    weight_bounds
                )

            elif method =="Risk Parity":
                weights, info = optimization.risk_parity(
                    cov_matrix.values,
                    weight_bounds
                )
                # Calculate return for risk parity
                info['return'] = metrics.portfolio_return(weights, mean_returns.values)
                info['sharpe_ratio'] = metrics.sharpe_ratio(
                    info['return'],
                    info['volatility'],
                    config['risk_free_rate']
                )

            elif method =="Mean-Variance (Target Return)":
                weights, info = optimization.mean_variance_optimization(
                    mean_returns.values,
                    cov_matrix.values,
                    config['target_return'],
                    allow_short,
                    weight_bounds
                )

            elif method =="Maximum Diversification":
                weights, info = optimization.maximum_diversification(
                    cov_matrix.values,
                    weight_bounds
                )

        if info['status'] != 'success':
            # Provide detailed error message
            error_msg = f"**Optimisation failed using {method}**\n\n"

            if 'message' in info:
                error_msg += f"**Error Details:** {info['message']}\n\n"

            error_msg += "**Possible solutions:**\n"
            error_msg += "- Try relaxing the maximum weight constraint (increase from current value)\n"
            error_msg += "- Reduce the number of assets in your portfolio\n"
            error_msg += "- Try a different optimisation method\n"

            if method == "Mean-Variance (Target Return)":
                error_msg += "- Lower your target return - it may be unachievable with current assets\n"

            if not allow_short:
                error_msg += "- Enable short positions if appropriate for your strategy\n"
            else:
                error_msg += "- Disable short positions for a more constrained problem\n"

            error_msg += f"\n**Current settings:** Max weight: {weight_bounds[1]*100:.0f}%, "
            error_msg += f"Short positions: {'Allowed' if allow_short else 'Not allowed'}"

            st.error(error_msg)
            return None

    except Exception as e:
        st.error(f"""
        **Optimisation Error**

        An unexpected error occurred during portfolio optimisation:

        **Error Type:** {type(e).__name__}

        **Error Message:** {str(e)}

        **Troubleshooting:**
        - Check that you have at least 2 assets selected
        - Ensure sufficient historical data is available for all assets
        - Try selecting different assets or a different date range
        - Verify that the risk-free rate is reasonable (typically 0-10%)

        If the problem persists, try starting with a simple 2-3 asset portfolio first.
        """)
        return None

    weights_series = pd.Series(weights, index=asset_names)
    st.session_state.portfolio_weights = weights_series
    st.session_state.optimisation_results = info
    # Display optimisation results
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(
            visualization.plot_portfolio_allocation(
                weights_series,
                title=f'Optimized Portfolio Allocation ({method})'
            ),
            use_container_width=True
        )

    with col2:
        st.subheader("Portfolio Metrics")
        st.metric("Expected Return", f"{info['return']:.2%}")
        st.metric("Volatility", f"{info['volatility']:.2%}")
        st.metric("Sharpe Ratio", f"{info.get('sharpe_ratio', info['return'] / info['volatility']):.2f}")

        # Weights table
        with st.expander("View Detailed Weights"):
            weights_df = pd.DataFrame({
                'Asset': weights_series.index,
                'Weight': weights_series.values
            }).sort_values('Weight', ascending=False)

            st.dataframe(
                weights_df.style.format({'Weight': '{:.2%}'}).background_gradient(
                    subset=['Weight'],
                    cmap='Blues'
                ),
                use_container_width=True,
                hide_index=True
            )

    return weights_series, mean_returns, cov_matrix, returns_data_extended


def display_efficient_frontier(returns_data, config, optimized_weights, mean_returns, cov_matrix):
    """Display efficient frontier analysis."""
    st.header("Efficient Frontier")

    with st.spinner("Generating efficient frontier..."):
        # Generate efficient frontier
        efficient_portfolios = optimization.efficient_frontier(
            mean_returns.values,
            cov_matrix.values,
            num_portfolios=config['num_portfolios_ef'],
            allow_short=config['allow_short'],
            weight_bounds=config['weight_bounds'],
            risk_free_rate=config['risk_free_rate']
        )

        # Generate random portfolios for visualization
        random_portfolios = optimization.random_portfolios(
            mean_returns.values,
            cov_matrix.values,
            num_portfolios=2000,
            risk_free_rate=config['risk_free_rate'],
            allow_short=config['allow_short'],
            weight_bounds=config['weight_bounds']
        )

        # Get special portfolios
        min_vol_weights, min_vol_info = optimization.minimum_volatility(
            mean_returns.values,
            cov_matrix.values,
            config['allow_short'],
            config['weight_bounds']
        )

        max_sharpe_weights, max_sharpe_info = optimization.maximum_sharpe_ratio(
            mean_returns.values,
            cov_matrix.values,
            config['risk_free_rate'],
            config['allow_short'],
            config['weight_bounds']
        )

        highlight_portfolios = {
            'Min Volatility': (
                min_vol_info['return'],
                min_vol_info['volatility'],
                metrics.sharpe_ratio(
                    min_vol_info['return'],
                    min_vol_info['volatility'],
                    config['risk_free_rate']
                )
            ),
            'Max Sharpe': (
                max_sharpe_info['return'],
                max_sharpe_info['volatility'],
                max_sharpe_info['sharpe_ratio']
            )
        }

        # Current optimized portfolio
        current_portfolio = (
            st.session_state.optimisation_results['return'],
            st.session_state.optimisation_results['volatility']
        )

        # Plot
        fig = visualization.plot_efficient_frontier(
            efficient_portfolios,
            random_portfolios,
            highlight_portfolios,
            current_portfolio,
            risk_free_rate=config['risk_free_rate']
        )

        st.plotly_chart(fig, use_container_width=True)


def display_risk_analysis(returns_data, optimized_weights, config):
    """Display comprehensive risk analysis."""
    st.header(" Risk Analysis")

    # Calculate portfolio returns
    portfolio_returns = (returns_data * optimized_weights.values).sum(axis=1)

    # Display risk metrics
    col1, col2, col3, col4 = st.columns(4)

    max_dd, dd_start, dd_end = metrics.maximum_drawdown(portfolio_returns)

    with col1:
        st.metric(
            "Maximum Drawdown",
            f"{max_dd:.2%}",
            help="Largest peak-to-trough decline"
        )

    with col2:
        var_95 = metrics.value_at_risk(portfolio_returns, 0.95)
        st.metric(
            "VaR (95%)",
            f"{var_95:.2%}",
            help="Value at Risk at 95% confidence"
        )

    with col3:
        cvar_95 = metrics.conditional_value_at_risk(portfolio_returns, 0.95)
        st.metric(
            "CVaR (95%)",
            f"{cvar_95:.2%}",
            help="Expected Shortfall beyond VaR"
        )

    with col4:
        sortino = metrics.sortino_ratio(portfolio_returns, config['risk_free_rate'])
        st.metric(
            "Sortino Ratio",
            f"{sortino:.2f}",
            help="Risk-adjusted return using downside deviation"
        )

    # Risk visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Returns Distribution",
        "Drawdown",
        "Risk Contribution",
        "= Correlation"
    ])

    with tab1:
        st.plotly_chart(
            visualization.plot_returns_distribution(portfolio_returns),
            use_container_width=True
        )

    with tab2:
        st.plotly_chart(
            visualization.plot_drawdown(portfolio_returns),
            use_container_width=True
        )

    with tab3:
        periods_per_year = 252
        cov_matrix = returns_data.cov() * periods_per_year
        risk_contrib = metrics.risk_contribution_pct(
            optimized_weights.values,
            cov_matrix.values
        )
        risk_contrib_series = pd.Series(risk_contrib, index=returns_data.columns)

        st.plotly_chart(
            visualization.plot_risk_contribution(risk_contrib_series),
            use_container_width=True
        )

        st.info("Risk contribution shows how much each asset contributes to total portfolio risk.")

    with tab4:
        corr_matrix = returns_data.corr()

        # Exclude CASH from correlation matrix if present
        if 'CASH' in corr_matrix.columns:
            corr_matrix_display = corr_matrix.drop('CASH', axis=0).drop('CASH', axis=1)
            st.info("Note: CASH is excluded from correlation analysis (zero correlation with all assets)")
        else:
            corr_matrix_display = corr_matrix

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                visualization.plot_correlation_heatmap(corr_matrix_display),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                visualization.plot_correlation_network(corr_matrix_display, threshold=0.3),
                use_container_width=True
            )


def display_monte_carlo_simulation(returns_data, optimized_weights, config):
    """Display Monte Carlo simulation."""
    st.header("Monte Carlo Simulation")

    periods_per_year = 252
    mean_returns = returns_data.mean() * periods_per_year
    cov_matrix = returns_data.cov() * periods_per_year

    sim_days = st.slider(
        "Simulation Period (days)",
        min_value=30,
        max_value=756,
        value=252,
        step=30
    )

    initial_value = st.number_input(
        "Initial Portfolio Value ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000
    )

    if st.button("Run Simulation", type="primary"):
        with st.spinner(f"Running {config['num_simulations']} simulations..."):
            final_values, paths_df = simulation.monte_carlo_portfolio_simulation(
                optimized_weights.values,
                mean_returns.values,
                cov_matrix.values,
                initial_value=initial_value,
                num_simulations=config['num_simulations'],
                num_days=sim_days,
                periods_per_year=periods_per_year
            )

            # Display simulation visualization
            st.plotly_chart(
                visualization.plot_monte_carlo_simulation(
                    paths_df,
                    final_values,
                    initial_value
                ),
                use_container_width=True
            )

            # Simulation statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Mean Final Value", f"${final_values.mean():,.0f}")
                st.metric("Median Final Value", f"${np.median(final_values):,.0f}")

            with col2:
                percentile_5 = np.percentile(final_values, 5)
                percentile_95 = np.percentile(final_values, 95)
                st.metric("5th Percentile", f"${percentile_5:,.0f}")
                st.metric("95th Percentile", f"${percentile_95:,.0f}")

            with col3:
                prob_profit = (final_values > initial_value).sum() / len(final_values)
                st.metric("Probability of Profit", f"{prob_profit:.1%}")
                expected_return = (final_values.mean() - initial_value) / initial_value
                st.metric("Expected Return", f"{expected_return:.2%}")

            # VaR/CVaR from simulation
            var_cvar = simulation.var_cvar_simulation(
                final_values,
                initial_value,
                confidence_levels=[0.95, 0.99]
            )

            with st.expander("Detailed Simulation Statistics"):
                sim_stats = simulation.calculate_simulation_statistics(
                    final_values,
                    initial_value
                )

                stats_df = pd.DataFrame(sim_stats, index=[0]).T
                stats_df.columns = ['Value']
                st.dataframe(stats_df, use_container_width=True)


def display_scenario_analysis(returns_data, optimized_weights, config):
    """Display scenario analysis."""
    st.header("Scenario Analysis")

    st.markdown("""
    Analyze how portfolio metrics change under different market conditions.
    Adjust the sliders to shock prices, volatility, and correlations.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        price_shock_min = st.slider(
            "Min Price Shock (%)",
            min_value=-50,
            max_value=0,
            value=-20,
            step=5
        )
        price_shock_max = st.slider(
            "Max Price Shock (%)",
            min_value=0,
            max_value=50,
            value=20,
            step=5
        )

    with col2:
        vol_shock_min = st.slider(
            "Min Volatility Shock (%)",
            min_value=-50,
            max_value=0,
            value=-20,
            step=5
        )
        vol_shock_max = st.slider(
            "Max Volatility Shock (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )

    with col3:
        st.write("Correlation Shock")
        corr_shock = st.slider(
            "Increase systemic risk",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="1.0 no change, 1.0 higher correlations"
        )

    if st.button("Run Scenario Analysis", type="primary"):
        with st.spinner("Analyzing scenarios..."):
            periods_per_year = 252
            mean_returns = returns_data.mean() * periods_per_year
            cov_matrix = returns_data.cov() * periods_per_year

            price_shocks = [1 + x/100 for x in range(price_shock_min, price_shock_max + 5, 5)]
            vol_shocks = [1 + x/100 for x in range(vol_shock_min, vol_shock_max + 10, 10)]
            corr_shocks = [corr_shock]

            scenario_results = simulation.scenario_analysis(
                optimized_weights.values,
                mean_returns.values,
                cov_matrix.values,
                price_shocks,
                vol_shocks,
                corr_shocks
            )

            # Display scenario plot
            st.plotly_chart(
                visualization.plot_scenario_analysis(scenario_results),
                use_container_width=True
            )

            # Scenario table
            with st.expander("Detailed Scenario Results"):
                st.dataframe(
                    scenario_results.style.format({
                        'Expected Return': '{:.2%}',
                        'Volatility': '{:.2%}',
                        'Sharpe': '{:.2f}'
                    }).background_gradient(subset=['Sharpe'], cmap='RdYlGn'),
                    use_container_width=True
                )


def display_performance_tracking(price_data, returns_data, optimized_weights):
    """Display portfolio performance tracking."""
    st.header("Performance Tracking")

    # Calculate portfolio cumulative returns
    # Note: returns_data contains LOG returns, so we use exp(cumsum) not cumprod
    portfolio_returns = (returns_data * optimized_weights.values).sum(axis=1)
    portfolio_cumulative = np.exp(portfolio_returns.cumsum())

    # Fetch benchmark data (SPY)
    benchmark_data = data.get_benchmark_data(
        'SPY',
        returns_data.index[0].strftime('%Y-%m-%d'),
        returns_data.index[-1].strftime('%Y-%m-%d')
    )

    if not benchmark_data.empty:
        try:
            # Benchmark returns are also log returns (same method='log' default)
            benchmark_returns = data.calculate_returns(benchmark_data)

            if benchmark_returns.empty:
                st.warning("Unable to calculate benchmark returns")
            else:
                # Convert log returns to cumulative using exp(cumsum)
                benchmark_cumulative = np.exp(benchmark_returns.cumsum())

                # Extract benchmark series properly
                if isinstance(benchmark_cumulative, pd.DataFrame):
                    benchmark_series = benchmark_cumulative.iloc[:, 0]
                else:
                    benchmark_series = benchmark_cumulative

                # Align indices
                common_index = portfolio_cumulative.index.intersection(benchmark_series.index)

                if len(common_index) == 0:
                    st.warning("No overlapping dates between portfolio and benchmark")
                else:
                    # Combine for comparison (both are already cumulative values)
                    comparison_df = pd.DataFrame({
                        'Portfolio': portfolio_cumulative.loc[common_index],
                        'S&P 500': benchmark_series.loc[common_index]
                    })
        except Exception as e:
            st.warning(f"Unable to process benchmark data: {str(e)}")
            comparison_df = None
    else:
        st.warning("Benchmark data unavailable")
        comparison_df = None

    if comparison_df is not None and not comparison_df.empty:

        st.plotly_chart(
            visualization.plot_cumulative_returns(
                comparison_df,
                title='Portfolio vs Benchmark Performance'
            ),
            use_container_width=True
        )

        # Performance metrics comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Portfolio Metrics")
            total_return = (portfolio_cumulative.iloc[-1] - 1)
            st.metric("Total Return", f"{total_return:.2%}")

            annual_return = metrics.annualize_returns(portfolio_returns, 252)
            st.metric("Annualized Return", f"{annual_return:.2%}")

            annual_vol = metrics.annualize_volatility(portfolio_returns, 252)
            st.metric("Annualized Volatility", f"{annual_vol:.2%}")

        with col2:
            st.subheader("vs S&P 500")
            # Extract benchmark returns properly
            if isinstance(benchmark_returns, pd.DataFrame):
                bench_ret = benchmark_returns.iloc[:, 0]
            else:
                bench_ret = benchmark_returns

            # Align with portfolio returns
            common_idx = portfolio_returns.index.intersection(bench_ret.index)
            port_ret_aligned = portfolio_returns.loc[common_idx]
            bench_ret_aligned = bench_ret.loc[common_idx]

            beta, alpha = metrics.beta_alpha(port_ret_aligned, bench_ret_aligned)
            st.metric("Beta", f"{beta:.2f}")
            st.metric("Alpha (Annual)", f"{alpha:.2%}")

            info_ratio = metrics.information_ratio(port_ret_aligned, bench_ret_aligned)
            st.metric("Information Ratio", f"{info_ratio:.2f}")

    # Rolling metrics
    with st.expander(" Rolling Metrics"):
        window = st.slider("Rolling Window (days)", 20, 120, 60)

        metric_type = st.selectbox(
            "Metric",
            ["volatility", "sharpe"]
        )

        st.plotly_chart(
            visualization.plot_rolling_metrics(
                portfolio_returns,
                window=window,
                metric=metric_type
            ),
            use_container_width=True
        )


def display_export_section(optimized_weights, returns_data, config):
    """Display export and download options."""
    st.header("Export Results")

    col1, col2 = st.columns(2)

    with col1:
        # Export portfolio weights
        weights_df = pd.DataFrame({
            'Asset': optimized_weights.index,
            'Weight': optimized_weights.values
        })

        st.download_button(
            label="Download Portfolio Weights (CSV)",
            data=weights_df.to_csv(index=False),
            file_name="portfolio_weights.csv",
            mime="text/csv"
        )

    with col2:
        # Export portfolio statistics
        portfolio_returns = (returns_data * optimized_weights.values).sum(axis=1)
        periods_per_year = 252
        mean_returns = returns_data.mean() * periods_per_year
        cov_matrix = returns_data.cov() * periods_per_year

        stats = metrics.portfolio_statistics(
            optimized_weights.values,
            returns_data,
            config['risk_free_rate'],
            periods_per_year
        )

        # Convert to DataFrame
        stats_export = {k: v for k, v in stats.items() if k != 'Risk Contributions'}
        stats_df = pd.DataFrame(stats_export, index=[0]).T
        stats_df.columns = ['Value']

        st.download_button(
            label="Download Portfolio Statistics (CSV)",
            data=stats_df.to_csv(),
            file_name="portfolio_statistics.csv",
            mime="text/csv"
        )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point.

    Orchestrates the entire dashboard flow:
    1. Initialize session state and URL persistence
    2. Display header and collect user inputs from sidebar
    3. Load and process market data
    4. Perform portfolio optimization
    5. Display all analysis sections (efficient frontier, Monte Carlo, etc.)
    """
    initialize_session_state()
    display_header()

    # Get configuration from sidebar
    config = sidebar_inputs()
    st.session_state.config = config

    # Check if any tickers are selected
    if len(config['tickers']) == 0:
        # Display welcome message in the center of the page
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center; padding: 3rem; margin: 2rem auto; max-width: 600px;
                        background-color: #f0f2f6; border-radius: 10px; border: 2px solid #e0e0e0;">
                <h2 style="color: #1f77b4; margin-bottom: 1rem;">Welcome to Portfolio Optimisation Dashboard</h2>
                <p style="font-size: 18px; color: #555; line-height: 1.6;">
                    To get started, please add some assets you'd like to consider for your portfolio
                    using the search box in the sidebar.
                </p>
                <p style="font-size: 16px; color: #666; margin-top: 1rem;">
                    You can search by ticker symbol (e.g., AAPL) or company name (e.g., Apple)
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Load data
    price_data, returns_data, current_prices = load_and_process_data(config)

    if price_data is None or returns_data is None:
        st.stop()

    # Display sections
    display_market_overview(price_data, returns_data, current_prices)

    st.divider()

    # Optimisation
    result = perform_optimisation(returns_data, config)

    if result is None:
        st.stop()

    optimized_weights, mean_returns, cov_matrix, returns_data_extended = result

    st.divider()

    # Efficient Frontier
    display_efficient_frontier(
        returns_data,
        config,
        optimized_weights,
        mean_returns,
        cov_matrix
    )

    st.divider()

    # Risk Analysis
    display_risk_analysis(returns_data_extended, optimized_weights, config)

    st.divider()

    # Monte Carlo Simulation
    display_monte_carlo_simulation(returns_data_extended, optimized_weights, config)

    st.divider()

    # Scenario Analysis
    display_scenario_analysis(returns_data_extended, optimized_weights, config)

    st.divider()

    # Performance Tracking
    display_performance_tracking(price_data, returns_data_extended, optimized_weights)

    st.divider()

    # Export Section
    display_export_section(optimized_weights, returns_data_extended, config)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        Modern Portfolio Theory Dashboard | Built with Python, Streamlit, and cvxpy<br>
        Showcasing quantitative finance and portfolio optimisation expertise
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
