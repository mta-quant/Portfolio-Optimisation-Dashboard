# 📈 Modern Portfolio Theory Dashboard

A professional, interactive Streamlit dashboard for portfolio optimization and risk analysis, showcasing advanced quantitative finance skills and modern portfolio theory concepts.

## 🎯 Overview

This application is a comprehensive portfolio optimization tool designed to demonstrate expertise in:
- **Quantitative Finance**: Mean-variance optimization, risk parity, efficient frontier
- **Risk Management**: VaR, CVaR, maximum drawdown, scenario analysis
- **Computational Finance**: Monte Carlo simulations, correlation analysis
- **Data Science**: Real-time market data processing, advanced visualizations
- **Software Engineering**: Clean, modular, production-ready code architecture

## ✨ Key Features

### 1. **Smart Asset Selection**
- **Fuzzy Search**: Intelligent searchbox with autocomplete across 400+ major tickers
- **Custom Tickers**: Add any valid ticker not in the database (international stocks, crypto, bonds)
- **Bulk Import**: Paste comma-separated lists to add multiple assets at once
- **URL Persistence**: Bookmark and share portfolios via URL parameters

### 2. **Live Market Data Integration**
- Real-time price fetching via yfinance
- Adjustable date ranges for historical analysis (1 month to 10 years)
- Automatic data validation and cleaning
- Support for all Yahoo Finance ticker symbols
- CASH asset support for risk-free allocation

### 3. **Portfolio Optimization Methods**
- **Maximum Sharpe Ratio**: Optimize risk-adjusted returns
- **Minimum Volatility**: Minimize portfolio variance
- **Risk Parity**: Equal risk contribution from all assets
- **Mean-Variance**: Target specific return levels
- **Maximum Diversification**: Maximize diversification ratio

### 4. **Efficient Frontier Analysis**
- Interactive efficient frontier visualization with constraint-respecting random portfolios
- Special portfolios highlighted (min vol, max Sharpe, equal weight)
- Customizable constraints (long-only, position limits)
- Real-time updates as parameters change

### 5. **Advanced Risk Metrics**
- **Traditional**: Volatility, Sharpe ratio, Sortino ratio
- **Downside**: Maximum drawdown, Calmar ratio, semi-deviation
- **Tail Risk**: VaR (95%, 99%), CVaR, tail ratios
- **Risk Attribution**: Marginal and percentage risk contributions
- **Greek Letters**: Beta, alpha relative to benchmark

### 6. **Monte Carlo Simulation**
- Multi-path portfolio value simulation with proper correlation handling
- Zero-variance asset support (CASH, bonds)
- Customizable simulation parameters (paths, time horizon)
- Distribution analysis of final values
- Probability-based outcome analysis
- VaR/CVaR from simulation results

### 7. **Scenario Analysis**
- Price shock scenarios (±50%)
- Volatility shock scenarios (±100%)
- Correlation shock modeling
- Interactive parameter adjustment
- Comprehensive stress testing

### 8. **Interactive Visualizations**
- Efficient frontier with Sharpe ratio color gradient
- Portfolio allocation pie charts with hover details
- Correlation heatmaps (CASH excluded automatically)
- Correlation network graphs with threshold filtering
- Cumulative return comparisons vs benchmark
- Drawdown analysis over time
- Risk contribution breakdown
- Monte Carlo path simulations (100 sample paths)
- Asset risk-return scatter plots

### 9. **Benchmark Comparison**
- S&P 500 (SPY) benchmark tracking
- Beta and alpha calculation
- Information ratio
- Relative performance visualization
- Correlation analysis with benchmark

### 10. **Export Functionality**
- Download portfolio weights (CSV)
- Export performance statistics (CSV)
- Save optimized allocations for implementation

## 🏗️ Architecture

The application follows a clean, modular architecture:

```
portfolio-optimisation/
├── app.py                  # Main Streamlit application
├── data.py                 # Market data fetching and processing
├── metrics.py              # Portfolio performance metrics
├── optimization.py         # Optimization algorithms
├── simulation.py           # Monte Carlo and scenario analysis
├── visualization.py        # Interactive charts and plots
├── ticker_search.py        # Fuzzy search for asset selection
├── requirements.txt        # Python dependencies
└── README.md              # Documentation
```

### Module Descriptions

- **app.py**: Main Streamlit application integrating all components with URL persistence
- **data.py**: Handles all data acquisition, validation, and preprocessing using yfinance
- **metrics.py**: Implements comprehensive portfolio metrics (Sharpe, Sortino, drawdown, VaR, CVaR, etc.)
- **optimization.py**: Multiple optimization methods using scipy and cvxpy with constraint handling
- **simulation.py**: Monte Carlo simulations with zero-variance asset support and scenario analysis
- **visualization.py**: Interactive Plotly visualizations and network graphs
- **ticker_search.py**: Fuzzy matching search with 400+ ticker database and custom ticker support

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the repository**:
```bash
cd portfolio-optimisation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Access the dashboard**:
The application will open automatically in your browser at `http://localhost:8501`

## 📖 Comprehensive Usage Guide

### 🎬 Getting Started: Step-by-Step

#### Step 1: Adding Assets to Your Portfolio

**Method 1: Fuzzy Search (Recommended for Individual Assets)**

1. In the sidebar, find the "Select Assets" section
2. Click on the searchbox that says "Search for ticker..."
3. Start typing a company name or ticker symbol (e.g., "apple", "AAPL", "microsoft")
4. The searchbox will show matching results with company names
5. Select your desired ticker from the dropdown
6. The asset is automatically added to your portfolio

**Features**:
- Fuzzy matching: Works with partial names or misspellings
- 400+ major tickers with company names
- If your ticker isn't found, it will appear as "TICKER - Add Custom Ticker"
- Supports any valid Yahoo Finance ticker (international, crypto, ETFs, bonds)

**Method 2: Bulk Import (Recommended for Multiple Assets)**

1. Below the searchbox, find the "Or paste multiple tickers:" section
2. Paste a comma-separated list of tickers, for example:
   ```
   AAPL, MSFT, GOOGL, AMZN, NVDA
   ```
3. Click the "Add All" button
4. The dashboard will show how many tickers were added and if any were duplicates

**Features**:
- Accepts comma-separated format: `AAPL, MSFT, GOOGL`
- Accepts newline-separated format (one ticker per line)
- Automatically removes whitespace and converts to uppercase
- Shows feedback on added vs duplicate tickers
- "Clear Input" button to reset the text area

**Method 3: URL Persistence (Bookmarking & Sharing)**

Your portfolio is automatically saved in the URL! You can:
- Bookmark the page to save your portfolio
- Share the URL with others
- Refresh the page without losing your portfolio

Example URL format:
```
http://localhost:8501/?tickers=AAPL,MSFT,GOOGL,AMZN
```

**Managing Your Assets**

- View all selected assets in the "Currently Selected Assets" box
- Remove any asset by clicking its "❌" button
- Assets update in real-time
- Portfolio statistics recalculate automatically

#### Step 2: Configure Data Parameters

**Time Period Selection**
1. Find the "Date Range" section in the sidebar
2. Choose your analysis period:
   - **1 Month**: Very short-term, high noise
   - **3 Months**: Short-term trends
   - **6 Months**: Medium-term patterns
   - **1 Year**: Standard for annual analysis
   - **2 Years**: Captures multiple market cycles
   - **3 Years** (Default): Balanced historical view
   - **5 Years**: Long-term perspective
   - **10 Years**: Full business cycle analysis

**Risk-Free Rate**
1. Set the annual risk-free rate (default: 4.5%)
2. Used for Sharpe ratio calculations
3. Typically use:
   - 10-year Treasury yield for long-term portfolios
   - 3-month T-bill rate for short-term portfolios
   - Current rate as of your analysis date

**When to Click "Fetch Data"**
- After adding/removing assets
- After changing the time period
- After modifying any data parameters
- The button fetches fresh market data from Yahoo Finance

#### Step 3: Set Optimization Constraints

**Weight Constraints**

1. **Maximum Weight per Asset** (Default: 40%)
   - Controls concentration risk
   - Prevents over-allocation to single assets
   - Examples:
     - 20%: Highly diversified (minimum 5 assets needed)
     - 40%: Moderately diversified
     - 100%: No concentration limits

2. **Allow Short Selling** (Default: Off)
   - Enable to allow negative weights (short positions)
   - When OFF: Long-only portfolio (all weights ≥ 0)
   - When ON: Can bet against assets (weights can be negative)

**Choosing an Optimization Method**

**Maximum Sharpe Ratio** (Recommended for Most Cases)
- Optimizes risk-adjusted returns
- Best for: Growth-oriented investors seeking efficiency
- Produces: Portfolio with highest Sharpe ratio
- May result in: Concentrated positions in best-performing assets

**Minimum Volatility**
- Minimizes portfolio risk/variance
- Best for: Conservative investors, capital preservation
- Produces: Lowest possible volatility portfolio
- May result in: Lower returns but stable performance

**Risk Parity**
- Equal risk contribution from each asset
- Best for: Diversification-focused strategies, institutional portfolios
- Produces: Balanced risk exposure across assets
- May result in: Higher allocation to lower-volatility assets

**Mean-Variance (Target Return)**
- Specify desired return, minimize risk
- Best for: Investors with specific return requirements
- Produces: Minimum risk portfolio for your target return
- Requires: Setting a realistic target return percentage

**Maximum Diversification**
- Maximizes the diversification ratio
- Best for: Reducing concentration risk, avoiding asset clusters
- Produces: Most diversified portfolio structure
- May result in: More even weight distribution

#### Step 4: Understanding the Dashboard Sections

### 📊 Portfolio Composition Section

**Allocation Chart**
- Interactive pie chart showing percentage allocation
- Hover over segments for exact percentages
- Color-coded for easy identification
- Updates with each optimization

**View Detailed Weights**
- Click to expand the weights table
- Shows Asset name and Weight percentage
- Sorted alphabetically
- Formatted with color gradient (darker blue = higher weight)
- Can be exported via the download button

**Download Weights Button**
- Downloads portfolio weights as CSV file
- Format: Asset, Weight
- Ready for implementation in brokerage accounts
- Time-stamped filename

### 📈 Portfolio Performance Section

**Key Metrics Display**
Shows 6-8 key metrics in an organized grid:

1. **Expected Annual Return**: Forecasted return based on historical data
2. **Annual Volatility**: Standard deviation (risk measure)
3. **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good, >2 is excellent)
4. **Maximum Drawdown**: Worst peak-to-trough decline
5. **VaR 95%**: Value at Risk (95% confidence, expected max loss)
6. **CVaR 95%**: Conditional VaR (average loss beyond VaR)
7. **Sortino Ratio**: Downside risk-adjusted return
8. **Calmar Ratio**: Return relative to max drawdown

**Risk-Adjusted Metrics Explained**
- **Sharpe > 1.0**: Good risk-adjusted performance
- **Sharpe > 2.0**: Excellent performance
- **Sortino**: Similar to Sharpe but only penalizes downside volatility
- **Calmar**: Higher is better, compares returns to worst losses

**Historical Performance Chart**
- Shows cumulative returns over time
- Blue line: Your portfolio
- Red line: S&P 500 benchmark (SPY)
- Hover to see exact values and dates
- Zoom and pan enabled

**Drawdown Analysis**
- Visualizes underwater periods (losses from peak)
- Shows recovery periods
- Identifies worst drawdown periods
- Helps assess portfolio resilience

### 📉 Asset Analysis Section

**Asset Statistics Table**
- **Automatically sorted by Sharpe Ratio** (best performers first)
- **Non-scrollable**: Shows entire portfolio at once
- Columns:
  - **Current Price**: Latest market price
  - **Annual Return**: Annualized historical return
  - **Annual Volatility**: Annualized risk measure
  - **Sharpe Ratio**: Color-coded (green=good, red=poor)

**Understanding Asset Performance**
- High Sharpe = Good risk-adjusted performer
- High Return + High Vol = Aggressive growth asset
- Low Vol + Moderate Return = Defensive asset
- Negative Sharpe = Underperformer

**Asset Risk-Return Scatter**
- X-axis: Volatility (risk)
- Y-axis: Return
- Each point represents one asset
- Top-left quadrant: Best (high return, low risk)
- Bottom-right quadrant: Worst (low return, high risk)

### 🔗 Correlation Analysis Section

**Four Tabs for Different Views**

**Tab 1: Correlation Heatmap**
- Color-coded correlation matrix
- Range: -1 (perfect negative) to +1 (perfect positive)
- Hover for exact correlation values
- **Note**: CASH is automatically excluded (zero correlation by definition)

**Interpreting Correlations**:
- **0.7 to 1.0**: Strong positive correlation (move together)
- **0.3 to 0.7**: Moderate positive correlation
- **-0.3 to 0.3**: Low/no correlation (good for diversification)
- **-0.7 to -0.3**: Moderate negative correlation
- **-1.0 to -0.7**: Strong negative correlation (hedge potential)

**Tab 2: Correlation Network**
- Visual representation of asset relationships
- Lines connect correlated assets (threshold: 0.3)
- Thicker lines = Stronger correlations
- Clustered nodes = Related assets
- Isolated nodes = Diversification opportunities

**Tab 3: Cumulative Returns**
- Individual asset performance over time
- Compare all assets on one chart
- Identifies outperformers and underperformers
- Shows convergence/divergence patterns

**Tab 4: Risk Contribution**
- Shows how much each asset contributes to total portfolio risk
- Useful for risk management
- May differ from weight percentages
- Identifies risk concentrations

### 🎯 Efficient Frontier Section

**What is the Efficient Frontier?**
The efficient frontier is the set of optimal portfolios offering the highest expected return for each level of risk.

**Chart Elements**:
- **Blue line**: Efficient frontier
- **Gray dots**: Random portfolios (all respecting your constraints)
- **Red star**: Minimum volatility portfolio
- **Green star**: Maximum Sharpe ratio portfolio
- **Orange star**: Equal weight portfolio

**How to Read the Chart**:
- X-axis: Volatility (risk)
- Y-axis: Expected return
- Color: Sharpe ratio (green=better, red=worse)
- Portfolios on the line are optimal
- Portfolios below the line are suboptimal
- No feasible portfolio can be above the line

**Using This for Portfolio Selection**:
1. Identify your risk tolerance on the X-axis
2. Find the corresponding point on the frontier
3. That point shows your optimal expected return
4. Select the optimization method that gets you closest to that point

### 🎲 Monte Carlo Simulation Section

**What it Does**:
Simulates thousands of possible future portfolio paths based on historical statistics.

**Parameters**:
1. **Number of Simulations**: 100-10,000 paths
   - More simulations = More accurate but slower
   - 1,000 is usually sufficient
   - 10,000 for final analysis

2. **Time Horizon (Days)**: 1-1260 days
   - 252 days = 1 trading year
   - Use 252 for annual projections
   - Use 1260 for 5-year projections

3. **Initial Value**: Starting portfolio value (default: $10,000)

**Outputs**:

**Simulation Paths Chart**
- Shows 100 sample paths (randomly selected from all simulations)
- Fan shape is normal (uncertainty increases over time)
- Wide fan = High volatility
- Narrow fan = Lower risk

**Final Value Distribution**
- Histogram of ending portfolio values
- Shows probability distribution
- Left tail = Downside risk
- Right tail = Upside potential

**Simulation Statistics**
- Mean/Median Final Value
- Standard Deviation
- Min/Max Values
- Probability of Loss
- Probability of >10% Gain
- Skewness (asymmetry)
- Kurtosis (tail thickness)

**VaR/CVaR from Simulation**
- VaR 95%: Loss exceeded in only 5% of scenarios
- CVaR 95%: Average loss in worst 5% of scenarios
- VaR 99%: Loss exceeded in only 1% of scenarios
- CVaR 99%: Average loss in worst 1% of scenarios

**Use Cases**:
- Retirement planning
- Risk assessment
- Worst-case scenario analysis
- Probability-based decisions

### 🌪️ Scenario Analysis Section

**What it Does**:
Tests your portfolio under various market shock scenarios.

**Three Types of Shocks**:

1. **Price Shocks**: ±50% return changes
   - +50%: Bull market scenario
   - 0%: Base case (no shock)
   - -50%: Bear market scenario

2. **Volatility Shocks**: ±100% volatility changes
   - +100%: Volatility doubles (crisis)
   - 0%: Normal volatility
   - -100%: Volatility halves (calm market)

3. **Correlation Shocks**: 0-100% correlation increase
   - 0%: Base correlations
   - 50%: Moderate crisis (assets move together more)
   - 100%: Extreme crisis (all correlations → 1)

**Reading the Results Table**:
- Each row is a different scenario combination
- Shows Expected Return, Volatility, Sharpe under that scenario
- Identifies how portfolio performs in different market environments

**Key Scenarios to Check**:
1. **-50% Price, +100% Vol**: Severe crisis
2. **+50% Price, 0% Vol**: Strong bull market
3. **-30% Price, +50% Vol, 50% Corr**: Moderate crisis
4. **Current Parameters**: Base case

**Use This For**:
- Stress testing
- Understanding portfolio behavior in extremes
- Comparing strategy robustness
- Risk management planning

### 📥 Export and Save

**Portfolio Weights CSV**
- Click "Download Weights" button
- Opens CSV with Asset and Weight columns
- Use for implementation in brokerage accounts

**Portfolio Statistics CSV**
- Click "Download Statistics" button
- Exports all performance metrics
- Use for reporting or further analysis

**URL Bookmark**
- Save browser bookmark to preserve portfolio selection
- Share URL with colleagues or clients
- Portfolio reconstructs from URL parameters

## 🔧 Advanced Features

### CASH Asset Support

Add "CASH" as a ticker to include a risk-free asset:
1. Type "CASH" in the searchbox or bulk import
2. System treats CASH as zero-volatility asset
3. Returns equal to risk-free rate
4. Useful for cash drag analysis or conservative portfolios
5. Automatically excluded from correlation analysis
6. Monte Carlo simulation handles zero-variance correctly

### Custom Covariance Estimation

In advanced settings (if enabled):
- **Standard**: Sample covariance matrix
- **EWMA**: Exponentially weighted (recent data weighted more)
- **Ledoit-Wolf**: Shrinkage estimator (reduces estimation error)

### Benchmark Selection

Currently defaults to SPY (S&P 500). To analyze against different benchmarks:
- Add your benchmark ticker to the portfolio
- Use correlation analysis to see relationship
- Compare cumulative returns in the correlation tabs

## 📊 Sample Portfolios

Try these pre-configured portfolios to explore the dashboard:

**Tech Growth Portfolio** (High Risk, High Return):
```
AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AMD
```

**Diversified Large Cap** (Moderate Risk):
```
AAPL,MSFT,GOOGL,AMZN,JPM,JNJ,V,PG,NVDA,MA
```

**Sector Rotation ETFs** (Broad Diversification):
```
XLF,XLK,XLE,XLV,XLY,XLP,XLI,XLU,XLB,XLRE
```

**Risk Parity Example** (Multi-Asset):
```
SPY,TLT,GLD,VNQ,DBC
```

**Conservative with CASH** (Capital Preservation):
```
JNJ,PG,KO,PEP,WMT,CASH
```

**International Diversification**:
```
SPY,EFA,EEM,VNQ,TLT,GLD
```

## 🎓 Technical Highlights

### Quantitative Finance Concepts

1. **Modern Portfolio Theory (MPT)**
   - Mean-variance optimization framework
   - Efficient frontier construction with constraint handling
   - Capital allocation line
   - Risk-return tradeoff optimization

2. **Risk Measures**
   - Parametric VaR and CVaR
   - Historical simulation with proper correlation
   - Tail risk analysis
   - Drawdown metrics

3. **Optimization Techniques**
   - Convex optimization (cvxpy)
   - Constrained optimization (scipy)
   - Risk budgeting algorithms
   - Maximum diversification ratio

4. **Statistical Methods**
   - Covariance matrix estimation
   - Zero-variance asset handling
   - EWMA (Exponentially Weighted Moving Average)
   - Ledoit-Wolf shrinkage
   - Cholesky decomposition with regularization

5. **Monte Carlo Methods**
   - Geometric Brownian Motion
   - Correlated asset simulation
   - Zero-variance asset integration
   - Bootstrap resampling
   - Regime-switching models

### Code Quality Features

- **Type Hints**: All functions include type annotations for clarity
- **Documentation**: Comprehensive docstrings following NumPy style
- **Error Handling**: Robust validation and graceful error management
- **Caching**: Streamlit caching for performance optimization
- **Modularity**: Clean separation of concerns across 7 modules
- **Scalability**: Efficient NumPy/SciPy operations for large portfolios
- **Numerical Stability**: Regularization for singular matrices
- **URL State Management**: Persistent portfolio state across sessions

## 🔬 Advanced Use Cases

### 1. Backtesting Strategies
- Set historical date ranges to test portfolio performance
- Compare different optimization methods across time periods
- Analyze strategy performance during specific market conditions

### 2. Crisis Analysis
- Adjust date range to include 2008 crisis, 2020 COVID crash, or 2022 bear market
- Use scenario analysis to stress test
- Check maximum drawdown and recovery patterns

### 3. Correlation Analysis
- Use correlation network to identify cluster risk
- Find negative correlations for hedging opportunities
- Assess diversification effectiveness

### 4. Risk Attribution
- Understand which assets drive portfolio risk
- Rebalance based on risk contribution
- Implement risk parity principles

### 5. Retirement Planning
- Use Monte Carlo with long time horizons (1260+ days)
- Set initial value to current savings
- Analyze probability of reaching goals

### 6. Multi-Asset Allocation
- Combine stocks, bonds (TLT), commodities (GLD, DBC), real estate (VNQ)
- Use risk parity for balanced exposure
- Include CASH for defensive positioning

## 🛠️ Customization

### Adding Custom Optimization Methods

To add a new optimization method, edit `optimization.py`:

```python
def custom_optimization(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    allow_short: bool = False,
    weight_bounds: Tuple[float, float] = (0, 1),
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Your custom optimization logic.

    Args:
        mean_returns: Expected returns
        cov_matrix: Covariance matrix
        allow_short: Whether to allow short positions
        weight_bounds: (min_weight, max_weight) per asset

    Returns:
        Tuple of (weights array, info dict)
    """
    n_assets = len(mean_returns)

    # Implement your optimization logic here
    weights = ...  # Your solution

    # Calculate metrics
    from metrics import portfolio_return, portfolio_volatility, sharpe_ratio
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    sharpe = sharpe_ratio(ret, vol, kwargs.get('risk_free_rate', 0.02))

    info = {
        'status': 'success',
        'return': ret,
        'volatility': vol,
        'sharpe': sharpe
    }

    return weights, info
```

Then add to `app.py` optimization method dropdown.

### Adding Custom Metrics

To add a new performance metric, edit `metrics.py`:

```python
def custom_metric(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    **kwargs
) -> float:
    """
    Calculate your custom metric.

    Args:
        returns: Portfolio returns series
        benchmark_returns: Optional benchmark for comparison

    Returns:
        Metric value
    """
    # Implement your calculation
    metric_value = ...

    return metric_value
```

Then display in the metrics section of `app.py`.

### Adding New Visualizations

To add custom charts, edit `visualization.py`:

```python
def custom_visualization(data: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Create your custom Plotly figure.

    Args:
        data: Input data

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add your traces
    fig.add_trace(...)

    # Update layout
    fig.update_layout(
        title=kwargs.get('title', 'Custom Chart'),
        xaxis_title=kwargs.get('xaxis', 'X'),
        yaxis_title=kwargs.get('yaxis', 'Y'),
        template='plotly_white',
        hovermode='closest'
    )

    return fig
```

Then add to appropriate section in `app.py`:
```python
st.plotly_chart(
    visualization.custom_visualization(your_data),
    use_container_width=True
)
```

## 📈 Performance Considerations

- **Caching**: Data fetching cached for 1 hour via `@st.cache_data`
- **Optimization**: Efficient NumPy vectorized operations
- **Visualization**: Sample paths (100) for Monte Carlo to reduce rendering time
- **Lazy Loading**: Components load on-demand to improve initial load time
- **Matrix Operations**: Optimized linear algebra with SciPy/NumPy
- **URL State**: Lightweight ticker-only persistence

**Performance Tips**:
- Start with fewer assets (5-10) for faster optimization
- Use 1,000 Monte Carlo simulations for interactive work
- Increase to 10,000 simulations for final analysis
- Reduce efficient frontier points (20-50) for speed
- Use shorter time periods (1-2 years) for rapid iteration

## 🔐 Data Privacy & Security

- All data fetched from public Yahoo Finance API (no authentication)
- No personal data collected or stored
- No backend server - all processing client-side
- No cookies or tracking
- Portfolio data only in URL (user-controlled)
- No data sent to external services except Yahoo Finance

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. "Invalid ticker" warning or empty data**
- **Cause**: Ticker doesn't exist or has insufficient historical data
- **Fix**: Verify ticker on Yahoo Finance website
- **Note**: Some tickers only have recent data (check date range)

**2. "Optimization failed" error**
- **Cause**: Constraints too restrictive or target return infeasible
- **Fix**:
  - Increase maximum weight per asset
  - Lower target return (if using mean-variance)
  - Check if enough assets for diversification
  - Try different optimization method

**3. "Matrix is not positive definite" error**
- **Cause**: Numerical issues with covariance matrix
- **Fix**: This should be handled automatically now
- **If persists**: Check for duplicate tickers or insufficient data

**4. Random portfolios appear above efficient frontier**
- **Fixed**: All random portfolios now respect weight constraints
- **If occurs**: Report as bug (should not happen)

**5. Slow performance with many assets**
- **Cause**: Computational complexity increases with portfolio size
- **Fix**:
  - Reduce Monte Carlo simulations (100-1000)
  - Decrease efficient frontier points (20-50)
  - Use shorter time period (1-2 years)
  - Consider portfolio pre-selection (top 10-15 assets)

**6. CASH asset errors**
- **Fixed**: Zero-variance assets handled correctly
- **Note**: CASH excluded from correlation analysis automatically

**7. Page refresh loses portfolio**
- **Fixed**: Portfolio saved in URL automatically
- **Check**: URL should contain `?tickers=AAPL,MSFT,...`
- **If missing**: Browser may have cleared URL, re-add tickers

**8. Fuzzy search not finding ticker**
- **Solution**: Type the full ticker and select "Add Custom Ticker"
- **Works for**: International stocks (e.g., 0700.HK), crypto (BTC-USD), any Yahoo Finance ticker

**9. Bulk import not working**
- **Check**: Tickers separated by commas or newlines
- **Remove**: Any extra characters or spaces (auto-handled)
- **Case**: Doesn't matter (auto-converted to uppercase)

**10. Charts not displaying**
- **Cause**: Browser compatibility or JavaScript disabled
- **Fix**:
  - Use modern browser (Chrome, Firefox, Safari, Edge)
  - Enable JavaScript
  - Clear browser cache
  - Check browser console for errors

## 📚 References & Further Reading

### Academic Papers
- Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*
- Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium"
- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization"
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). "On the Properties of Equally-Weighted Risk Contributions Portfolios"
- Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix"

### Books
- "Modern Portfolio Theory and Investment Analysis" by Elton, Gruber, Brown, and Goetzmann
- "Quantitative Risk Management" by McNeil, Frey, and Embrechts
- "Active Portfolio Management" by Grinold and Kahn
- "Risk Parity Fundamentals" by Edward Qian

### Online Resources
- [Yahoo Finance](https://finance.yahoo.com): Data source
- [CVXPY Documentation](https://www.cvxpy.org): Convex optimization
- [Plotly Python](https://plotly.com/python): Interactive visualizations
- [Streamlit Docs](https://docs.streamlit.io): Web app framework

## 🤝 Contributing

This is a demonstration project showcasing quantitative finance and software engineering skills.

Feel free to:
- Fork for your own use
- Extend with new features
- Adapt for different asset classes
- Integrate with your data sources

## 📄 License

This project is provided as-is for educational and demonstration purposes.

**Disclaimer**: This tool is for educational purposes only. Not financial advice. Past performance does not guarantee future results. Always do your own research and consult financial professionals before making investment decisions.

## 👤 Author

Created as a quantitative finance portfolio project demonstrating:
- Advanced Python programming and software architecture
- Deep understanding of modern portfolio theory and quantitative finance
- Data visualization and user experience design
- Production-ready code with proper error handling
- Real-world application of academic finance concepts

## 🎯 Skills Demonstrated

### Technical Skills
- **Python**: NumPy, Pandas, SciPy (advanced linear algebra, statistics)
- **Optimization**: cvxpy (convex optimization), scipy.optimize (constrained optimization)
- **Data Visualization**: Plotly (interactive charts), Seaborn, Matplotlib, NetworkX
- **Web Development**: Streamlit (including state management, URL persistence)
- **API Integration**: yfinance (real-time market data)
- **Software Architecture**: Modular design, separation of concerns

### Financial Skills
- **Portfolio Optimization**: Mean-variance, risk parity, maximum diversification
- **Risk Management**: VaR, CVaR, tail risk, drawdown analysis
- **Quantitative Analysis**: Monte Carlo simulation, scenario analysis, stress testing
- **Financial Modeling**: Covariance estimation, correlation analysis
- **Market Data Analysis**: Time series processing, returns calculation
- **Performance Attribution**: Risk decomposition, factor analysis

### Software Engineering
- **Modular Architecture**: 7 separate modules with clear responsibilities
- **Clean Code**: Type hints, comprehensive docstrings, PEP 8 compliance
- **Documentation**: Extensive README with usage examples
- **Error Handling**: Robust validation, graceful degradation
- **Performance Optimization**: Caching, vectorization, efficient algorithms
- **User Experience**: Intuitive interface, helpful feedback, persistent state
- **Numerical Stability**: Matrix regularization, edge case handling

### Finance Domain Expertise
- Modern Portfolio Theory (MPT)
- Capital Asset Pricing Model (CAPM)
- Risk-adjusted performance metrics
- Correlation and covariance modeling
- Monte Carlo methods in finance
- Optimization under constraints
- Multi-asset portfolio construction

---

**Built with** Python 🐍 | Streamlit 🎈 | Plotly 📊 | cvxpy 🔢 | NumPy 🔬 | Pandas 🐼

*Professional quantitative finance project showcasing technical and domain expertise*

## 🚀 Quick Start Examples

### Example 1: Basic Portfolio
```python
# In searchbox or bulk import:
AAPL, MSFT, GOOGL
# Click "Fetch Data"
# Select "Maximum Sharpe Ratio"
# View your optimized portfolio!
```

### Example 2: Conservative with Cash
```python
# Bulk import:
JNJ, PG, KO, PEP, CASH
# Set time period: 3 years
# Risk-free rate: 4.5%
# Method: Minimum Volatility
# Max weight: 30%
```

### Example 3: Aggressive Tech Growth
```python
# Bulk import:
NVDA, AMD, TSLA, PLTR, COIN
# Set time period: 1 year
# Method: Maximum Sharpe
# Max weight: 40%
# Run Monte Carlo with 5000 simulations
```

### Example 4: Risk Parity Multi-Asset
```python
# Bulk import:
SPY, TLT, GLD, VNQ, DBC, CASH
# Method: Risk Parity
# Check risk contribution tab
# Verify equal risk distribution
```

**Ready to optimize your portfolio? Launch the dashboard and start exploring!** 🎉
