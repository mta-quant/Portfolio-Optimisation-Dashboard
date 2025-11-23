# Quick Setup Guide

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

Or use the convenience script:
```bash
./run_dashboard.sh
```

### 4. Access the Dashboard
The application will automatically open in your browser at:
```
http://localhost:8501
```

## Package Dependencies

All required packages are specified in `requirements.txt`:

- **streamlit** (≥1.28.0) - Web application framework
- **streamlit-searchbox** (≥0.1.0) - Fuzzy search widget
- **yfinance** (≥0.2.28) - Market data fetching
- **pandas** (≥2.0.0) - Data manipulation
- **numpy** (≥1.24.0) - Numerical computing
- **scipy** (≥1.10.0) - Scientific computing
- **cvxpy** (≥1.3.0) - Convex optimization
- **plotly** (≥5.14.0) - Interactive visualizations
- **seaborn** (≥0.12.0) - Statistical visualizations
- **matplotlib** (≥3.7.0) - Plotting library
- **networkx** (≥3.1) - Network graphs
- **scikit-learn** (≥1.3.0) - Ledoit-Wolf covariance estimator

## Troubleshooting

### Installation Issues

**If you get permission errors:**
```bash
pip install --user -r requirements.txt
```

**If you want to use a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Runtime Issues

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**Clear cache:**
```bash
streamlit cache clear
```

## First Time Usage

1. **Add Assets**: Use the searchbox in the sidebar to add stock tickers
2. **Or Bulk Import**: Paste comma-separated tickers like: `AAPL, MSFT, GOOGL`
3. **Configure Settings**: Adjust date range, risk-free rate, and optimization method
4. **Fetch Data**: Click the "Fetch Data" button
5. **View Results**: Explore the optimized portfolio and analysis sections

## Default Settings

- **Time Period**: 3 years
- **Risk-Free Rate**: 4.5%
- **Optimization Method**: Maximum Sharpe Ratio
- **Max Weight per Asset**: 40%
- **Monte Carlo Simulations**: 10,000
- **Efficient Frontier Points**: 200
- **Covariance Estimation**: Ledoit-Wolf
- **Include Risk-Free Asset**: Yes

## Features

- ✅ Fuzzy ticker search with 400+ database
- ✅ Custom ticker support for any Yahoo Finance symbol
- ✅ Bulk import via comma-separated list
- ✅ URL persistence for bookmarking
- ✅ 5 optimization methods
- ✅ Monte Carlo simulation (handles zero-variance assets)
- ✅ Efficient frontier visualization
- ✅ Scenario analysis
- ✅ Risk metrics (VaR, CVaR, Sharpe, Sortino, etc.)
- ✅ Benchmark comparison vs S&P 500
- ✅ CSV export functionality

## Project Structure

```
portfolio-optimisation/
├── app.py                 # Main Streamlit application
├── data.py                # Market data fetching and preprocessing
├── metrics.py             # Portfolio performance metrics
├── optimization.py        # Optimization algorithms
├── simulation.py          # Monte Carlo and scenario analysis
├── visualization.py       # Interactive charts
├── ticker_search.py       # Fuzzy search functionality
├── requirements.txt       # Python dependencies
├── README.md             # Comprehensive documentation
├── SETUP.md              # This file
├── run_dashboard.sh      # Launch script
└── .gitignore            # Git ignore rules
```

## Support

For detailed usage instructions, see [README.md](README.md)

For issues or questions:
- Check the troubleshooting section in README.md
- Review error messages in the Streamlit interface
- Verify all dependencies are installed correctly

---

**Ready to optimize your portfolio!** 🚀
