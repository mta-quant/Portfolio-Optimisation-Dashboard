"""
Data Module - Market Data Fetching and Processing

This module handles all data acquisition and preprocessing for portfolio analysis.
Provides functions to fetch historical price data, compute returns, and handle caching.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import streamlit as st

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce API calls
def fetch_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical adjusted close prices for multiple tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval (1d, 1wk, 1mo, etc.)

    Returns:
        DataFrame with adjusted close prices for each ticker
    """
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            st.error(f"No data returned for tickers: {tickers}")
            return pd.DataFrame()

        # Handle MultiIndex columns (happens with single or multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            # Extract Close prices from MultiIndex
            close_data = data.xs('Close', level=0, axis=1)
            df = pd.DataFrame(close_data)
        elif 'Close' in data.columns:
            # Simple column structure
            if len(tickers) == 1:
                df = pd.DataFrame({tickers[0]: data['Close']})
            else:
                df = data['Close']
        else:
            # Fallback: assume entire data is price data
            df = pd.DataFrame(data)

        # Ensure we have proper column names
        if len(tickers) == 1 and len(df.columns) == 1:
            df.columns = [tickers[0]]

        # Handle missing data
        df = df.ffill().bfill()

        return df

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_current_prices(tickers: List[str]) -> pd.Series:
    """
    Get current (near real-time) prices for given tickers.

    Args:
        tickers: List of stock ticker symbols

    Returns:
        Series with current prices
    """
    try:
        prices = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            prices[ticker] = info.get('currentPrice', info.get('regularMarketPrice', np.nan))

        return pd.Series(prices)

    except Exception as e:
        st.error(f"Error fetching current prices: {str(e)}")
        return pd.Series()

# ============================================================================
# RETURNS CALCULATION & PREPROCESSING
# ============================================================================

def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """
    Calculate returns from price data.

    Args:
        prices: DataFrame of price data
        method: 'log' for log returns or 'simple' for arithmetic returns

    Returns:
        DataFrame of returns
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:  # simple returns
        returns = prices.pct_change()

    return returns.dropna()

# ============================================================================
# COVARIANCE MATRIX ESTIMATION
# ============================================================================

def calculate_covariance_matrix(
    returns: pd.DataFrame,
    method: str = 'sample',
    shrinkage_target: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate covariance matrix with optional shrinkage estimators.

    Three methods available:
    - 'sample': Standard sample covariance (unbiased estimator)
    - 'ewma': Exponentially weighted moving average (recent data weighted more)
    - 'ledoit_wolf': Ledoit-Wolf shrinkage (reduces estimation error)

    Args:
        returns: DataFrame of asset returns
        method: 'sample', 'ewma', or 'ledoit_wolf'
        shrinkage_target: Target for shrinkage ('constant_correlation', 'single_factor', etc.)

    Returns:
        Covariance matrix as DataFrame
    """
    if method == 'sample':
        # Standard sample covariance matrix
        return returns.cov()

    elif method == 'ewma':
        # Exponentially Weighted Moving Average - gives more weight to recent data
        span = 60  # Approximately 3 months of daily data
        return returns.ewm(span=span).cov().iloc[-len(returns.columns):, :]

    elif method == 'ledoit_wolf':
        # Ledoit-Wolf shrinkage estimator - reduces estimation error
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns)
        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index=returns.columns,
            columns=returns.columns
        )
        return cov_matrix

    else:
        return returns.cov()


def get_company_names(tickers: List[str]) -> dict:
    """
    Fetch company names for given tickers.

    Args:
        tickers: List of stock ticker symbols

    Returns:
        Dictionary mapping tickers to company names
    """
    names = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            names[ticker] = info.get('longName', info.get('shortName', ticker))
        except:
            names[ticker] = ticker

    return names


def get_stock_info(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch additional stock information (sector, market cap, etc.).

    Args:
        tickers: List of stock ticker symbols

    Returns:
        DataFrame with stock metadata
    """
    info_data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            info_data.append({
                'Ticker': ticker,
                'Name': info.get('longName', ticker),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Market Cap': info.get('marketCap', np.nan),
                'Beta': info.get('beta', np.nan)
            })
        except:
            info_data.append({
                'Ticker': ticker,
                'Name': ticker,
                'Sector': 'N/A',
                'Industry': 'N/A',
                'Market Cap': np.nan,
                'Beta': np.nan
            })

    return pd.DataFrame(info_data)


def validate_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate that tickers exist and can be downloaded.

    If a ticker is invalid, automatically tries common exchange suffixes
    (e.g., .L, .AQ, .PA) and returns the corrected ticker if found.

    Args:
        tickers: List of ticker symbols to validate

    Returns:
        Tuple of (valid_tickers, invalid_tickers)
        Note: valid_tickers may contain corrected tickers with exchange suffixes
    """
    # Common Yahoo Finance exchange suffixes
    EXCHANGE_SUFFIXES = [
        '.L',    # London Stock Exchange
        '.AQ',   # Euronext Amsterdam (Alternative)
        '.AS',   # Euronext Amsterdam
        '.PA',   # Euronext Paris
        '.DE',   # XETRA (Germany)
        '.F',    # Frankfurt Stock Exchange
        '.MI',   # Milan Stock Exchange
        '.SW',   # Swiss Exchange
        '.TO',   # Toronto Stock Exchange
        '.V',    # TSX Venture Exchange
        '.AX',   # Australian Securities Exchange
        '.NZ',   # New Zealand Exchange
        '.HK',   # Hong Kong Stock Exchange
        '.T',    # Tokyo Stock Exchange
        '.KS',   # Korea Stock Exchange
        '.KQ',   # KOSDAQ (Korea)
        '.SA',   # Brazil (Sao Paulo)
        '.MX',   # Mexico Stock Exchange
        '.ST',   # Stockholm (Nasdaq Nordic)
        '.OL',   # Oslo Stock Exchange
        '.CO',   # Copenhagen Stock Exchange
        '.HE',   # Helsinki Stock Exchange
        '.IC',   # Iceland Stock Exchange
        '.LS',   # Lisbon Stock Exchange
        '.AT',   # Athens Stock Exchange
        '.PR',   # Prague Stock Exchange
        '.WA',   # Warsaw Stock Exchange
        '.IS',   # Istanbul Stock Exchange
        '.JK',   # Jakarta Stock Exchange
        '.KL',   # Kuala Lumpur Stock Exchange
        '.SI',   # Singapore Exchange
        '.BK',   # Bangkok Stock Exchange
        '.TW',   # Taiwan Stock Exchange
    ]

    valid = []
    invalid = []

    for ticker in tickers:
        # First try the ticker as-is
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            if not hist.empty:
                valid.append(ticker)
                continue
        except:
            pass

        # If original ticker failed, try with common exchange suffixes
        found_valid = False
        for suffix in EXCHANGE_SUFFIXES:
            # Skip if ticker already has a suffix
            if '.' in ticker:
                break

            test_ticker = ticker + suffix
            try:
                stock = yf.Ticker(test_ticker)
                hist = stock.history(period='5d')
                if not hist.empty:
                    valid.append(test_ticker)
                    found_valid = True
                    # Show user the corrected ticker
                    st.info(f"Ticker '{ticker}' corrected to '{test_ticker}'")
                    break
            except:
                continue

        if not found_valid:
            invalid.append(ticker)

    return valid, invalid


def resample_data(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample price data to different frequency.

    Args:
        prices: DataFrame of price data
        frequency: 'D' (daily), 'W' (weekly), 'M' (monthly)

    Returns:
        Resampled price DataFrame
    """
    return prices.resample(frequency).last().dropna()


def get_benchmark_data(
    benchmark: str = 'SPY',
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Fetch benchmark index data for comparison.

    Args:
        benchmark: Benchmark ticker (default: SPY for S&P 500)
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with benchmark prices
    """
    return fetch_stock_data([benchmark], start_date, end_date)
