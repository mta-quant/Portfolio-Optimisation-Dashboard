"""
Ticker Search Module - Autocomplete and Fuzzy Search for Stock Tickers

Provides fuzzy search functionality for stock tickers and company names.
"""

import pandas as pd
import streamlit as st
from typing import List, Tuple, Optional


# Comprehensive list of popular stocks with their company names
POPULAR_TICKERS = {
    # Tech Giants
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. Class A',
    'GOOG': 'Alphabet Inc. Class C',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.',
    'AMD': 'Advanced Micro Devices Inc.',
    'INTC': 'Intel Corporation',
    'CRM': 'Salesforce Inc.',
    'ORCL': 'Oracle Corporation',
    'ADBE': 'Adobe Inc.',
    'NFLX': 'Netflix Inc.',
    'AVGO': 'Broadcom Inc.',
    'CSCO': 'Cisco Systems Inc.',
    'QCOM': 'QUALCOMM Incorporated',
    'TXN': 'Texas Instruments Incorporated',
    'IBM': 'International Business Machines',
    'UBER': 'Uber Technologies Inc.',
    'SHOP': 'Shopify Inc.',
    'SQ': 'Block Inc.',
    'PYPL': 'PayPal Holdings Inc.',
    'SNAP': 'Snap Inc.',
    'TWTR': 'Twitter Inc.',
    'SPOT': 'Spotify Technology S.A.',
    'ZOOM': 'Zoom Video Communications',
    'DOCU': 'DocuSign Inc.',
    'SNOW': 'Snowflake Inc.',
    'PLTR': 'Palantir Technologies Inc.',
    'NOW': 'ServiceNow Inc.',
    'PANW': 'Palo Alto Networks Inc.',
    'CRWD': 'CrowdStrike Holdings Inc.',
    'ZS': 'Zscaler Inc.',
    'DDOG': 'Datadog Inc.',
    'NET': 'Cloudflare Inc.',
    'TEAM': 'Atlassian Corporation',
    'WDAY': 'Workday Inc.',
    'OKTA': 'Okta Inc.',
    'ZM': 'Zoom Video Communications Inc.',
    'TWLO': 'Twilio Inc.',
    'RBLX': 'Roblox Corporation',
    'U': 'Unity Software Inc.',
    'PATH': 'UiPath Inc.',
    'S': 'SentinelOne Inc.',
    'FTNT': 'Fortinet Inc.',
    'SNPS': 'Synopsys Inc.',
    'CDNS': 'Cadence Design Systems Inc.',
    'ANSS': 'ANSYS Inc.',
    'INTU': 'Intuit Inc.',
    'ADSK': 'Autodesk Inc.',
    'EA': 'Electronic Arts Inc.',
    'ATVI': 'Activision Blizzard Inc.',
    'TTWO': 'Take-Two Interactive Software',
    'RBLX': 'Roblox Corporation',

    # Professional Services & Consulting
    'ACN': 'Accenture plc',
    'IBM': 'International Business Machines',
    'INFY': 'Infosys Limited',
    'WIT': 'Wipro Limited',

    # Financial Services
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corporation',
    'WFC': 'Wells Fargo & Company',
    'GS': 'The Goldman Sachs Group',
    'MS': 'Morgan Stanley',
    'C': 'Citigroup Inc.',
    'BLK': 'BlackRock Inc.',
    'SCHW': 'The Charles Schwab Corporation',
    'AXP': 'American Express Company',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Incorporated',
    'COIN': 'Coinbase Global Inc.',
    'SOFI': 'SoFi Technologies Inc.',
    'USB': 'U.S. Bancorp',
    'PNC': 'PNC Financial Services Group',
    'TFC': 'Truist Financial Corporation',
    'COF': 'Capital One Financial Corporation',
    'AFL': 'Aflac Incorporated',
    'MET': 'MetLife Inc.',
    'PRU': 'Prudential Financial Inc.',
    'AIG': 'American International Group',
    'TRV': 'The Travelers Companies Inc.',
    'ALL': 'The Allstate Corporation',
    'PGR': 'The Progressive Corporation',
    'CB': 'Chubb Limited',
    'MMC': 'Marsh & McLennan Companies',
    'AON': 'Aon plc',
    'SPGI': 'S&P Global Inc.',
    'MCO': 'Moody\'s Corporation',
    'CME': 'CME Group Inc.',
    'ICE': 'Intercontinental Exchange Inc.',
    'NDAQ': 'Nasdaq Inc.',

    # Healthcare & Pharma
    'JNJ': 'Johnson & Johnson',
    'UNH': 'UnitedHealth Group Incorporated',
    'PFE': 'Pfizer Inc.',
    'ABBV': 'AbbVie Inc.',
    'TMO': 'Thermo Fisher Scientific Inc.',
    'MRK': 'Merck & Co. Inc.',
    'LLY': 'Eli Lilly and Company',
    'ABT': 'Abbott Laboratories',
    'DHR': 'Danaher Corporation',
    'BMY': 'Bristol-Myers Squibb Company',
    'AMGN': 'Amgen Inc.',
    'GILD': 'Gilead Sciences Inc.',
    'CVS': 'CVS Health Corporation',
    'MRNA': 'Moderna Inc.',
    'BNTX': 'BioNTech SE',
    'VRTX': 'Vertex Pharmaceuticals Inc.',
    'REGN': 'Regeneron Pharmaceuticals Inc.',
    'ISRG': 'Intuitive Surgical Inc.',
    'SYK': 'Stryker Corporation',
    'BSX': 'Boston Scientific Corporation',
    'MDT': 'Medtronic plc',
    'CI': 'The Cigna Group',
    'HUM': 'Humana Inc.',
    'ANTM': 'Anthem Inc.',
    'ZTS': 'Zoetis Inc.',
    'IDXX': 'IDEXX Laboratories Inc.',
    'ALGN': 'Align Technology Inc.',
    'DXCM': 'DexCom Inc.',
    'ILMN': 'Illumina Inc.',
    'BIO': 'Bio-Rad Laboratories Inc.',
    'WAT': 'Waters Corporation',
    'A': 'Agilent Technologies Inc.',
    'PKI': 'PerkinElmer Inc.',

    # Consumer & Retail
    'WMT': 'Walmart Inc.',
    'HD': 'The Home Depot Inc.',
    'PG': 'The Procter & Gamble Company',
    'KO': 'The Coca-Cola Company',
    'PEP': 'PepsiCo Inc.',
    'COST': 'Costco Wholesale Corporation',
    'MCD': 'McDonald\'s Corporation',
    'NKE': 'NIKE Inc.',
    'SBUX': 'Starbucks Corporation',
    'DIS': 'The Walt Disney Company',
    'CMCSA': 'Comcast Corporation',
    'LOW': 'Lowe\'s Companies Inc.',
    'TGT': 'Target Corporation',
    'TJX': 'The TJX Companies Inc.',
    'ROST': 'Ross Stores Inc.',
    'DG': 'Dollar General Corporation',
    'DLTR': 'Dollar Tree Inc.',
    'KR': 'The Kroger Co.',
    'SYY': 'Sysco Corporation',
    'YUM': 'Yum! Brands Inc.',
    'CMG': 'Chipotle Mexican Grill Inc.',
    'QSR': 'Restaurant Brands International',
    'MO': 'Altria Group Inc.',
    'PM': 'Philip Morris International',
    'BUD': 'Anheuser-Busch InBev SA/NV',
    'TAP': 'Molson Coors Beverage Company',
    'STZ': 'Constellation Brands Inc.',
    'CL': 'Colgate-Palmive Company',
    'EL': 'The Estée Lauder Companies',
    'CLX': 'The Clorox Company',
    'KMB': 'Kimberly-Clark Corporation',
    'GIS': 'General Mills Inc.',
    'K': 'Kellogg Company',
    'HSY': 'The Hershey Company',
    'MDLZ': 'Mondelez International Inc.',
    'KHC': 'The Kraft Heinz Company',
    'CPB': 'Campbell Soup Company',
    'CAG': 'Conagra Brands Inc.',
    'HRL': 'Hormel Foods Corporation',
    'TSN': 'Tyson Foods Inc.',
    'MNST': 'Monster Beverage Corporation',
    'KDP': 'Keurig Dr Pepper Inc.',

    # Energy
    'CVX': 'Chevron Corporation',
    'XOM': 'Exxon Mobil Corporation',
    'COP': 'ConocoPhillips',
    'SLB': 'Schlumberger N.V.',
    'EOG': 'EOG Resources Inc.',
    'PXD': 'Pioneer Natural Resources Company',
    'MPC': 'Marathon Petroleum Corporation',
    'PSX': 'Phillips 66',
    'VLO': 'Valero Energy Corporation',
    'OXY': 'Occidental Petroleum Corporation',
    'HAL': 'Halliburton Company',
    'BKR': 'Baker Hughes Company',
    'FANG': 'Diamondback Energy Inc.',
    'DVN': 'Devon Energy Corporation',
    'HES': 'Hess Corporation',

    # Industrial & Manufacturing
    'BA': 'The Boeing Company',
    'CAT': 'Caterpillar Inc.',
    'GE': 'General Electric Company',
    'MMM': '3M Company',
    'HON': 'Honeywell International Inc.',
    'LMT': 'Lockheed Martin Corporation',
    'RTX': 'Raytheon Technologies Corporation',
    'UPS': 'United Parcel Service Inc.',
    'FDX': 'FedEx Corporation',
    'DE': 'Deere & Company',
    'EMR': 'Emerson Electric Co.',
    'ETN': 'Eaton Corporation plc',
    'PH': 'Parker-Hannifin Corporation',
    'ITW': 'Illinois Tool Works Inc.',
    'CMI': 'Cummins Inc.',
    'PCAR': 'PACCAR Inc.',
    'NSC': 'Norfolk Southern Corporation',
    'UNP': 'Union Pacific Corporation',
    'CSX': 'CSX Corporation',
    'DAL': 'Delta Air Lines Inc.',
    'UAL': 'United Airlines Holdings Inc.',
    'AAL': 'American Airlines Group Inc.',
    'LUV': 'Southwest Airlines Co.',
    'ALK': 'Alaska Air Group Inc.',
    'WM': 'Waste Management Inc.',
    'RSG': 'Republic Services Inc.',
    'URI': 'United Rentals Inc.',
    'JCI': 'Johnson Controls International',
    'CARR': 'Carrier Global Corporation',
    'OTIS': 'Otis Worldwide Corporation',
    'ROK': 'Rockwell Automation Inc.',
    'DOV': 'Dover Corporation',
    'FTV': 'Fortive Corporation',
    'IR': 'Ingersoll Rand Inc.',
    'PWR': 'Quanta Services Inc.',
    'EME': 'EMCOR Group Inc.',

    # Utilities
    'NEE': 'NextEra Energy Inc.',
    'DUK': 'Duke Energy Corporation',
    'SO': 'The Southern Company',
    'D': 'Dominion Energy Inc.',
    'AEP': 'American Electric Power Company',
    'EXC': 'Exelon Corporation',
    'SRE': 'Sempra Energy',
    'XEL': 'Xcel Energy Inc.',
    'ED': 'Consolidated Edison Inc.',
    'ES': 'Eversource Energy',
    'AWK': 'American Water Works Company',
    'PEG': 'Public Service Enterprise Group',

    # Telecommunications
    'T': 'AT&T Inc.',
    'VZ': 'Verizon Communications Inc.',
    'TMUS': 'T-Mobile US Inc.',
    'CHTR': 'Charter Communications Inc.',

    # ETFs and Index Funds
    'SPY': 'SPDR S&P 500 ETF Trust',
    'QQQ': 'Invesco QQQ Trust',
    'IWM': 'iShares Russell 2000 ETF',
    'DIA': 'SPDR Dow Jones Industrial Average ETF',
    'VOO': 'Vanguard S&P 500 ETF',
    'VTI': 'Vanguard Total Stock Market ETF',
    'VEA': 'Vanguard FTSE Developed Markets ETF',
    'VWO': 'Vanguard FTSE Emerging Markets ETF',
    'AGG': 'iShares Core U.S. Aggregate Bond ETF',
    'BND': 'Vanguard Total Bond Market ETF',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'GLD': 'SPDR Gold Shares',
    'SLV': 'iShares Silver Trust',
    'VNQ': 'Vanguard Real Estate ETF',
    'XLE': 'Energy Select Sector SPDR Fund',
    'XLF': 'Financial Select Sector SPDR Fund',
    'XLK': 'Technology Select Sector SPDR Fund',
    'XLV': 'Health Care Select Sector SPDR Fund',
    'XLY': 'Consumer Discretionary Select Sector SPDR Fund',
    'XLP': 'Consumer Staples Select Sector SPDR Fund',
    'XLI': 'Industrial Select Sector SPDR Fund',
    'XLU': 'Utilities Select Sector SPDR Fund',
    'XLB': 'Materials Select Sector SPDR Fund',
    'XLRE': 'Real Estate Select Sector SPDR Fund',
    'DBC': 'Invesco DB Commodity Index Tracking Fund',
    'USO': 'United States Oil Fund',
    'EEM': 'iShares MSCI Emerging Markets ETF',
    'EFA': 'iShares MSCI EAFE ETF',
    'IEMG': 'iShares Core MSCI Emerging Markets ETF',
    'VIG': 'Vanguard Dividend Appreciation ETF',
    'SCHD': 'Schwab U.S. Dividend Equity ETF',

    # Semiconductors & Hardware
    'TSM': 'Taiwan Semiconductor Manufacturing',
    'ASML': 'ASML Holding N.V.',
    'MU': 'Micron Technology Inc.',
    'AMAT': 'Applied Materials Inc.',
    'LRCX': 'Lam Research Corporation',
    'KLAC': 'KLA Corporation',
    'MRVL': 'Marvell Technology Inc.',

    # Automotive
    'F': 'Ford Motor Company',
    'GM': 'General Motors Company',
    'RIVN': 'Rivian Automotive Inc.',
    'LCID': 'Lucid Group Inc.',
    'NIO': 'NIO Inc.',
    'STLA': 'Stellantis N.V.',
    'TM': 'Toyota Motor Corporation',
    'HMC': 'Honda Motor Co. Ltd.',
    'RACE': 'Ferrari N.V.',

    # Materials & Chemicals
    'LIN': 'Linde plc',
    'APD': 'Air Products and Chemicals Inc.',
    'ECL': 'Ecolab Inc.',
    'SHW': 'The Sherwin-Williams Company',
    'DD': 'DuPont de Nemours Inc.',
    'DOW': 'Dow Inc.',
    'PPG': 'PPG Industries Inc.',
    'NEM': 'Newmont Corporation',
    'FCX': 'Freeport-McMoRan Inc.',
    'NUE': 'Nucor Corporation',
    'STLD': 'Steel Dynamics Inc.',
    'VMC': 'Vulcan Materials Company',
    'MLM': 'Martin Marietta Materials Inc.',
    'ALB': 'Albemarle Corporation',
    'CE': 'Celanese Corporation',
    'EMN': 'Eastman Chemical Company',
    'IFF': 'International Flavors & Fragrances',
    'FMC': 'FMC Corporation',
    'MOS': 'The Mosaic Company',
    'CF': 'CF Industries Holdings Inc.',

    # Real Estate & REITs
    'AMT': 'American Tower Corporation',
    'PLD': 'Prologis Inc.',
    'CCI': 'Crown Castle Inc.',
    'EQIX': 'Equinix Inc.',
    'SPG': 'Simon Property Group Inc.',
    'PSA': 'Public Storage',
    'O': 'Realty Income Corporation',
    'WELL': 'Welltower Inc.',
    'DLR': 'Digital Realty Trust Inc.',
    'AVB': 'AvalonBay Communities Inc.',
    'EQR': 'Equity Residential',
    'VTR': 'Ventas Inc.',
    'ARE': 'Alexandria Real Estate Equities',
    'SBAC': 'SBA Communications Corporation',
    'CBRE': 'CBRE Group Inc.',
    'HST': 'Host Hotels & Resorts Inc.',
    'MAA': 'Mid-America Apartment Communities',
    'KIM': 'Kimco Realty Corporation',
    'REG': 'Regency Centers Corporation',

    # Media & Entertainment
    'WBD': 'Warner Bros. Discovery Inc.',
    'PARA': 'Paramount Global',
    'FOX': 'Fox Corporation',
    'NFLX': 'Netflix Inc.',
    'ROKU': 'Roku Inc.',
    'LYV': 'Live Nation Entertainment Inc.',

    # Retail Specialty
    'LULU': 'Lululemon Athletica Inc.',
    'DECK': 'Deckers Outdoor Corporation',
    'CROX': 'Crocs Inc.',
    'BBY': 'Best Buy Co. Inc.',
    'ULTA': 'Ulta Beauty Inc.',
    'AZO': 'AutoZone Inc.',
    'ORLY': 'O\'Reilly Automotive Inc.',
    'AAP': 'Advance Auto Parts Inc.',
    'EBAY': 'eBay Inc.',
    'ETSY': 'Etsy Inc.',
    'W': 'Wayfair Inc.',
    'CHWY': 'Chewy Inc.',

    # International Stocks
    'BABA': 'Alibaba Group Holding Limited',
    'TSM': 'Taiwan Semiconductor Manufacturing',
    'ASML': 'ASML Holding N.V.',
    'SAP': 'SAP SE',
    'NVO': 'Novo Nordisk A/S',
    'UL': 'Unilever PLC',
    'SNY': 'Sanofi',
    'DEO': 'Diageo plc',
    'BHP': 'BHP Group Limited',
    'RIO': 'Rio Tinto plc',
    'BP': 'BP p.l.c.',
    'SHEL': 'Shell plc',
    'TTE': 'TotalEnergies SE',
    'SONY': 'Sony Group Corporation',
    'BIDU': 'Baidu Inc.',
    'JD': 'JD.com Inc.',
    'PDD': 'PDD Holdings Inc.',
    'NTES': 'NetEase Inc.',
    'SE': 'Sea Limited',
    'GRAB': 'Grab Holdings Limited',
    'MELI': 'MercadoLibre Inc.',
    'NU': 'Nu Holdings Ltd.',

    # Cryptocurrency-related
    'MSTR': 'MicroStrategy Incorporated',
    'RIOT': 'Riot Platforms Inc.',
    'MARA': 'Marathon Digital Holdings Inc.',

    # Others
    'BRK.B': 'Berkshire Hathaway Inc. Class B',
    'BRK.A': 'Berkshire Hathaway Inc. Class A',
}


@st.cache_data(ttl=3600)
def get_ticker_list() -> List[Tuple[str, str]]:
    """
    Get list of tickers with company names.

    Returns:
        List of tuples (ticker, company_name)
    """
    return [(ticker, name) for ticker, name in POPULAR_TICKERS.items()]


def search_tickers(search_term: str) -> List[Tuple[str, str]]:
    """
    Search for tickers matching the search term.
    Performs fuzzy matching on both ticker symbols and company names.

    Args:
        search_term: User's search input

    Returns:
        List of matching (ticker, company_name) tuples
    """
    if not search_term:
        return []

    search_term = search_term.upper().strip()
    matches = []

    for ticker, name in POPULAR_TICKERS.items():
        # Check if search term matches ticker symbol (prefix match)
        if ticker.startswith(search_term):
            matches.append((ticker, name, 0))  # Priority 0 (highest)
        # Check if search term is in ticker symbol
        elif search_term in ticker:
            matches.append((ticker, name, 1))  # Priority 1
        # Check if search term is in company name (case insensitive)
        elif search_term.lower() in name.lower():
            matches.append((ticker, name, 2))  # Priority 2

    # Sort by priority, then alphabetically
    matches.sort(key=lambda x: (x[2], x[0]))

    # Return without priority field
    return [(ticker, name) for ticker, name, _ in matches]


def format_ticker_option(ticker: str, name: str) -> str:
    """
    Format ticker and name for display.

    Args:
        ticker: Stock ticker symbol
        name: Company name

    Returns:
        Formatted string
    """
    # Truncate long names
    max_name_length = 40
    if len(name) > max_name_length:
        name = name[:max_name_length-3] + "..."

    return f"{ticker} - {name}"


def search_callback(search_term: str, **kwargs) -> List[str]:
    """
    Callback function for streamlit-searchbox component.

    Args:
        search_term: User's search input
        **kwargs: Additional arguments from st_searchbox (ignored)

    Returns:
        List of formatted ticker options
    """
    if not search_term or len(search_term) < 1:
        # Return top 10 popular tickers by default
        popular = [
            ('AAPL', 'Apple Inc.'),
            ('MSFT', 'Microsoft Corporation'),
            ('GOOGL', 'Alphabet Inc.'),
            ('AMZN', 'Amazon.com Inc.'),
            ('NVDA', 'NVIDIA Corporation'),
            ('TSLA', 'Tesla Inc.'),
            ('META', 'Meta Platforms Inc.'),
            ('SPY', 'SPDR S&P 500 ETF Trust'),
            ('QQQ', 'Invesco QQQ Trust'),
            ('VOO', 'Vanguard S&P 500 ETF'),
        ]
        return [format_ticker_option(t, n) for t, n in popular]

    matches = search_tickers(search_term)

    # Limit to top 15 results
    matches = matches[:15]

    results = [format_ticker_option(ticker, name) for ticker, name in matches]

    # Always add the raw search term as an option if it's valid-looking (alphanumeric)
    if search_term and search_term.replace('.', '').replace('-', '').isalnum():
        # Add as first option if no matches, or last option if there are matches
        custom_option = f"{search_term.upper()} - Add Custom Ticker"
        if not results:
            results = [custom_option]
        else:
            # Check if search term isn't already in results as exact match
            if not any(search_term.upper() == r.split(' - ')[0] for r in results):
                results.append(custom_option)

    return results


def extract_ticker_from_selection(selection: str) -> Optional[str]:
    """
    Extract ticker symbol from formatted selection string.

    Args:
        selection: Formatted string like "AAPL - Apple Inc."

    Returns:
        Ticker symbol or None
    """
    if not selection:
        return None

    # Extract ticker (everything before " - ")
    if ' - ' in selection:
        return selection.split(' - ')[0].strip()

    return selection.strip().upper()


def get_company_name(ticker: str) -> str:
    """
    Get company name for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Company name or ticker if not found
    """
    return POPULAR_TICKERS.get(ticker.upper(), ticker)
