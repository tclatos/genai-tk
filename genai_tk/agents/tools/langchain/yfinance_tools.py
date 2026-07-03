"""Stock market data tools (Yahoo Finance) for LangChain-based agents.

Framework-agnostic — used by any LangChain agent type (react | deep | custom),
including LangChain deep agents (DeepAgents SDK) and DeerFlow (which forwards
LangChain tools directly).
"""

from datetime import date

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from loguru import logger


@tool
def get_stock_info(symbol: str, key: str) -> dict:
    """Retrieve specific information about a stock using its ticker symbol and key.

    This tool interfaces with Yahoo Finance to fetch various stock metrics
    including price, company information, and financial indicators.
    If asked generically for 'stock price', use currentPrice.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL' for Apple).
        key: Specific metric to retrieve from the stock info, e.g. currentPrice,
            marketCap, trailingPE, dividendYield, fiftyTwoWeekHigh, sector, industry.

    Returns:
        Dictionary containing the requested stock information.
    """
    data = yf.Ticker(symbol)
    stock_info = data.info
    return stock_info[key]


@tool
def get_historical_price(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch historical stock prices for a given symbol from start_date to end_date.

    Args:
        symbol: Stock ticker symbol.
        start_date: Start of the date range. Must be before end_date.
        end_date: End of the date range. Typically today unless specified otherwise.

    Returns:
        DataFrame with a ``Date`` column and a column named after the symbol.
    """
    try:
        data = yf.Ticker(symbol)
        hist = data.history(start=start_date, end=end_date)
        hist = hist.reset_index()
        hist[symbol] = hist["Close"]
        return hist[["Date", symbol]]
    except Exception as ex:
        logger.error("failed to call get_historical_price: {}", ex)
        return pd.DataFrame()
