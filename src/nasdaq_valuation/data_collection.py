"""Collect NASDAQ-100 fundamentals from Yahoo Finance through yfinance."""

from __future__ import annotations

from pathlib import Path
from time import sleep

import pandas as pd
import yfinance as yf

NASDAQ_100_TICKERS = [
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "NFLX",
    "PLTR",
    "COST",
    "AMD",
    "ASML",
    "CSCO",
    "AZN",
    "MU",
    "TMUS",
    "SHOP",
    "APP",
    "LIN",
    "PEP",
    "ISRG",
    "LRCX",
    "INTU",
    "PDD",
    "INTC",
    "AMAT",
    "QCOM",
    "ARM",
    "BKNG",
    "TXN",
    "AMGN",
    "ADBE",
    "GILD",
    "HON",
    "PANW",
    "KLAC",
    "ADP",
    "CMCSA",
    "MELI",
    "ADI",
    "CRWD",
    "VRTX",
    "SBUX",
    "CEG",
    "DASH",
    "MSTR",
    "ORLY",
    "CDNS",
    "SNPS",
    "MDLZ",
    "MAR",
    "CTAS",
    "FTNT",
    "ABNB",
    "PYPL",
    "REGN",
    "ADSK",
    "MNST",
    "ROP",
    "CSX",
    "NXPI",
    "WDAY",
    "AEP",
    "AXON",
    "CPRT",
    "PCAR",
    "PAYX",
    "FAST",
    "KDP",
    "ROST",
    "DDOG",
    "TEAM",
    "ZS",
    "IDXX",
    "LULU",
    "EXC",
    "BKR",
    "XEL",
    "FANG",
    "CCEP",
    "EA",
    "TTWO",
    "CHTR",
    "GEHC",
    "KHC",
    "ODFL",
    "DXCM",
    "CSGP",
    "MCHP",
    "TTD",
    "CDW",
    "ON",
    "BIIB",
    "GFS",
    "MRNA",
    "WBD",
    "MDB",
    "ILMN",
    "SIRI",
]


def fetch_fundamentals(tickers: list[str] | None = None, pause_seconds: float = 0.25) -> pd.DataFrame:
    """Fetch a fundamentals snapshot for the supplied tickers."""

    rows = []
    for ticker in tickers or NASDAQ_100_TICKERS:
        info = yf.Ticker(ticker).info
        rows.append(
            {
                "Ticker": ticker,
                "Company": info.get("longName"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "MarketCap": info.get("marketCap"),
                "Revenue": info.get("totalRevenue"),
                "NetIncome": info.get("netIncomeToCommon"),
                "FreeCashFlow": info.get("freeCashflow"),
                "TotalDebt": info.get("totalDebt"),
                "Cash": info.get("totalCash"),
                "PE_Ratio": info.get("trailingPE"),
                "EV_to_EBITDA": info.get("enterpriseToEbitda"),
                "Price_to_Sales": info.get("priceToSalesTrailing12Months"),
            }
        )
        sleep(pause_seconds)

    return pd.DataFrame(rows)


def save_fundamentals(output_path: str | Path, tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch fundamentals and write them to a CSV file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_fundamentals(tickers=tickers)
    df.to_csv(output, index=False)
    return df
