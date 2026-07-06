"""Core transformations for the NASDAQ-100 valuation project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED_COLUMNS = [
    "Ticker",
    "Company",
    "Sector",
    "Industry",
    "MarketCap",
    "Revenue",
    "NetIncome",
    "FreeCashFlow",
    "TotalDebt",
    "Cash",
    "PE_Ratio",
    "EV_to_EBITDA",
    "Price_to_Sales",
]

FEATURE_COLUMNS = [
    "EV_to_EBITDA",
    "Price_to_Sales",
    "MarketCap",
    "EV_avg",
    "PS_avg",
]


@dataclass(frozen=True)
class RegressionMetrics:
    """Train/validation metrics for a fitted regression model."""

    train_rmse: float
    validation_rmse: float
    train_mae: float
    validation_mae: float
    train_r2: float
    validation_r2: float


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    """Load and validate a fundamentals CSV."""

    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")
    return df


def clean_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill finance fields where missing values behave like absent balances."""

    cleaned = df.copy()
    for column in ["TotalDebt", "Cash", "PE_Ratio", "EV_to_EBITDA"]:
        cleaned[column] = cleaned[column].fillna(0)
    return cleaned


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log and signed-log features used in exploratory analysis."""

    transformed = df.copy()
    for column in ["PE_Ratio", "MarketCap", "Revenue", "TotalDebt", "Cash", "Price_to_Sales"]:
        transformed[f"{column}_log"] = np.log1p(transformed[column])

    for column in ["NetIncome", "FreeCashFlow", "EV_to_EBITDA"]:
        transformed[f"{column}_log"] = np.sign(transformed[column]) * np.log1p(
            np.abs(transformed[column])
        )
    return transformed


def build_sector_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each company with market-cap-weighted sector valuation benchmarks."""

    cols = ["Ticker", "Sector", "Industry", "EV_to_EBITDA", "Price_to_Sales", "MarketCap"]
    valuation = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    valuation["EV_weighted"] = valuation["EV_to_EBITDA"] * valuation["MarketCap"]
    valuation["PS_weighted"] = valuation["Price_to_Sales"] * valuation["MarketCap"]

    summary = (
        valuation.groupby("Sector")
        .agg(
            EV_avg=("EV_weighted", "sum"),
            PS_avg=("PS_weighted", "sum"),
            EV_median=("EV_to_EBITDA", "median"),
            PS_median=("Price_to_Sales", "median"),
            MC_median=("MarketCap", "median"),
            MC_total=("MarketCap", "sum"),
        )
        .reset_index()
    )

    summary["EV_avg"] /= summary["MC_total"]
    summary["PS_avg"] /= summary["MC_total"]

    valuation = valuation.merge(summary, on="Sector", how="left")
    valuation["EV_relative"] = valuation["EV_to_EBITDA"] / valuation["EV_avg"]
    valuation["PS_relative"] = valuation["Price_to_Sales"] / valuation["PS_avg"]
    valuation["ValuationScore"] = valuation[["EV_relative", "PS_relative"]].mean(axis=1)

    inverse_score = 1 / valuation["ValuationScore"]
    valuation["Undervalued_Z"] = (inverse_score - inverse_score.mean()) / inverse_score.std()
    return valuation


def add_required_growth_rates(df: pd.DataFrame, periods: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """Estimate annual revenue growth needed to revert to sector-average P/S."""

    result = df.copy()
    for period in periods:
        result[f"Growth_Req_{period}yr"] = (
            result["Price_to_Sales"] / result["PS_avg"]
        ) ** (1 / period) - 1
    return result


def prepare_modeling_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and the sector-relative undervaluation target."""

    modeling_df = (
        df[FEATURE_COLUMNS + ["Undervalued_Z"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    return modeling_df[FEATURE_COLUMNS], modeling_df["Undervalued_Z"]


def evaluate_regressor(model, x_train, y_train, x_validation, y_validation) -> RegressionMetrics:
    """Score a fitted regressor on train and validation data."""

    train_pred = model.predict(x_train)
    validation_pred = model.predict(x_validation)

    return RegressionMetrics(
        train_rmse=float(np.sqrt(mean_squared_error(y_train, train_pred))),
        validation_rmse=float(np.sqrt(mean_squared_error(y_validation, validation_pred))),
        train_mae=float(mean_absolute_error(y_train, train_pred)),
        validation_mae=float(mean_absolute_error(y_validation, validation_pred)),
        train_r2=float(r2_score(y_train, train_pred)),
        validation_r2=float(r2_score(y_validation, validation_pred)),
    )
