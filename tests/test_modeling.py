import numpy as np
import pandas as pd

from nasdaq_valuation.modeling import (
    add_required_growth_rates,
    build_sector_benchmarks,
    prepare_modeling_frame,
)


def sample_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
            "Company": ["Alpha", "Beta", "Gamma", "Delta"],
            "Sector": ["Technology", "Technology", "Retail", "Retail"],
            "Industry": ["Software", "Hardware", "Stores", "Stores"],
            "MarketCap": [100.0, 300.0, 200.0, 200.0],
            "Revenue": [10.0, 30.0, 20.0, 25.0],
            "NetIncome": [1.0, 3.0, 2.0, 2.5],
            "FreeCashFlow": [0.5, 2.5, 1.5, 2.0],
            "TotalDebt": [5.0, 10.0, 4.0, 6.0],
            "Cash": [2.0, 6.0, 3.0, 4.0],
            "PE_Ratio": [20.0, 30.0, 15.0, 18.0],
            "EV_to_EBITDA": [10.0, 20.0, 8.0, 12.0],
            "Price_to_Sales": [2.0, 6.0, 1.5, 2.5],
        }
    )


def test_sector_benchmarks_are_market_cap_weighted():
    result = build_sector_benchmarks(sample_fundamentals())
    tech = result[result["Sector"] == "Technology"].iloc[0]

    assert tech["EV_avg"] == 17.5
    assert tech["PS_avg"] == 5.0
    assert "Undervalued_Z" in result.columns


def test_required_growth_rates_use_sector_price_to_sales_average():
    result = add_required_growth_rates(build_sector_benchmarks(sample_fundamentals()))
    aaa = result[result["Ticker"] == "AAA"].iloc[0]

    assert np.isclose(aaa["Growth_Req_1yr"], -0.6)
    assert np.isclose(aaa["Growth_Req_3yr"], (2.0 / 5.0) ** (1 / 3) - 1)


def test_prepare_modeling_frame_removes_invalid_rows():
    result = build_sector_benchmarks(sample_fundamentals())
    result.loc[0, "EV_to_EBITDA"] = np.inf

    x, y = prepare_modeling_frame(result)

    assert len(x) == 3
    assert len(y) == 3
    assert list(x.columns) == ["EV_to_EBITDA", "Price_to_Sales", "MarketCap", "EV_avg", "PS_avg"]
