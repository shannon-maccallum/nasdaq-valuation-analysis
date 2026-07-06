# NASDAQ-100 Valuation Analysis

This project analyzes NASDAQ-100 company fundamentals to compare valuation across sectors and estimate how much revenue growth companies would need to justify current price-to-sales multiples.

The work combines exploratory data analysis, sector-relative valuation benchmarks, PCA, and regression model comparison. It is designed as a portfolio-ready data science project: the notebooks preserve the original analysis narrative, while the `src/` package makes the core logic reusable and testable.

## Project Highlights

- Built a reproducible fundamentals dataset from Yahoo Finance via `yfinance`
- Compared companies against market-cap-weighted sector valuation benchmarks
- Created an `Undervalued_Z` score to identify companies trading at discounts or premiums relative to sector peers
- Estimated 1-year, 3-year, and 5-year revenue growth required for valuation multiples to revert to sector averages
- Benchmarked linear regression, ridge regression, random forest, and gradient boosting models
- Packaged the core transformations into tested Python modules

## Repository Structure

```text
.
├── data/
│   └── raw/
│       └── nasdaq100_fundamentals.csv
├── notebooks/
│   ├── 01_data_collection.ipynb
│   └── 02_valuation_analysis.ipynb
├── reports/
│   └── figures/
│       └── project_screenshot.png
├── src/
│   └── nasdaq_valuation/
│       ├── data_collection.py
│       ├── modeling.py
│       └── cli.py
├── tests/
│   └── test_modeling.py
├── pyproject.toml
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Analysis

Run the reusable pipeline:

```bash
nasdaq-valuation
```

This reads `data/raw/nasdaq100_fundamentals.csv` and writes ranked results to:

```text
data/processed/valuation_rankings.csv
```

You can also open the notebooks for the full narrative workflow:

```bash
jupyter notebook notebooks/02_valuation_analysis.ipynb
```

## Tests

```bash
pytest
```

## Methods

The analysis uses sector-level benchmarks because valuation ratios are not directly comparable across very different types of businesses. For each sector, EV/EBITDA and price-to-sales averages are weighted by market capitalization, giving larger companies more influence on the sector benchmark.

Each company is scored against its sector average using:

- `EV_relative = EV_to_EBITDA / sector_EV_avg`
- `PS_relative = Price_to_Sales / sector_PS_avg`
- `ValuationScore = average(EV_relative, PS_relative)`
- `Undervalued_Z = standardized inverse valuation score`

Required growth is estimated as the annualized growth rate needed for a company's current price-to-sales ratio to converge back to its sector average.

## Data Notes

The included CSV is a point-in-time snapshot of NASDAQ-100 fundamentals collected from Yahoo Finance using `yfinance`. Market data changes frequently, so rerunning the collection notebook or script may produce different values.

This project is for educational and analytical purposes only. It is not financial advice.
