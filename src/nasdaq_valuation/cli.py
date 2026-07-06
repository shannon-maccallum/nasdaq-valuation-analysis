"""Command line entry points for the valuation analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from .modeling import (
    add_required_growth_rates,
    build_sector_benchmarks,
    evaluate_regressor,
    load_fundamentals,
    prepare_modeling_frame,
)


def run_analysis(input_path: Path, output_path: Path) -> None:
    """Run the sector valuation analysis and write ranked results."""

    fundamentals = load_fundamentals(input_path)
    valuation = build_sector_benchmarks(fundamentals)
    valuation = add_required_growth_rates(valuation)

    x, y = prepare_modeling_frame(valuation)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(x_train, y_train)
    metrics = evaluate_regressor(model, x_train, y_train, x_test, y_test)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    valuation.sort_values("Growth_Req_3yr", ascending=False).to_csv(output_path, index=False)

    print(f"Wrote ranked valuation results to {output_path}")
    print(
        "Gradient Boosting test metrics: "
        f"RMSE={metrics.validation_rmse:.3f}, "
        f"MAE={metrics.validation_mae:.3f}, "
        f"R2={metrics.validation_r2:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NASDAQ-100 valuation analysis.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/nasdaq100_fundamentals.csv"),
        help="Path to the fundamentals CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/valuation_rankings.csv"),
        help="Path for the generated ranking CSV.",
    )
    args = parser.parse_args()
    run_analysis(args.input, args.output)


if __name__ == "__main__":
    main()
