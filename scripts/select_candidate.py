"""후보군에서 선택한 ETF와 비중으로 candidate 테이블을 만든다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from daily_backtest import END_DATE, INITIAL_CAPITAL_KRW, INITIAL_INDEX, START_DATE, align_weekday_prices, business_calendar

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.set_index("Date").sort_index()
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_config(path: Path) -> tuple[dict[str, float], set[str]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    weights = {str(k): float(v) for k, v in config["weights"].items()}
    usd_etfs = {str(name) for name in config.get("usd_etfs", [])}
    if not weights or any(value < 0 for value in weights.values()):
        raise ValueError("weights는 0 이상인 숫자여야 합니다.")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-8):
        raise ValueError(f"weights 합계가 1이 아닙니다: {sum(weights.values()):.12f}")
    return weights, usd_etfs


def build_candidate_table(
    universe: pd.DataFrame,
    weights: dict[str, float],
    usd_etfs: set[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    required = set(weights) | {"USDKRW"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"후보군 CSV에 필요한 열이 없습니다: {sorted(missing)}")
    unknown_usd = usd_etfs - set(weights)
    if unknown_usd:
        raise ValueError(f"usd_etfs에 선택되지 않은 종목이 있습니다: {sorted(unknown_usd)}")

    calendar = business_calendar(start_date, end_date)
    prices = align_weekday_prices(universe[list(weights) + ["USDKRW"]], calendar)
    returns = prices.pct_change()
    names = list(weights)
    target = np.array([weights[name] for name in names], dtype=float)
    sleeve_values = INITIAL_CAPITAL_KRW * target
    total_value = float(INITIAL_CAPITAL_KRW)
    index_value = float(INITIAL_INDEX)
    rows: list[dict[str, float | pd.Timestamp]] = []

    for i, date in enumerate(calendar):
        if i == 0:
            daily_return = np.nan
        else:
            asset_returns = returns.loc[date, names].to_numpy(dtype=float).copy()
            fx_return = float(returns.loc[date, "USDKRW"])
            for j, name in enumerate(names):
                if name in usd_etfs:
                    asset_returns[j] = (1.0 + asset_returns[j]) * (1.0 + fx_return) - 1.0
            sleeve_values *= 1.0 + asset_returns
            total_value = float(sleeve_values.sum())
            index_value = total_value / INITIAL_CAPITAL_KRW * INITIAL_INDEX
            daily_return = index_value / float(rows[-1]["Candidate_Price"]) - 1.0

        if date.weekday() == 4:
            sleeve_values = total_value * target

        row: dict[str, float | pd.Timestamp] = {
            "Date": date,
            "Candidate_Price": index_value,
            "Candidate_Return": daily_return,
            "USDKRW_Close": float(prices.loc[date, "USDKRW"]),
        }
        for name in names:
            row[f"{name}_Close"] = float(prices.loc[date, name])
        rows.append(row)

    return pd.DataFrame(rows).set_index("Date")


def main() -> None:
    parser = argparse.ArgumentParser(description="후보군에서 candidate와 비중을 선택")
    parser.add_argument(
        "--universe",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "candidate_universe_daily.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "candidate_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "candidate_table_daily.csv",
    )
    args = parser.parse_args()

    weights, usd_etfs = load_config(args.config)
    universe = read_table(args.universe)
    result = build_candidate_table(universe, weights, usd_etfs, START_DATE, END_DATE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.reset_index().to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {args.output} ({len(result):,}행; 선택 종목: {', '.join(weights)})")


if __name__ == "__main__":
    main()
