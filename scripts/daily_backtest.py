"""Daily backtest for the weekly-rebalanced benchmark.

The strategy is rebalanced at the Friday close (or the last available
weekday price when Friday is a holiday).  The new weights are applied from
the next weekday return, which avoids using the Friday return to trade at the
same close.

Running this file writes:
    table1_daily_aligned.csv
    daily_metrics.csv
    daily_backtest_all.csv

The market CSV is optional.  When it is omitted, KOSPI200, S&P 500 and
USD/KRW are downloaded with FinanceDataReader, as in the original notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


START_DATE = "2023-09-22"
END_DATE = "2026-09-18"
INITIAL_INDEX = 100.0
INITIAL_CAPITAL_KRW = 10_000_000_000
USD_FUTURES_MULTIPLIER = 10_000
RISK_FREE_WEEKLY = 0.00035
PROJECT_ROOT = Path(__file__).resolve().parents[1]

W_KOSPI = 0.40
W_SP500_HEDGED = 0.30
W_SP500_UNHEDGED = 0.30

CANDIDATE1_WEIGHTS = {
    "KODEX200": 0.15,
    "USMV": 0.30,
    "QUAL": 0.15,
    "IEF": 0.20,
    "SGOV": 0.10,
    "KODEX_GOLD_H": 0.10,
}

MARKET_COLUMNS = ["KOSPI200", "SP500", "USDKRW"]
CANDIDATE_COLUMNS = list(CANDIDATE1_WEIGHTS)
TABLE1_CLOSE_COLUMNS = [
    "KOSPI200_Close",
    "SP500_Close_USD",
    "USDKRW_Close",
    "USD_Futures_Settlement_Close",
    *[f"{name}_Close" for name in CANDIDATE_COLUMNS],
]


def business_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Return weekdays only; exchange holidays are filled later from prior data."""
    return pd.date_range(start_date, end_date, freq="B")


def read_candidate_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "Date"})
    required = set(CANDIDATE_COLUMNS + ["USDKRW"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate CSV missing columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.set_index("Date").sort_index()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[CANDIDATE_COLUMNS + ["USDKRW"]]


def read_market_prices(
    path: Path | None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Read a local market CSV or download the same series as the notebook."""
    if path is not None:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.rename(columns={df.columns[0]: "Date"})
        missing = set(MARKET_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Market CSV missing columns: {sorted(missing)}")
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        df = df.set_index("Date").sort_index()
        for column in MARKET_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df[MARKET_COLUMNS]

    try:
        import FinanceDataReader as fdr
    except ImportError as exc:
        raise ImportError(
            "FinanceDataReader가 없습니다. requirements.txt를 설치하거나 "
            "--market-csv Date,KOSPI200,SP500,USDKRW 파일을 지정하세요."
        ) from exc

    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    fetch_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def read_close(symbols: list[str]) -> pd.Series:
        last_error: Exception | None = None
        for symbol in symbols:
            try:
                data = fdr.DataReader(symbol, fetch_start, fetch_end)
                if data is not None and not data.empty and "Close" in data:
                    series = pd.to_numeric(data["Close"], errors="coerce")
                    series.index = pd.to_datetime(series.index)
                    return series
            except Exception as exc:  # try the next symbol
                last_error = exc
        raise RuntimeError(f"시장가격을 읽지 못했습니다: {symbols}; {last_error}")

    return pd.concat(
        [
            read_close(["KS200"]).rename("KOSPI200"),
            read_close(["S&P500", "US500"]).rename("SP500"),
            read_close(["USD/KRW"]).rename("USDKRW"),
        ],
        axis=1,
        sort=True,
    )


def align_weekday_prices(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Remove weekends and carry the last valid price across weekday holidays."""
    df = df.loc[df.index.dayofweek < 5].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Keep lookback observations before the requested start date so a first-day
    # holiday can use the prior trading day's price.
    expanded_index = df.index.union(calendar).sort_values()
    aligned = df.reindex(expanded_index).ffill().reindex(calendar)
    if aligned.isna().any().any():
        missing = aligned.columns[aligned.isna().any()].tolist()
        raise ValueError(f"No prior valid price exists for: {missing}")
    return aligned


def load_futures(path: Path, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Create daily futures settlement changes; non-trading weekdays have zero change."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Date", "종목코드", "종가", "대비", "정산가"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Futures CSV missing columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    for column in ["종가", "대비", "정산가"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df = df.sort_values("Date")
    if df["Date"].duplicated().any():
        raise ValueError("Futures CSV has duplicate dates.")
    df["Previous_Settlement_Inferred"] = df["종가"] - df["대비"]
    df["Settlement_Change"] = df["정산가"] - df["Previous_Settlement_Inferred"]
    df["Contract_Changed"] = df["종목코드"].ne(df["종목코드"].shift()).astype(int)
    df.loc[df.index[0], "Contract_Changed"] = 0
    same_contract = df["종목코드"].eq(df["종목코드"].shift())
    direct_change = df["정산가"].diff()
    error = (df.loc[same_contract, "Settlement_Change"] - direct_change.loc[same_contract]).abs()
    if error.max(skipna=True) > 1e-8:
        raise ValueError("Futures settlement-change validation failed.")

    daily = df.groupby("Date", as_index=True).agg(
        USD_Futures_Settlement_Close=("정산가", "last"),
        Futures_Settlement_Change=("Settlement_Change", "sum"),
        Roll_Events=("Contract_Changed", "sum"),
    )
    daily = daily.loc[daily.index.dayofweek < 5]
    expanded_index = daily.index.union(calendar).sort_values()
    aligned = daily.reindex(expanded_index).sort_index()
    aligned["USD_Futures_Settlement_Close"] = aligned["USD_Futures_Settlement_Close"].ffill()
    aligned = aligned.reindex(calendar)
    aligned["Futures_Settlement_Change"] = aligned["Futures_Settlement_Change"].fillna(0.0)
    aligned["Roll_Events"] = aligned["Roll_Events"].fillna(0).astype(int)
    if aligned["USD_Futures_Settlement_Close"].isna().any():
        raise ValueError("No prior futures settlement exists at the start date.")
    return aligned


def round_half_up_positive(value: float) -> int:
    return int(np.floor(value + 0.5))


def hedge_contracts(hedged_sleeve_value: float, usdkrw: float) -> int:
    raw = hedged_sleeve_value / usdkrw / USD_FUTURES_MULTIPLIER
    return -round_half_up_positive(raw)


def build_daily_backtest(
    candidate: pd.DataFrame,
    market: pd.DataFrame,
    futures: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    calendar = business_calendar(start_date, end_date)
    # USDKRW is present in both sources.  Use the market series once so that
    # the benchmark and the USD-listed ETF conversions share one FX series.
    candidate_assets = candidate.drop(columns=["USDKRW"], errors="ignore")
    prices = align_weekday_prices(pd.concat([market, candidate_assets], axis=1), calendar)
    futures = futures.reindex(calendar)

    returns = prices.pct_change()
    sleeve_values = INITIAL_CAPITAL_KRW * np.array([W_KOSPI, W_SP500_HEDGED, W_SP500_UNHEDGED])
    portfolio_value = float(INITIAL_CAPITAL_KRW)
    contracts = hedge_contracts(sleeve_values[1], prices.iloc[0]["USDKRW"])
    component_values = np.array([INITIAL_INDEX, INITIAL_INDEX, INITIAL_INDEX])
    candidate_sleeves = INITIAL_CAPITAL_KRW * np.array(list(CANDIDATE1_WEIGHTS.values()))
    candidate_value = float(INITIAL_CAPITAL_KRW)
    portfolio_index = INITIAL_INDEX
    candidate_index = INITIAL_INDEX
    daily_rows: list[dict[str, float | int | pd.Timestamp]] = []

    for i, date in enumerate(calendar):
        if i == 0:
            r_kospi = r_unhedged = r_hedged = r_composite = np.nan
            candidate_return = np.nan
        else:
            r_kospi = returns.loc[date, "KOSPI200"]
            r_sp500_usd = returns.loc[date, "SP500"]
            r_fx = returns.loc[date, "USDKRW"]
            r_unhedged = (1.0 + r_sp500_usd) * (1.0 + r_fx) - 1.0
            contracts_used = contracts
            futures_pnl = contracts_used * USD_FUTURES_MULTIPLIER * futures.loc[date, "Futures_Settlement_Change"]
            hedged_pnl = sleeve_values[1] * r_unhedged + futures_pnl
            r_hedged = hedged_pnl / sleeve_values[1]
            sleeve_values *= 1.0 + np.array([r_kospi, r_hedged, r_unhedged])
            portfolio_value = float(sleeve_values.sum())
            portfolio_index = portfolio_value / INITIAL_CAPITAL_KRW * INITIAL_INDEX
            r_composite = portfolio_index / float(daily_rows[-1]["Benchmark_Composite_Price"]) - 1.0
            component_values *= 1.0 + np.array([r_kospi, r_hedged, r_unhedged])

            asset_returns = []
            for asset in CANDIDATE_COLUMNS:
                asset_return = returns.loc[date, asset]
                if asset in {"USMV", "QUAL", "IEF", "SGOV"}:
                    asset_return = (1.0 + asset_return) * (1.0 + r_fx) - 1.0
                asset_returns.append(asset_return)
            candidate_sleeves *= 1.0 + np.array(asset_returns)
            candidate_value = float(candidate_sleeves.sum())
            candidate_index = candidate_value / INITIAL_CAPITAL_KRW * INITIAL_INDEX
            candidate_return = candidate_index / float(daily_rows[-1]["Candidate1_Price"]) - 1.0

        if date.weekday() == 4:
            sleeve_values = portfolio_value * np.array([W_KOSPI, W_SP500_HEDGED, W_SP500_UNHEDGED])
            contracts = hedge_contracts(sleeve_values[1], prices.loc[date, "USDKRW"])
            candidate_sleeves = candidate_value * np.array(list(CANDIDATE1_WEIGHTS.values()))

        row = {
            "Date": date,
            "Benchmark1_Price": component_values[0],
            "Benchmark2_Hedged_Price": component_values[1],
            "Benchmark3_Unhedged_Price": component_values[2],
            "Benchmark_Composite_Price": portfolio_index,
            "Candidate1_Price": candidate_index,
            "Candidate1_Return": candidate_return,
            "KOSPI200_Return": r_kospi,
            "SP500_Hedged_Return": r_hedged,
            "SP500_Unhedged_Return": r_unhedged,
            "Benchmark_Composite_Return": r_composite,
            "USD_Futures_Contracts": contracts,
            "Futures_Settlement_Change": futures.loc[date, "Futures_Settlement_Change"],
            "Roll_Events": futures.loc[date, "Roll_Events"],
        }
        daily_rows.append(row)

    result = pd.DataFrame(daily_rows).set_index("Date")
    for column in ["KOSPI200", "KODEX200", *CANDIDATE_COLUMNS, "SP500", "USDKRW"]:
        if column in prices:
            name = {"SP500": "SP500_Close_USD", "USDKRW": "USDKRW_Close"}.get(column, f"{column}_Close")
            result[name] = prices[column]
    result["USD_Futures_Settlement_Close"] = futures["USD_Futures_Settlement_Close"]
    return result


def calculate_metrics(returns: pd.Series) -> dict[str, float | int]:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2:
        raise ValueError("At least two daily returns are required.")
    cumulative = float((1.0 + returns).prod() - 1.0)
    annualized_return = float((1.0 + cumulative) ** (252.0 / len(returns)) - 1.0)
    volatility = float(returns.std(ddof=1))
    daily_rf = (1.0 + RISK_FREE_WEEKLY) ** (1.0 / 5.0) - 1.0
    sharpe = float((returns.mean() - daily_rf) / volatility * np.sqrt(252.0)) if volatility else np.nan
    var_95 = float(-returns.quantile(0.05))
    tail = returns[returns <= -var_95]
    cvar_95 = float(-tail.mean()) if len(tail) else np.nan
    return {
        "Observations": int(len(returns)),
        "Cumulative Return": cumulative,
        "Annualized Return": annualized_return,
        "Annualized Volatility": volatility * np.sqrt(252.0),
        "Annualized Sharpe Ratio": sharpe,
        "Maximum Drawdown": float((1.0 + returns).cumprod().div((1.0 + returns).cumprod().cummax()).sub(1.0).min()),
        "1-day 95% VaR": var_95,
        "1-day 95% CVaR": cvar_95,
        "1-week 95% VaR Proxy": var_95 * np.sqrt(5.0),
    }


def write_outputs(result: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table1 = result[[
        "Benchmark1_Price",
        "Benchmark2_Hedged_Price",
        "Benchmark3_Unhedged_Price",
        "Benchmark_Composite_Price",
        "Candidate1_Price",
        *TABLE1_CLOSE_COLUMNS,
    ]].reset_index()
    table1.to_csv(output_dir / "table1_daily_aligned.csv", index=False, encoding="utf-8-sig")
    result.reset_index().to_csv(output_dir / "daily_backtest_all.csv", index=False, encoding="utf-8-sig")

    metric_series = {
        "Benchmark 1 - KOSPI200": result["Benchmark1_Price"].pct_change(),
        "Benchmark 2 - S&P500 Hedged": result["Benchmark2_Hedged_Price"].pct_change(),
        "Benchmark 3 - S&P500 Unhedged": result["Benchmark3_Unhedged_Price"].pct_change(),
        "Benchmark Composite": result["Benchmark_Composite_Price"].pct_change(),
        "Candidate 1": result["Candidate1_Price"].pct_change(),
    }
    metrics = pd.DataFrame({name: calculate_metrics(series) for name, series in metric_series.items()}).T
    metrics.to_csv(output_dir / "daily_metrics.csv", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--market-csv", type=Path, default=None)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    args = parser.parse_args()

    calendar = business_calendar(args.start, args.end)
    candidate_path = args.data_dir / "candidate1_etf_prices_2023_2026.csv"
    if not candidate_path.exists():
        candidate_path = PROJECT_ROOT / "data" / "processed" / "candidate_universe_daily.csv"
    candidate = read_candidate_prices(candidate_path)
    market = read_market_prices(args.market_csv, args.start, args.end)
    futures = load_futures(args.data_dir / "krx_usd_futures_2023_2026_merged.csv", calendar)
    result = build_daily_backtest(candidate, market, futures, args.start, args.end)
    write_outputs(result, args.output_dir)
    print(f"Wrote {len(result):,} weekday rows to {args.output_dir}")


if __name__ == "__main__":
    main()
