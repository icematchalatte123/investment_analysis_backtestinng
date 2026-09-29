"""후보 ETF 가격을 후보군 CSV에 추가한다.

기존 후보군 파일은 그대로 두고, 정규화된
``candidate_universe_daily.csv``를 새로 만든다.  새 ETF를 추가할 때는
FinanceDataReader의 Close 가격을 내려받아 기존 열과 합친다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from daily_backtest import END_DATE, START_DATE, align_weekday_prices, business_calendar

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def download_close(ticker: str, start: str, end: str) -> pd.Series:
    try:
        import FinanceDataReader as fdr
    except ImportError as exc:
        raise ImportError(
            "FinanceDataReader가 없습니다. .venv에서 requirements.txt를 설치하세요."
        ) from exc

    data = fdr.DataReader(ticker, start, end)
    if data is None or data.empty or "Close" not in data:
        raise ValueError(f"{ticker}의 Close 가격을 받지 못했습니다.")
    series = pd.to_numeric(data["Close"], errors="coerce")
    series.index = pd.to_datetime(series.index)
    return series.rename(ticker)


def main() -> None:
    parser = argparse.ArgumentParser(description="후보 ETF를 후보군 가격 CSV에 추가")
    parser.add_argument("--ticker", help="추가할 FDR 티커. 예: SPY, QQQ, 069500")
    parser.add_argument("--column", help="CSV에 저장할 열 이름. 기본값은 ticker")
    parser.add_argument(
        "--universe",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "candidate_universe_daily.csv",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        help="universe가 아직 없을 때 사용할 기존 원본 후보군 CSV",
    )
    parser.add_argument("--start", default=START_DATE, help="고정 기간 시작일")
    parser.add_argument("--end", default=END_DATE, help="고정 기간 종료일")
    args = parser.parse_args()

    if args.universe.exists():
        base = read_prices(args.universe)
    elif args.source_csv is not None:
        base = read_prices(args.source_csv)
    else:
        raise FileNotFoundError(
            f"{args.universe}가 없습니다. 최초 실행은 --source-csv를 지정하세요."
        )

    if args.ticker:
        column = args.column or args.ticker
        if column in base.columns:
            raise ValueError(f"이미 존재하는 후보 열입니다: {column}")
        fetch_start = (pd.Timestamp(args.start) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        fetch_end = (pd.Timestamp(args.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        base[column] = download_close(args.ticker, fetch_start, fetch_end)

    calendar = business_calendar(args.start, args.end)
    aligned = align_weekday_prices(base, calendar)
    aligned.index.name = "Date"
    aligned.reset_index().to_csv(args.universe, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {args.universe} ({len(aligned):,}행, {len(aligned.columns):,}개 가격 열)")


if __name__ == "__main__":
    main()
