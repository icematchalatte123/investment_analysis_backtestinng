"""Local ETF candidate web app using only the Python standard library."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from add_candidate_etf import download_close  # noqa: E402
from daily_backtest import (  # noqa: E402
    END_DATE,
    RISK_FREE_WEEKLY,
    START_DATE,
    align_weekday_prices,
    business_calendar,
    calculate_metrics,
)
from select_candidate import build_candidate_table  # noqa: E402


UNIVERSE_PATH = ROOT / "data" / "processed" / "candidate_universe_daily.csv"
BENCHMARK_PATH = ROOT / "data" / "processed" / "benchmark_table_daily.csv"
CANDIDATE_PATH = ROOT / "data" / "processed" / "candidate_table_daily.csv"
CONFIG_PATH = ROOT / "config" / "candidate_config.json"
METADATA_PATH = ROOT / "config" / "etf_metadata.json"
RESULTS_DIR = ROOT / "results"

PRICE_COLUMNS = {
    "Candidate": "Candidate_Price",
    "Benchmark Composite": "Benchmark_Composite_Price",
}


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    return df.set_index("Date").sort_index()


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"weights": {}, "usd_etfs": []}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_config(weights: dict[str, float], usd_etfs: set[str]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {"weights": weights, "usd_etfs": sorted(usd_etfs)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_metadata() -> dict[str, dict[str, str]]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    config = load_config()
    return {
        name: {"currency": "USD" if name in config.get("usd_etfs", []) else "KRW"}
        for name in read_table(UNIVERSE_PATH).columns
        if name != "USDKRW"
    }


def save_metadata(metadata: dict[str, dict[str, str]]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def metric_records(metrics: pd.DataFrame) -> list[dict]:
    records = []
    for name, row in metrics.iterrows():
        record = {"Portfolio": name}
        for key, value in row.items():
            record[key] = None if pd.isna(value) else float(value)
        records.append(record)
    return records


def build_comparison(candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = read_table(BENCHMARK_PATH)
    data = benchmark.join(candidate[["Candidate_Price"]], how="inner")
    if data.empty:
        raise ValueError("The benchmark and candidate have no common dates.")
    returns = {name: data[column].pct_change() for name, column in PRICE_COLUMNS.items()}
    metrics = pd.DataFrame({name: calculate_metrics(value) for name, value in returns.items()}).T
    market_returns = returns["Benchmark Composite"]
    sml_metrics = pd.DataFrame(
        {
            name: calculate_sml_metrics(value, market_returns)
            for name, value in returns.items()
        }
    ).T
    metrics = metrics.join(sml_metrics)
    return data, metrics


def calculate_sml_metrics(
    returns: pd.Series,
    market_returns: pd.Series,
) -> dict[str, float]:
    """Calculate beta, Jensen's alpha, and Treynor ratio against the composite benchmark."""
    joined = pd.concat(
        [pd.to_numeric(returns, errors="coerce"), pd.to_numeric(market_returns, errors="coerce")],
        axis=1,
        join="inner",
    ).dropna()
    joined.columns = ["asset", "market"]
    if len(joined) < 2:
        return {"Beta to Benchmark": np.nan, "Jensen's Alpha": np.nan, "Treynor Ratio": np.nan}

    daily_rf = (1.0 + RISK_FREE_WEEKLY) ** (1.0 / 5.0) - 1.0
    asset_excess = joined["asset"] - daily_rf
    market_excess = joined["market"] - daily_rf
    market_variance = float(market_excess.var(ddof=1))
    if not market_variance:
        return {"Beta to Benchmark": np.nan, "Jensen's Alpha": np.nan, "Treynor Ratio": np.nan}

    beta = float(asset_excess.cov(market_excess) / market_variance)
    alpha = float((asset_excess.mean() - beta * market_excess.mean()) * 252.0)
    treynor = float(asset_excess.mean() * 252.0 / beta) if beta else np.nan
    return {
        "Beta to Benchmark": beta,
        "Jensen's Alpha": alpha,
        "Treynor Ratio": treynor,
    }


def build_correlation_matrix(
    candidate: pd.DataFrame,
    benchmark: pd.DataFrame,
    weights: dict[str, float],
    usd_etfs: set[str],
) -> pd.DataFrame:
    """Build a daily-return correlation matrix for selected ETFs and the benchmark."""
    returns: dict[str, pd.Series] = {}
    fx_returns = candidate["USDKRW_Close"].pct_change()
    for name in weights:
        close_column = f"{name}_Close"
        if close_column not in candidate:
            continue
        asset_returns = candidate[close_column].pct_change()
        if name in usd_etfs:
            asset_returns = (1.0 + asset_returns) * (1.0 + fx_returns) - 1.0
        returns[name] = asset_returns
    returns["Benchmark Composite"] = benchmark["Benchmark_Composite_Price"].pct_change()
    return pd.DataFrame(returns).dropna().corr()


def weighted_average_correlation(
    correlation: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """Return the off-diagonal pairwise correlation weighted by ETF weights."""
    names = [name for name in weights if name in correlation.index]
    numerator = 0.0
    denominator = 0.0
    for first, name in enumerate(names):
        for other in names[first + 1 :]:
            pair_weight = float(weights[name]) * float(weights[other])
            numerator += pair_weight * float(correlation.loc[name, other])
            denominator += pair_weight
    return numerator / denominator if denominator else np.nan


def save_correlation_heatmap(correlation: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(correlation.to_numpy(), vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(correlation.columns)), labels=correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.index)), labels=correlation.index)
    for row in range(len(correlation.index)):
        for col in range(len(correlation.columns)):
            ax.text(col, row, f"{correlation.iat[row, col]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Daily Return Correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "candidate_asset_correlation.png", dpi=160)
    plt.close(fig)


def current_weighted_correlation(config: dict) -> float | None:
    weights = {str(name): float(value) for name, value in config.get("weights", {}).items()}
    if not weights or not CANDIDATE_PATH.exists():
        return None
    candidate = read_table(CANDIDATE_PATH)
    required = {f"{name}_Close" for name in weights} | {"USDKRW_Close"}
    if not required.issubset(candidate.columns):
        return None
    correlation = build_correlation_matrix(
        candidate,
        read_table(BENCHMARK_PATH),
        weights,
        set(config.get("usd_etfs", [])),
    )
    score = weighted_average_correlation(correlation, weights)
    return None if pd.isna(score) else float(score)


def save_weights_pie(weights: dict[str, float]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    positive = {name: float(value) for name, value in weights.items() if float(value) > 0}
    if not positive:
        return
    palette = [
        "#264653",  # deep navy
        "#2A9D8F",  # teal
        "#E9C46A",  # soft gold
        "#F4A261",  # warm orange
        "#E76F51",  # coral
        "#7B6D8D",  # muted violet
        "#5B8EAD",  # slate blue
        "#6BAF92",  # sage green
        "#D88C9A",  # dusty rose
    ]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, labels, percentages = ax.pie(
        list(positive.values()),
        labels=list(positive),
        autopct="%1.0f%%",
        startangle=90,
        counterclock=False,
        colors=[palette[index % len(palette)] for index in range(len(positive))],
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
        pctdistance=0.7,
        labeldistance=1.08,
        textprops={"fontsize": 13},
    )
    for label in labels:
        label.set_fontsize(13)
    for percentage in percentages:
        percentage.set_fontsize(14)
        percentage.set_fontweight("bold")
    ax.set_title("Candidate Portfolio Weights", fontsize=18, pad=18)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "candidate_weights_pie.png", dpi=160)
    plt.close(fig)


def save_comparison_outputs(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    weights: dict[str, float],
    correlation: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        RESULTS_DIR / "candidate_benchmark_metrics.csv",
        index_label="Portfolio",
        encoding="utf-8-sig",
    )

    price_data = data[list(PRICE_COLUMNS.values())]
    normalized = price_data.div(price_data.iloc[0]).mul(100.0)
    normalized.columns = list(PRICE_COLUMNS)
    fig, ax = plt.subplots(figsize=(12, 6))
    normalized.plot(ax=ax, linewidth=1.2)
    ax.set_title("Candidate vs Benchmark (100 at start)")
    ax.set_ylabel("Index")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "candidate_vs_benchmark_cumulative.png", dpi=160)
    plt.close(fig)

    returns = price_data.pct_change()
    returns.columns = list(PRICE_COLUMNS)
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1.0)
    fig, ax = plt.subplots(figsize=(12, 6))
    drawdown.plot(ax=ax, linewidth=1.0)
    ax.set_title("Candidate vs Benchmark Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "candidate_vs_benchmark_drawdown.png", dpi=160)
    plt.close(fig)

    rolling_vol = returns.rolling(21).std() * np.sqrt(252.0)
    fig, ax = plt.subplots(figsize=(12, 6))
    rolling_vol.plot(ax=ax, linewidth=1.0)
    ax.set_title("21-day Rolling Annualized Volatility")
    ax.set_ylabel("Volatility")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "candidate_vs_benchmark_rolling_volatility.png", dpi=160)
    plt.close(fig)
    save_weights_pie(weights)
    save_correlation_heatmap(correlation)


def state_payload() -> dict:
    universe = read_table(UNIVERSE_PATH)
    config = load_config()
    metadata = load_metadata()
    names = [name for name in universe.columns if name != "USDKRW"]
    return {
        "start": START_DATE,
        "end": END_DATE,
        "etfs": [
            {
                "name": name,
                "ticker": metadata.get(name, {}).get("ticker", name),
                "currency": metadata.get(name, {}).get("currency", "KRW"),
                "weight": float(config.get("weights", {}).get(name, 0.0)),
            }
            for name in names
        ],
        "usd_etfs": sorted(set(config.get("usd_etfs", []))),
        "candidate_exists": CANDIDATE_PATH.exists(),
        "weighted_correlation": current_weighted_correlation(config),
    }


def add_etf(payload: dict) -> dict:
    ticker = str(payload.get("ticker", "")).strip()
    column = str(payload.get("column", ticker)).strip()
    currency = str(payload.get("currency", "KRW")).upper()
    if not ticker or not column:
        raise ValueError("Enter both ticker and column name.")
    if currency not in {"KRW", "USD"}:
        raise ValueError("currency must be KRW or USD.")

    universe = read_table(UNIVERSE_PATH)
    if column == "USDKRW" or column in universe.columns:
        raise ValueError(f"This column already exists or is reserved: {column}")
    fetch_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    fetch_end = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        universe[column] = download_close(ticker, fetch_start, fetch_end)
    except Exception as exc:
        raise ValueError(
            "External price lookup failed. Allow Yahoo Finance network access and try again. "
            f"Details: {exc}"
        ) from exc
    aligned = align_weekday_prices(universe, business_calendar(START_DATE, END_DATE))
    aligned.reset_index().to_csv(UNIVERSE_PATH, index=False, encoding="utf-8-sig")

    metadata = load_metadata()
    metadata[column] = {"ticker": ticker, "currency": currency}
    save_metadata(metadata)
    return state_payload()


def remove_etf(column: str) -> dict:
    if column == "USDKRW":
        raise ValueError("USDKRW is required for FX conversion and cannot be deleted.")
    universe = read_table(UNIVERSE_PATH)
    if column not in universe.columns:
        raise ValueError(f"ETF column not found in the candidate universe: {column}")
    universe.drop(columns=[column]).reset_index().to_csv(
        UNIVERSE_PATH, index=False, encoding="utf-8-sig"
    )

    config = load_config()
    weights = {
        name: float(value)
        for name, value in config.get("weights", {}).items()
        if name != column
    }
    total = sum(weights.values())
    if total:
        weights = {name: value / total for name, value in weights.items()}
    save_config(weights, set(config.get("usd_etfs", [])) - {column})

    metadata = load_metadata()
    metadata.pop(column, None)
    save_metadata(metadata)
    return state_payload()


def calculate_candidate(payload: dict) -> dict:
    selected = list(dict.fromkeys(str(name) for name in payload.get("selected", [])))
    if not selected or len(selected) > 9:
        raise ValueError("Select between 1 and 9 ETFs.")
    raw_weights = payload.get("weights", {})
    weights = {name: float(raw_weights.get(name, 0.0)) for name in selected}
    if any(value < 0 for value in weights.values()):
        raise ValueError("Weights must be non-negative.")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-8):
        raise ValueError(f"Weights must sum to 100%: {sum(weights.values()):.4%}")
    usd_etfs = set(payload.get("usd_etfs", [])) & set(selected)

    candidate = build_candidate_table(
        read_table(UNIVERSE_PATH), weights, usd_etfs, START_DATE, END_DATE
    )
    candidate.reset_index().to_csv(CANDIDATE_PATH, index=False, encoding="utf-8-sig")
    save_config(weights, usd_etfs)
    data, metrics = build_comparison(candidate)
    benchmark = read_table(BENCHMARK_PATH)
    correlation = build_correlation_matrix(candidate, benchmark, weights, usd_etfs)
    save_comparison_outputs(data, metrics, weights, correlation)
    score = weighted_average_correlation(correlation, weights)
    return {
        "metrics": metric_records(metrics),
        "weighted_correlation": None if pd.isna(score) else float(score),
        "state": state_payload(),
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ETF Candidate Backtest</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background:#f5f7fb; color:#1f2937; }
    main { max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1 { margin-bottom: 4px; } h2 { margin-top: 0; }
    .muted { color:#64748b; } .grid { display:grid; grid-template-columns: 360px 1fr; gap:18px; }
    .card { background:white; border:1px solid #dbe2ea; border-radius:10px; padding:18px; margin-bottom:18px; }
    label { display:block; margin:9px 0 4px; font-weight:600; }
    input, select, button { box-sizing:border-box; width:100%; padding:8px; border:1px solid #b8c4d1; border-radius:6px; }
    select[multiple] { min-height:180px; } button { background:#2563eb; color:white; border:0; cursor:pointer; margin-top:10px; }
    button.danger { background:#dc2626; width:auto; padding:6px 10px; margin:0; } button:disabled { background:#94a3b8; cursor:not-allowed; }
    .weight-row { display:grid; grid-template-columns: 1fr 120px; gap:8px; align-items:center; }
    .weight-row label { margin:4px 0; } table { border-collapse:collapse; width:100%; font-size:14px; }
    th, td { padding:8px; border-bottom:1px solid #e2e8f0; text-align:right; } th:first-child, td:first-child { text-align:left; }
    .metrics-table-wrap { overflow-x:hidden; width:100%; }
    .metrics-table { width:100%; min-width:0; table-layout:fixed; font-size:14px; }
    .metrics-table th { min-width:0; white-space:normal; overflow-wrap:anywhere; line-height:1.25; vertical-align:bottom; font-size:13px; }
    .metrics-table th:first-child { min-width:0; width:12%; }
    .metrics-table td { white-space:normal; overflow-wrap:anywhere; }
    textarea.paste-block { width:100%; box-sizing:border-box; resize:vertical; font-family:Consolas, monospace; font-size:12px; line-height:1.45; padding:8px; border:1px solid #b8c4d1; border-radius:6px; background:#f8fafc; }
    .copy-row { display:grid; grid-template-columns:1fr 160px; gap:8px; align-items:start; margin-top:8px; }
    .copy-row button { margin-top:0; }
    .correlation-chart { width:100%; max-width:540px; max-height:540px; object-fit:contain; display:block; margin:8px auto 0; }
    img.chart { width:100%; border:1px solid #e2e8f0; margin-top:8px; } .status { min-height:24px; margin-top:10px; }
    .ok { color:#15803d; } .error { color:#b91c1c; }
    .loading-overlay { display:none; position:fixed; inset:0; z-index:1000; background:rgba(15,23,42,.42); align-items:center; justify-content:center; }
    .loading-overlay.show { display:flex; } .loading-box { min-width:240px; padding:24px; text-align:center; color:#0f172a; background:white; border-radius:12px; box-shadow:0 10px 35px rgba(15,23,42,.25); }
    .spinner { width:34px; height:34px; margin:0 auto 12px; border:4px solid #dbeafe; border-top-color:#2563eb; border-radius:50%; animation:spin .8s linear infinite; }
    @keyframes spin { to { transform:rotate(360deg); } }
    @media (max-width: 900px) {
      main { padding:12px; }
      .grid { grid-template-columns:1fr; gap:12px; }
      .card { padding:12px; }
      .metrics-table { font-size:11px; }
      .metrics-table th { font-size:10px; padding:5px 3px; }
      .metrics-table td { padding:5px 3px; }
    }
  </style>
</head>
<body>
<main>
  <h1>ETF Candidate Backtest</h1>
  <div id="period" class="muted"></div>
  <div class="grid">
    <section>
      <div class="card">
        <h2>Candidate Selection</h2>
        <label for="etfs">ETF Candidates (max 9)</label>
        <select id="etfs" multiple></select>
        <div id="weights"></div>
        <label for="usd">USD-denominated ETFs</label>
        <select id="usd" multiple></select>
        <div id="sum" class="muted"></div>
        <button id="calculate">Calculate, Save &amp; Compare</button>
        <div id="candidate-status" class="status"></div>
      </div>
      <div class="card">
        <h2>Position Sheet Paste</h2>
        <div class="muted">Copy the tab-separated rows into <code>Positions!B6:K14</code>. The workbook's formula columns remain untouched.</div>
        <label for="position-paste">Position input block</label>
        <textarea id="position-paste" class="paste-block" rows="9" readonly></textarea>
        <div class="copy-row"><span class="muted">ETF, ticker, region, direction and default FX fields</span><button id="copy-position">Copy inputs</button></div>
        <label for="weight-paste">Candidate target weights</label>
        <textarea id="weight-paste" class="paste-block" rows="9" readonly></textarea>
        <div class="copy-row"><span class="muted">Copy into <code>Positions!R6:R14</code> (Notes)</span><button id="copy-weights">Copy weights</button></div>
        <div id="position-status" class="status"></div>
      </div>
      <div class="card"><h2>Candidate Portfolio Weights</h2><img id="weights-pie" class="chart" src="/results/candidate_weights_pie.png" alt="Candidate portfolio weights"></div>
      <div class="card">
        <h2>Add ETF Data</h2>
        <label for="ticker">FinanceDataReader Ticker</label>
        <input id="ticker" placeholder="e.g. SPY or 069500">
        <label for="column">Candidate Universe Column</label>
        <input id="column" placeholder="e.g. SPY">
        <label for="currency">Currency</label>
        <select id="currency"><option>USD</option><option>KRW</option></select>
        <button id="add">Add ETF Price</button>
        <div id="add-status" class="status"></div>
      </div>
    </section>
    <section>
      <div class="card metrics-card"><h2>Performance Metrics</h2><div class="muted">SML metrics use Benchmark Composite as the market portfolio.</div><div id="metrics" class="muted">Run the candidate calculation to display metrics.</div></div>
      <div class="card"><h2>Asset Correlation</h2><div class="muted">Daily return correlation for the selected ETFs and Benchmark Composite.</div><div id="correlation-summary" class="correlation-summary muted">Weighted average correlation: loading...</div><img id="correlation" class="correlation-chart" src="/results/candidate_asset_correlation.png" alt="Asset correlation heatmap"></div>
      <div class="card"><h2>Cumulative Performance</h2><img id="cumulative" class="chart" alt="Cumulative performance"></div>
      <div class="card"><h2>Drawdown</h2><img id="drawdown" class="chart" alt="Drawdown"></div>
      <div class="card"><h2>21-Day Rolling Volatility</h2><img id="volatility" class="chart" alt="Rolling volatility"></div>
    </section>
  </div>
  <section class="card"><h2>Candidate Universe Management</h2><div id="universe"></div></section>
</main>
<div id="loading" class="loading-overlay" aria-live="polite" aria-busy="false">
  <div class="loading-box"><div class="spinner"></div><div id="loading-message">Processing...</div></div>
</div>
<script>
let state = null;
const $ = (id) => document.getElementById(id);
function esc(value) { return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function selectedNames() { return Array.from($('etfs').selectedOptions).map(o => o.value); }
function renderWeights() {
  const selected = selectedNames();
  $('weights').innerHTML = selected.map(name => {
    const item = state.etfs.find(x => x.name === name);
    const value = item && item.weight ? item.weight * 100 : 100 / selected.length;
    return `<div class="weight-row"><label>${esc(name)}</label><input class="weight" data-name="${esc(name)}" type="number" min="0" max="100" step="1" value="${Math.round(value)}"></div>`;
  }).join('');
  const usd = $('usd');
  usd.innerHTML = selected.map(name => `<option value="${esc(name)}">${esc(name)}</option>`).join('');
  const usdSet = new Set(state.usd_etfs);
  Array.from(usd.options).forEach(o => o.selected = usdSet.has(o.value));
  updateSum();
}
function updateSum() {
  const sum = Array.from(document.querySelectorAll('.weight')).reduce((a, el) => a + Number(el.value || 0), 0);
  $('sum').textContent = `Weight total: ${sum.toFixed(0)}%`;
}
function renderPositionPaste() {
  const selected = selectedNames();
  const weights = {};
  document.querySelectorAll('.weight').forEach(el => weights[el.dataset.name] = Number(el.value || 0));
  const rows = selected.map((name, index) => {
    const item = state.etfs.find(x => x.name === name) || {};
    const region = item.currency === 'USD' ? 'United States' : 'South Korea';
    return [index + 1, 'ETF', item.ticker || name, region, 'Long', '', '', '', '1', '1'];
  });
  $('position-paste').value = rows.map(row => row.join('\t')).join('\n');
  $('weight-paste').value = selected.map(name => `${name}\tCandidate target weight: ${(Number(weights[name] || 0)).toFixed(0)}%`).join('\n');
}
async function copyPasteBlock(fieldId, message) {
  const value = $(fieldId).value;
  if (!value) { setStatus('position-status', 'Select at least one ETF first.', false); return; }
  try {
    await navigator.clipboard.writeText(value);
  } catch (e) {
    $(fieldId).focus(); $(fieldId).select(); document.execCommand('copy');
  }
  setStatus('position-status', message, true);
}
function renderState(next) {
  state = next; $('period').textContent = `Fixed period: ${state.start} ~ ${state.end}`;
  if (state.weighted_correlation == null) $('correlation-summary').textContent = 'Weighted average correlation: run the candidate calculation to update.';
  else $('correlation-summary').textContent = `Weighted average correlation: ${Number(state.weighted_correlation).toFixed(3)}`;
  const select = $('etfs'); const prior = new Set(selectedNames());
  select.innerHTML = state.etfs.map(item => `<option value="${esc(item.name)}">${esc(item.name)} (${esc(item.currency)})</option>`).join('');
  state.etfs.forEach((item, i) => { select.options[i].selected = prior.size ? prior.has(item.name) : item.weight > 0; });
  renderWeights();
  renderPositionPaste();
  $('universe').innerHTML = `<table><thead><tr><th>ETF</th><th>Ticker</th><th>Currency</th><th>Actions</th></tr></thead><tbody>${state.etfs.map(item => `<tr><td>${esc(item.name)}</td><td>${esc(item.ticker)}</td><td>${esc(item.currency)}</td><td><button class="danger" data-delete="${esc(item.name)}">Delete</button></td></tr>`).join('')}</tbody></table>`;
  document.querySelectorAll('[data-delete]').forEach(btn => btn.onclick = () => removeEtf(btn.dataset.delete));
}
function setStatus(id, message, ok) { $(id).textContent = message; $(id).className = `status ${ok ? 'ok' : 'error'}`; }
function setLoading(show, message) { $('loading').classList.toggle('show', show); $('loading').setAttribute('aria-busy', show ? 'true' : 'false'); if (message) $('loading-message').textContent = message; }
async function api(path, body) {
  const response = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {});
  const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Request failed'); return result;
}
async function removeEtf(name) {
  if (!confirm(`${name}: delete this ETF from the candidate universe?`)) return;
  setLoading(true, 'Updating the candidate universe...');
  try { const result = await api('/api/remove-etf', {column:name}); renderState(result.state); setStatus('add-status', `${name} deleted. Recalculate the candidate.`, true); }
  catch (e) { setStatus('add-status', e.message, false); }
  finally { setLoading(false); }
}
 $('etfs').onchange = () => { if (selectedNames().length > 9) { alert('You can select up to 9 ETFs.'); $('etfs').options[$('etfs').selectedIndex].selected = false; } renderWeights(); renderPositionPaste(); };
document.addEventListener('input', e => { if (e.target.classList.contains('weight')) { updateSum(); renderPositionPaste(); } });
$('copy-position').onclick = () => copyPasteBlock('position-paste', 'Position inputs copied. Paste them into Positions!B6:K14.');
$('copy-weights').onclick = () => copyPasteBlock('weight-paste', 'Target weights copied. Paste them into Positions!R6:R14.');
$('calculate').onclick = async () => {
  const selected = selectedNames(); const weights = {}; document.querySelectorAll('.weight').forEach(el => weights[el.dataset.name] = Number(el.value) / 100);
  const usd = Array.from($('usd').selectedOptions).map(o => o.value);
  setLoading(true, 'Calculating the candidate and generating charts...');
  try {
    const result = await api('/api/candidate', {selected, weights, usd_etfs:usd});
    renderState({...result.state, weighted_correlation: result.weighted_correlation}); setStatus('candidate-status', 'Calculation and save completed.', true); renderMetrics(result.metrics);
    const stamp = Date.now(); $('correlation').src = `/results/candidate_asset_correlation.png?${stamp}`; $('cumulative').src = `/results/candidate_vs_benchmark_cumulative.png?${stamp}`; $('drawdown').src = `/results/candidate_vs_benchmark_drawdown.png?${stamp}`; $('volatility').src = `/results/candidate_vs_benchmark_rolling_volatility.png?${stamp}`; $('weights-pie').src = `/results/candidate_weights_pie.png?${stamp}`;
  } catch (e) { setStatus('candidate-status', e.message, false); }
  finally { setLoading(false); }
};
function renderMetrics(rows) { const keys = ['Cumulative Return','Annualized Return','Annualized Volatility','Annualized Sharpe Ratio','Beta to Benchmark',"Jensen's Alpha",'Treynor Ratio','Maximum Drawdown','1-day 95% VaR','1-day 95% CVaR','1-week 95% VaR Proxy']; const percentKeys = new Set(['Cumulative Return','Annualized Return','Annualized Volatility',"Jensen's Alpha",'Maximum Drawdown','1-day 95% VaR','1-day 95% CVaR','1-week 95% VaR Proxy']); const label = k => percentKeys.has(k) ? `${k} (%)` : k; const format = (key, value) => value == null ? '' : percentKeys.has(key) ? `${(Number(value) * 100).toFixed(2)}%` : Number(value).toFixed(4); $('metrics').innerHTML = `<div class="metrics-table-wrap"><table class="metrics-table"><thead><tr><th>Portfolio</th>${keys.map(k => `<th>${label(k)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr><td>${esc(row.Portfolio)}</td>${keys.map(k => `<td>${format(k, row[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`; }
$('add').onclick = async () => {
  const ticker = $('ticker').value.trim(); const column = ($('column').value.trim() || ticker).toUpperCase(); const currency = $('currency').value;
  setLoading(true, 'Downloading ETF prices and adding to the candidate universe...');
  try { const result = await api('/api/add-etf', {ticker, column, currency}); renderState(result); setStatus('add-status', `${column} added successfully.`, true); $('ticker').value=''; $('column').value=''; }
  catch (e) { setStatus('add-status', e.message, false); }
  finally { setLoading(false); }
};
api('/api/state').then(renderState).catch(e => setStatus('candidate-status', e.message, false));
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            try:
                self.send_json(state_payload())
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return
        if path.startswith("/results/"):
            name = Path(path.removeprefix("/results/")).name
            target = RESULTS_DIR / name
            if target.exists() and target.is_file() and target.suffix.lower() == ".png":
                body = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            path = urlparse(self.path).path
            if path == "/api/candidate":
                self.send_json(calculate_candidate(payload))
            elif path == "/api/add-etf":
                self.send_json(add_etf(payload))
            elif path == "/api/remove-etf":
                self.send_json({"state": remove_etf(str(payload.get("column", "")))})
            else:
                self.send_json({"error": "Not found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local ETF candidate web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    config = load_config()
    if config.get("weights"):
        save_weights_pie(config["weights"])
        if CANDIDATE_PATH.exists():
            candidate = read_table(CANDIDATE_PATH)
            benchmark = read_table(BENCHMARK_PATH)
            correlation = build_correlation_matrix(
                candidate,
                benchmark,
                config["weights"],
                set(config.get("usd_etfs", [])),
            )
            save_correlation_heatmap(correlation)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ETF web app running at: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the web app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
