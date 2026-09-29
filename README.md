# ETF Candidate Backtest — Local Web App

A local web application for selecting ETF candidates, setting portfolio weights, comparing the candidate portfolio with the benchmark, and generating performance metrics and charts. Each user runs the server on their own computer, so configuration and result files remain local to that computer.

## Included files

```text
web/app.py
scripts/add_candidate_etf.py
scripts/daily_backtest.py
scripts/select_candidate.py
config/candidate_config.json
config/etf_metadata.json
data/processed/benchmark_table_daily.csv
data/processed/candidate_table_daily.csv
data/processed/candidate_universe_daily.csv
results/*.png
requirements.txt
```

The original Excel workbook, raw CSV files, notebooks, and virtual environment are not required by the local web app. The processed data included in `data/processed/` is already available on the first run.

## Requirements

- Python 3.10 or newer is recommended.
- An internet connection is required when adding new ETF prices.

## Windows setup

Open PowerShell and move to the repository folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe web\app.py
```

Open the following address in a browser:

```text
http://127.0.0.1:8501/
```

If `python` points to the Microsoft Store alias, select `.venv\Scripts\python.exe` as the Python interpreter in VS Code or use the full path to your installed Python executable.

## macOS / Linux setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python web/app.py
```

Open `http://127.0.0.1:8501/` in a browser.

## Using the app

1. Select up to nine ETFs under **Candidate Selection**.
2. Enter the portfolio weights and make sure the total is 100%.
3. Select the USD-denominated ETFs.
4. Click **Calculate, Save & Compare**.
5. Review the performance metrics, correlation matrix, cumulative performance, drawdown, and rolling volatility.
6. Use **Position Sheet Paste** to copy position inputs into the `Positions` sheet of the risk-management workbook.

The **Add ETF Data** feature retrieves prices through FinanceDataReader. Adding an ETF may fail if Yahoo Finance is blocked by a corporate or public network.

## Local data behavior

The app saves user changes in the following local files:

- `config/candidate_config.json`: selected ETFs and portfolio weights
- `config/etf_metadata.json`: ETF tickers and currencies
- `data/processed/candidate_universe_daily.csv`: candidate-universe prices
- `data/processed/candidate_table_daily.csv`: selected-candidate prices
- `results/`: generated charts and comparison outputs

Each computer has its own copy of these files. To stop the server, press `Ctrl+C` in the terminal running the app.

## Troubleshooting

If port 8501 is already in use, start the app on another port:

```powershell
.\.venv\Scripts\python.exe web\app.py --port 8502
```

Then open `http://127.0.0.1:8502/` in a browser.
