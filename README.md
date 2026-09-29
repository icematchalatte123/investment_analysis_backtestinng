# ETF Candidate Backtest — Local Web App

ETF 후보 선택, 비중 조절, 벤치마크 비교, 성과 지표·그래프 생성을 제공하는 로컬 웹 앱입니다. 각 사용자는 자신의 컴퓨터에서 서버를 실행하며, 설정과 결과 파일도 각자의 로컬 폴더에 저장됩니다.

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

원본 엑셀·원본 CSV·가상환경은 실행에 포함하지 않습니다. `data/processed/`에 저장된 초기 데이터는 저장소에 포함되어 있어 첫 실행부터 사용할 수 있습니다.

## Requirements

- Python 3.10 이상 권장
- ETF 가격을 새로 추가할 때는 인터넷 연결 필요

## Windows setup

PowerShell에서 저장소 폴더로 이동한 뒤 실행합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe web\app.py
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8501/
```

Python 명령이 `Windows Store alias`로 연결되면 VS Code에서 `.venv\Scripts\python.exe`를 Python 인터프리터로 선택하거나, 설치된 Python의 실제 경로를 사용합니다.

## macOS / Linux setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python web/app.py
```

브라우저에서 `http://127.0.0.1:8501/`을 엽니다.

## Using the app

1. Candidate Selection에서 ETF를 최대 9개까지 선택합니다.
2. 각 비중을 입력하고 합계가 100%인지 확인합니다.
3. USD ETF 여부를 선택합니다.
4. `Calculate, Save & Compare`를 누릅니다.
5. 성과 지표, 상관관계, 누적수익률, Drawdown, Rolling Volatility를 확인합니다.
6. `Position Sheet Paste`에서 `Positions` 시트에 붙여넣을 입력 블록을 복사할 수 있습니다.

`Add ETF Data`는 FinanceDataReader를 통해 가격을 조회합니다. 회사·공용 네트워크에서 Yahoo Finance 접속이 차단되면 ETF 추가가 실패할 수 있습니다.

## Local data behavior

웹 앱에서 저장하는 변경 내용은 다음 파일에 기록됩니다.

- `config/candidate_config.json`: 선택 종목과 비중
- `config/etf_metadata.json`: ETF 티커와 통화
- `data/processed/candidate_universe_daily.csv`: 후보군 가격
- `data/processed/candidate_table_daily.csv`: 선택된 candidate 가격
- `results/`: 계산 결과 그래프

이 파일들은 각 컴퓨터에서 독립적으로 변경됩니다. 서버를 종료하려면 실행 중인 터미널에서 `Ctrl+C`를 누릅니다.

## Troubleshooting

포트 8501이 이미 사용 중이면 다른 포트로 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe web\app.py --port 8502
```

그 경우 브라우저에서 `http://127.0.0.1:8502/`를 엽니다.
