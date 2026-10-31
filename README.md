# Trading Agent

## Project layout

- `src/` — Python application and forecasting modules
- `assets/` — generated charts and archived chart images
- `logs/` — prediction, trade, and runtime logs; `logs/backups/` contains archived CSVs
- `templates/` — dashboard HTML
- `tests/` — automated tests

## Forecast records and evaluation

The Gemini advisory receives a normalized JSON payload with `current_price`,
`technical_indicators` (`rsi_15m` and formatted `dxy_macro_trend`), and
`institutional_order_flow` (formatted order-book walls, taker-pressure status,
and open-interest trend). The application retains the detailed market context
internally for risk checks and prediction logging.

- The 2-hour BTC predictor writes only to `logs/btc_forecasts_2h.csv`.
- Dashboard and scheduled advisory predictions write only to `logs/advisory_log.csv`.
- Both forecast pipelines also append a display-ready record to
  `logs/forecast_results.csv`. This unified file uses the same nine columns for
  2-hour and 4-hour forecasts; times are displayed in IST, and unavailable
  historical values remain blank.
- The dashboard history API reads every record from the unified results file,
  newest first. The detailed evaluator and advisory logs remain unchanged.
- The prior CSV files are migration inputs. Before migration, they are copied to
  `logs/backups/*.pre-phase1.bak`; the original files are left untouched.
- Old 2-hour BTC forecasts without verifiable UTC timestamps and source-candle data
  are marked `NEEDS_REVIEW`, not assigned a guessed result. Malformed or
  market-shaped rows in an old shared advisory file are left in the original and
  its backup, rather than misclassified as dashboard predictions.
- Pending BTC forecasts are evaluated after their intended two-hour expiry. The
  evaluator records direction correctness, final-close range coverage, full-period
  range coverage, and projected-price errors as separate measurements. For
  `SIDEWAYS`, direction correctness is not applicable; the two range measurements
  are used instead.
- Evaluation prefers a 15-minute close at the exact expiry boundary. When expiry
  falls between 15-minute closes, it uses the latest completed 1-minute close no
  more than one minute before expiry. Range measurements use complete OHLC candle
  bars overlapping the evaluated interval, and the actual close timestamp is
  recorded so the candle resolution is visible.

## Run

Run commands from the repository root in PowerShell. Install dependencies into
the project virtual environment first; using a different Python interpreter may
fail with `ModuleNotFoundError` for project dependencies such as `schedule`.

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn src.app:app --reload
.\venv\Scripts\python.exe -m src.auto_run
.\venv\Scripts\python.exe -m src.market_predictor
.\venv\Scripts\python.exe -m src.trading_agent
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```
