# Trading Agent

## Project layout

- `src/` — Python application and forecasting modules
- `assets/` — generated charts and archived chart images
- `logs/` — prediction, trade, and runtime logs; `logs/backups/` contains archived CSVs
- `vault/01_Playbooks/` — human-readable playbooks
- `vault/02_Trade_Memories/` — Markdown forecast outcomes used as the secondary brain
- `templates/` — dashboard HTML
- `tests/` — automated tests

## Forecast records and evaluation

The Gemini advisory receives a normalized JSON payload with `current_price`,
`technical_indicators` (`rsi_15m` and formatted `dxy_macro_trend`), and
`institutional_order_flow` (formatted order-book walls, taker-pressure status,
and open-interest trend). The application retains the detailed market context
internally for risk checks and prediction logging.

- The 2-hour BTC predictor writes only to `logs/btc_forecasts_2h.csv`.
- The 2-hour forecast baseline uses Binance BTCUSDT live spot pricing; responses
  without an HTTP `Date` header within five seconds are rejected.
- Four-hour market-context downloads use one UTC `execution_timestamp` for both
  4-hour and 15-minute candle cutoffs. Advisory logs retain that execution time
  and the completed 15-minute candle close time, rather than pairing a 15-minute
  price with a 4-hour candle's opening time.
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
- Pending BTC forecasts and four-hour advisories are evaluated after their expiry.
  Evaluation outcomes and full forecast context are saved to dated Markdown files
  under `vault/02_Trade_Memories/`. Both horizons are graded strictly as `WIN` or
  `LOSS` by comparing the prediction-time BTC price with the historical Binance
  1-minute close at expiration; an unchanged close is a loss. Intraperiod highs,
  lows, targets, and ranges do not affect grading. Expiration timestamps are
  rounded up to the next full minute so an exact 1-minute candle close exists.
- The scheduled forecast cycle grades expired 2-hour and 4-hour forecasts, evolves
  the master playbook from unaudited losses, and only then generates new forecasts.
  Playbook lessons and updates are isolated by forecast horizon: 2-hour forecasts
  receive only 2-hour memories/rules, and 4-hour forecasts receive only 4-hour
  memories/rules.
- Order-book walls are aggregated in $10 price bins from Binance's 1,000-level
  depth snapshot. The evaluator reports the observed snapshot span because the
  endpoint may return fewer levels than needed to cover the full $2,000 search
  radius.

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

### Polymarket paper trading

The scheduler also runs an isolated Polymarket paper engine. It makes public
read-only Gamma and CLOB requests; it does not connect a wallet or submit orders.
The engine only enters a BTC Up/Down market when the market's published opening,
closing, duration, and resolution rules match a newly issued forecast. It logs
and skips unavailable, mismatched, stale, or illiquid markets. Positions are
settled only after Gamma reports a confirmed final outcome.

By default the separate virtual account starts with a `$100` bankroll and uses
a fixed `$5` stake per entry. Set `POLYMARKET_PAPER_BANKROLL` and
`POLYMARKET_PAPER_STAKE` before the first run to configure those defaults.
Entry costs include the ask-side spread, available ask liquidity, and the
market's taker fee using Polymarket's published crypto fee curve where enabled.
Paper state, entries/skips, and confirmed settlement results are stored in
`logs/polymarket_paper_state.json`,
`logs/polymarket_paper_trades.csv`, and
`logs/polymarket_paper_results.csv`. To run the integrated forecast and paper
trading scheduler:

```powershell
.\venv\Scripts\python.exe -m src.auto_run
```
