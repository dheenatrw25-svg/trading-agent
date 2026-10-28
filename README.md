# Trading Agent

## Project layout

- `src/` — Python application and forecasting modules
- `assets/` — generated charts and archived chart images
- `logs/` — prediction, trade, and runtime logs; `logs/backups/` contains archived CSVs
- `templates/` — dashboard HTML
- `tests/` — automated tests

## Run

Run commands from the repository root:

```powershell
python -m uvicorn src.app:app --reload
python -m src.auto_run
python -m src.market_predictor
python -m src.trading_agent
python -m unittest discover -s tests
```
