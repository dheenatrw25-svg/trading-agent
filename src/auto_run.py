import datetime
import time
from pathlib import Path

import schedule

from . import (
    chart_engine,
    data_engine,
    market_predictor,
    polymarket_paper,
    pro_strategist,
)
from .prediction_logs import (
    ADVISORY_LOG_FIELDS,
    ADVISORY_LOG_FILE,
    append_record,
    append_forecast_result,
    forecast_expiration_timestamp,
    initialize_advisory_log,
    to_utc_iso,
    utc_now,
)


CSV_FILENAME = str(ADVISORY_LOG_FILE)
CSV_HEADERS = ADVISORY_LOG_FIELDS
FORECAST_HORIZON_HOURS = 4


def initialize_csv(filename: str = CSV_FILENAME) -> None:
    initialize_advisory_log(Path(filename))


def _validate_advisory(advisory: object) -> bool:
    try:
        pro_strategist.validate_advisory(advisory)
        return True
    except (TypeError, ValueError):
        return False


def _evaluate_and_evolve_before_forecast() -> None:
    from .trading_agent import evaluate_and_evolve_expired_forecasts

    evaluate_and_evolve_expired_forecasts()


def run_agent_pipeline(*, reviewed: bool = False) -> None:
    print("\nGatekeeper triggered.")

    try:
        if not reviewed:
            _evaluate_and_evolve_before_forecast()
        execution_timestamp = utc_now()
        spot_data = data_engine.fetch_spot_data(execution_timestamp)
        market_context = data_engine.compile_market_context(spot_data=spot_data)
        btc_price = market_context["spot"]["current_close"]

        volume_profile = chart_engine.generate_pro_chart(spot_data)
        full_context = {
            **market_context,
            "volume_profile": volume_profile,
        }
        advisory = pro_strategist.generate_trading_advisory(full_context)

        if not _validate_advisory(advisory):
            print("[GATEKEEPER] Invalid advisory payload; prediction was not logged.")
            return

        initialize_csv()
        issue_time = datetime.datetime.fromisoformat(
            str(
                market_context.get(
                    "execution_timestamp", execution_timestamp.isoformat()
                )
            ).replace("Z", "+00:00")
        )
        print(f"[GATEKEEPER] execution_timestamp={to_utc_iso(issue_time)}.")
        issue_timestamp = to_utc_iso(issue_time)
        source_timestamp = datetime.datetime.fromisoformat(
            str(
                market_context.get(
                    "source_candle_timestamp", spot_data.index[-1].isoformat()
                )
            ).replace("Z", "+00:00")
        )
        if source_timestamp.tzinfo is None:
            raise ValueError("Source candle timestamp must include a timezone.")
        source_timestamp = source_timestamp.astimezone(datetime.timezone.utc)
        if source_timestamp > issue_time:
            raise ValueError("Source candle timestamp is later than the forecast issue time.")
        row = {
            "timestamp": issue_timestamp,
            "forecast_issue_timestamp": issue_timestamp,
            "source_candle_timestamp": to_utc_iso(source_timestamp),
            "target_timestamp": to_utc_iso(
                forecast_expiration_timestamp(issue_time, FORECAST_HORIZON_HOURS)
            ),
            "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
            "btc_price_at_prediction": btc_price,
            "direction": advisory["direction"],
            "target_low": advisory["target_range_low"],
            "target_high": advisory["target_range_high"],
            "invalidation_price": advisory["invalidation_price"],
            "confidence": advisory["confidence_score"],
            "graded": "False",
            "actual_price_timestamp": "",
            "actual_outcome": "Pending",
            "market_regime": advisory["market_regime"],
            "reasoning": advisory["reasoning"],
        }
        append_record(ADVISORY_LOG_FILE, ADVISORY_LOG_FIELDS, row)
        result_row = {
            **row,
            "target_price": advisory["target_price"],
            "setup_actionable": pro_strategist.calculate_risk_reward(
                btc_price,
                advisory["direction"],
                advisory["target_range_low"],
                advisory["target_range_high"],
                advisory["invalidation_price"],
            )["actionable"],
        }
        append_forecast_result(result_row, FORECAST_HORIZON_HOURS)
        print(f"[GATEKEEPER] Advisory logged to {CSV_FILENAME}.")
        try:
            result = polymarket_paper.process_forecast(row, "advisory")
            print(f"[POLYMARKET PAPER] 4H advisory: {result['status']}.")
        except Exception as exc:
            print(f"[POLYMARKET PAPER] 4H advisory processing failed: {exc}")
    except Exception as exc:
        print(f"[GATEKEEPER] Pipeline failed: {exc}")


def run_market_prediction() -> None:
    print(
        f"\n--- Waking up AI Predictor at "
        f"{datetime.datetime.now().strftime('%H:%M:%S')} ---"
    )
    try:
        _evaluate_and_evolve_before_forecast()
    except Exception as exc:
        print(f"Error during expired-forecast evaluation/evolution: {exc}")
        return
    try:
        forecast = market_predictor.run_predictor("BTC-USD")
        if forecast:
            try:
                result = polymarket_paper.process_forecast(forecast, "forecast")
                print(f"[POLYMARKET PAPER] 2H forecast: {result['status']}.")
            except Exception as exc:
                print(f"[POLYMARKET PAPER] 2H forecast processing failed: {exc}")
    except Exception as exc:
        print(f"Error during prediction: {exc}")
def settle_polymarket_positions() -> None:
    try:
        result = polymarket_paper.settle_open_positions()
        if result["settled"] or result["pending"]:
            print(
                "[POLYMARKET PAPER] "
                f"{result['settled']} positions settled, "
                f"{result['pending']} awaiting confirmed resolution."
            )
    except Exception as exc:
        print(f"[POLYMARKET PAPER] Settlement check failed: {exc}")


if __name__ == "__main__":
    initialize_csv()
    run_agent_pipeline()
    run_market_prediction()
    schedule.every(4).hours.do(run_agent_pipeline)
    schedule.every(60).minutes.do(run_market_prediction)
    schedule.every(1).minutes.do(settle_polymarket_positions)

    print("Starting trading agent and hourly predictor scheduler.")
    print("Leave this terminal open. Press Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(60)
