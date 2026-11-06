import unittest
from unittest.mock import patch

import pandas as pd

from src import app


class DashboardForecastRouteTests(unittest.TestCase):
    def test_four_hour_analysis_executes_advisory_paper_trade(self) -> None:
        spot_data = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.DatetimeIndex(["2026-11-03T00:00:00Z"]),
        )
        decision = {
            "direction": "UP",
            "target_range_low": 100.0,
            "target_range_high": 110.0,
            "confidence_score": 7,
            "market_regime": "uptrend",
            "reasoning": "Test forecast.",
        }
        paper_status = {"status": "entered", "trade": {"market_id": "4h-market"}}

        with (
            patch.object(app.data_engine, "fetch_spot_data", return_value=spot_data),
            patch.object(
                app.data_engine,
                "compile_market_context",
                return_value={"spot": {"current_close": 100.0}},
            ),
            patch.object(app.chart_engine, "generate_pro_chart", return_value={}),
            patch.object(
                app.pro_strategist,
                "generate_trading_advisory",
                return_value=decision,
            ),
            patch.object(app, "log_prediction") as log_prediction,
            patch.object(
                app.polymarket_paper,
                "process_forecast",
                return_value=paper_status,
            ) as process_forecast,
        ):
            result = app.analyze()

        forecast_row = process_forecast.call_args.args[0]
        self.assertEqual(forecast_row["direction"], "UP")
        self.assertEqual(forecast_row["forecast_horizon_hours"], 4)
        forecast_duration = (
            pd.Timestamp(forecast_row["target_timestamp"])
            - pd.Timestamp(forecast_row["forecast_issue_timestamp"])
        )
        self.assertGreaterEqual(forecast_duration, pd.Timedelta(hours=4))
        self.assertLess(forecast_duration, pd.Timedelta(hours=4, minutes=1))
        self.assertEqual(
            pd.Timestamp(forecast_row["target_timestamp"]).second,
            0,
        )
        process_forecast.assert_called_once_with(forecast_row, "advisory")
        self.assertEqual(result["polymarket_paper"], paper_status)
        log_prediction.assert_called_once()
        self.assertEqual(
            pd.Timestamp(log_prediction.call_args.args[3]).floor("s"),
            pd.Timestamp(forecast_row["forecast_issue_timestamp"]),
        )

    def test_two_hour_route_executes_forecast_paper_trade(self) -> None:
        forecast = {
            "forecast_id": "forecast-test",
            "predicted_direction": "BULLISH",
            "expected_low": 100.0,
            "expected_high": 110.0,
        }
        paper_status = {"status": "entered", "trade": {"market_id": "2h-market"}}

        with (
            patch.object(
                app.market_predictor, "run_predictor", return_value=forecast
            ) as run_predictor,
            patch.object(
                app.polymarket_paper,
                "process_forecast",
                return_value=paper_status,
            ) as process_forecast,
        ):
            result = app.predict_2h()

        run_predictor.assert_called_once_with("BTC-USD")
        process_forecast.assert_called_once_with(forecast, "forecast")
        self.assertEqual(result["polymarket_paper"], paper_status)


if __name__ == "__main__":
    unittest.main()
