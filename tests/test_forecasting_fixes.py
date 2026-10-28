import csv
import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import data_engine
import evaluator
import market_predictor
import prediction_logs
import pro_strategist


def valid_market_forecast() -> dict:
    return {
        "predicted_direction": "BULLISH",
        "confidence_score": 7,
        "expected_high_next_2h": 110.0,
        "expected_low_next_2h": 90.0,
        "projected_price_in_2h": 105.0,
        "predicted_scenario": "Price is expected to rise over the next two hours.",
    }


def valid_advisory() -> dict:
    return {
        "direction": "UP",
        "target_range_low": 90.0,
        "target_range_high": 110.0,
        "invalidation_price": 80.0,
        "confidence_score": 7,
        "market_regime": "Ranging",
        "reasoning": "Price is holding above support.",
    }


class ForecastingFixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_market_csv_appends_validated_rows_with_two_hour_timestamps(self) -> None:
        log_file = self.directory / "market.csv"
        issue_time = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
        metrics = {
            "current_price": 100.0,
            "source_candle_timestamp": "2026-01-01T11:45:00+00:00",
        }

        for _ in range(2):
            market_predictor.log_prediction(
                "BTC-USD",
                metrics,
                valid_market_forecast(),
                log_file=log_file,
                issued_at=issue_time,
            )

        with log_file.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["forecast_horizon_hours"], "2")
        self.assertEqual(
            rows[0]["target_timestamp"],
            "2026-01-01T14:00:00+00:00",
        )
        self.assertEqual(rows[0]["source_candle_timestamp"], metrics["source_candle_timestamp"])
        self.assertEqual(rows[0]["starting_btc_price"], "100.0")

    def test_invalid_or_incomplete_ai_forecasts_are_rejected(self) -> None:
        log_file = self.directory / "market.csv"
        prediction_logs.initialize_market_log(log_file)
        metrics = {
            "current_price": 100.0,
            "source_candle_timestamp": "2026-01-01T11:45:00+00:00",
        }
        missing_field = valid_market_forecast()
        missing_field.pop("confidence_score")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            market_predictor.validate_prediction(missing_field)
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            market_predictor.log_prediction(
                "BTC-USD",
                metrics,
                missing_field,
                log_file=log_file,
                issued_at=datetime.datetime(
                    2026, 1, 1, 12, 0, tzinfo=datetime.timezone.utc
                ),
            )

        inverted_range = valid_market_forecast()
        inverted_range["expected_low_next_2h"] = 120
        with self.assertRaisesRegex(ValueError, "low price exceeds"):
            market_predictor.validate_prediction(inverted_range)

        bad_direction = valid_market_forecast()
        bad_direction["predicted_direction"] = "UP"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            market_predictor.validate_prediction(bad_direction)

        bad_confidence = valid_market_forecast()
        bad_confidence["confidence_score"] = 11
        with self.assertRaisesRegex(ValueError, "confidence_score"):
            market_predictor.validate_prediction(bad_confidence)

        bad_advisory = valid_advisory()
        bad_advisory["target_range_low"] = 120
        with self.assertRaisesRegex(ValueError, "target low exceeds"):
            pro_strategist.validate_advisory(bad_advisory)
        with log_file.open(newline="", encoding="utf-8") as file:
            self.assertEqual(len(list(csv.reader(file))), 1)

    def test_missing_fields_do_not_append_a_csv_row(self) -> None:
        log_file = self.directory / "market.csv"
        prediction_logs.initialize_market_log(log_file)
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            prediction_logs.append_record(
                log_file,
                prediction_logs.MARKET_LOG_FIELDS,
                {"timestamp": "2026-01-01T12:00:00+00:00"},
            )
        with log_file.open(newline="", encoding="utf-8") as file:
            self.assertEqual(len(list(csv.reader(file))), 1)

    def test_mixed_legacy_rows_migrate_without_changing_original(self) -> None:
        legacy_file = self.directory / "predictions_log.csv"
        app_file = self.directory / "app.csv"
        market_file = self.directory / "market.csv"
        app_values = [
            "2026-01-01 10:00:00",
            "100",
            "UP",
            "95",
            "105",
            "90",
            "7",
            "Legacy",
            "Not graded",
            "Ranging",
            "Historical app forecast",
        ]
        market_values = [
            "2026-01-01 12:00:00",
            "BTC-USD",
            "100",
            "BULLISH",
            "7",
            "95",
            "110",
            "105",
            "Historical market forecast",
        ]
        with legacy_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(prediction_logs.LEGACY_APP_FIELDS)
            writer.writerow(app_values)
            writer.writerow(market_values)
        original_bytes = legacy_file.read_bytes()

        prediction_logs.initialize_app_log(app_file, legacy_file=legacy_file)
        prediction_logs.initialize_market_log(market_file, legacy_file=legacy_file)

        with app_file.open(newline="", encoding="utf-8") as file:
            app_rows = list(csv.DictReader(file))
        with market_file.open(newline="", encoding="utf-8") as file:
            market_rows = list(csv.DictReader(file))
        self.assertEqual(len(app_rows), 1)
        self.assertEqual(app_rows[0]["graded"], "Legacy")
        self.assertEqual(app_rows[0]["target_timestamp"], "")
        self.assertEqual(len(market_rows), 1)
        self.assertEqual(market_rows[0]["graded"], "Legacy")
        self.assertEqual(market_rows[0]["source_candle_timestamp"], "")
        self.assertEqual(legacy_file.read_bytes(), original_bytes)

    def test_target_time_must_match_horizon_and_source_candle(self) -> None:
        row = pd.Series(
            {
                "forecast_issue_timestamp": "2026-01-01T12:00:00+00:00",
                "source_candle_timestamp": "2026-01-01T11:45:00+00:00",
                "target_timestamp": "2026-01-01T14:00:00+00:00",
                "forecast_horizon_hours": 2,
            }
        )
        target = evaluator._validate_forecast_timestamps(row, 2)
        self.assertEqual(target, pd.Timestamp("2026-01-01T14:00:00+00:00"))

        row["target_timestamp"] = "2026-01-01T15:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluator._validate_forecast_timestamps(row, 2)

        row["target_timestamp"] = "2026-01-01T14:00:00+00:00"
        row["source_candle_timestamp"] = "2026-01-01T12:01:00+00:00"
        with self.assertRaisesRegex(ValueError, "later than"):
            evaluator._validate_forecast_timestamps(row, 2)

    def test_missing_evaluation_price_keeps_forecast_pending(self) -> None:
        log_file = self.directory / "market.csv"
        issue_time = pd.Timestamp.now(tz="UTC").floor("s") - pd.Timedelta(hours=3)
        source_time = issue_time - pd.Timedelta(minutes=15)
        target_time = issue_time + pd.Timedelta(hours=2)
        row = {
            "timestamp": issue_time.isoformat(),
            "forecast_issue_timestamp": issue_time.isoformat(),
            "source_candle_timestamp": source_time.isoformat(),
            "target_timestamp": target_time.isoformat(),
            "forecast_horizon_hours": 2,
            "ticker": "BTC-USD",
            "starting_btc_price": 100.0,
            "predicted_direction": "BULLISH",
            "confidence_score": 7,
            "expected_low_2h": 95.0,
            "expected_high_2h": 110.0,
            "projected_price_2h": 105.0,
            "scenario": "Test forecast",
            "graded": "False",
            "actual_price": "",
            "actual_price_timestamp": "",
            "actual_outcome": "Pending",
        }
        prediction_logs.initialize_market_log(log_file)
        prediction_logs.append_record(log_file, prediction_logs.MARKET_LOG_FIELDS, row)

        with patch.object(
            evaluator,
            "_fetch_price_at",
            side_effect=RuntimeError("No matching historical price."),
        ), patch.object(evaluator, "store_memory") as store_memory:
            summary = evaluator.grade_pending_predictions(log_file)

        with log_file.open(newline="", encoding="utf-8") as file:
            saved = next(csv.DictReader(file))
        self.assertEqual(summary["graded"], 0)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(saved["graded"], "False")
        self.assertEqual(saved["actual_outcome"], "Pending")
        store_memory.assert_not_called()

    def test_successful_grading_is_saved_and_not_repeated(self) -> None:
        log_file = self.directory / "market.csv"
        issue_time = pd.Timestamp.now(tz="UTC").floor("s") - pd.Timedelta(hours=3)
        target_time = issue_time + pd.Timedelta(hours=2)
        row = {
            "timestamp": issue_time.isoformat(),
            "forecast_issue_timestamp": issue_time.isoformat(),
            "source_candle_timestamp": (
                issue_time - pd.Timedelta(minutes=15)
            ).isoformat(),
            "target_timestamp": target_time.isoformat(),
            "forecast_horizon_hours": 2,
            "ticker": "BTC-USD",
            "starting_btc_price": 100.0,
            "predicted_direction": "BULLISH",
            "confidence_score": 7,
            "expected_low_2h": 95.0,
            "expected_high_2h": 110.0,
            "projected_price_2h": 105.0,
            "scenario": "Test forecast",
            "graded": "False",
            "actual_price": "",
            "actual_price_timestamp": "",
            "actual_outcome": "Pending",
        }
        prediction_logs.initialize_market_log(log_file)
        prediction_logs.append_record(log_file, prediction_logs.MARKET_LOG_FIELDS, row)
        price_timestamp = target_time - pd.Timedelta(minutes=1)

        with patch.object(
            evaluator,
            "_fetch_price_at",
            return_value=(101.0, price_timestamp),
        ), patch.object(evaluator, "store_memory") as store_memory:
            first_result = evaluator.grade_pending_predictions(log_file)
            second_result = evaluator.grade_pending_predictions(log_file)

        with log_file.open(newline="", encoding="utf-8") as file:
            saved = next(csv.DictReader(file))
        self.assertEqual(
            first_result,
            {"graded": 1, "memorized": 1, "errors": 0, "pending": 0},
        )
        self.assertEqual(
            second_result,
            {"graded": 0, "memorized": 0, "errors": 0, "pending": 0},
        )
        self.assertEqual(saved["graded"], "True")
        self.assertEqual(saved["actual_price"], "101.0")
        self.assertEqual(saved["actual_price_timestamp"], price_timestamp.isoformat())
        self.assertEqual(saved["actual_outcome"], "Win")
        store_memory.assert_called_once()

    def test_price_lookup_uses_one_minute_close_at_target(self) -> None:
        target_time = pd.Timestamp("2026-01-01T12:30:00+00:00")
        history = pd.DataFrame(
            {"Close": [123.0]},
            index=pd.DatetimeIndex([target_time - pd.Timedelta(minutes=1)]),
        )
        with patch.object(evaluator.yf, "download", return_value=history) as download:
            price, price_timestamp = evaluator._fetch_price_at(target_time)
        self.assertEqual(price, 123.0)
        self.assertEqual(price_timestamp, target_time)
        self.assertEqual(download.call_args.kwargs["interval"], "1m")

    def test_stale_price_is_not_used_to_grade(self) -> None:
        target_time = pd.Timestamp("2026-01-01T12:30:00+00:00")
        stale_history = pd.DataFrame(
            {"Close": [123.0]},
            index=pd.DatetimeIndex([target_time - pd.Timedelta(minutes=3)]),
        )
        with patch.object(evaluator.yf, "download", return_value=stale_history):
            with self.assertRaisesRegex(RuntimeError, "within one minute"):
                evaluator._fetch_price_at(target_time)

    def test_incomplete_candles_are_excluded(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        older_candle = now.floor("15min") - pd.Timedelta(minutes=30)
        previous_candle = now.floor("15min") - pd.Timedelta(minutes=15)
        forming_candle = now.floor("15min")
        candles = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [102.0, 103.0, 104.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [101.0, 102.0, 103.0],
                "Volume": [1.0, 1.0, 1.0],
            },
            index=pd.DatetimeIndex([older_candle, previous_candle, forming_candle]),
        )
        with patch.object(market_predictor.yf, "download", return_value=candles):
            confirmed = market_predictor.download_clean_df(
                "BTC-USD",
                interval="15m",
                period="1d",
            )
        self.assertEqual(len(confirmed), 2)
        self.assertEqual(confirmed.index[-1], previous_candle)

        confirmed_spot = data_engine._confirmed_candles(candles, "15m")
        self.assertEqual(len(confirmed_spot), 2)
        self.assertEqual(confirmed_spot.index[-1], previous_candle)

    def test_empty_and_insufficient_candle_data_fail_safely(self) -> None:
        empty_candles = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([]),
        )
        with patch.object(market_predictor.yf, "download", return_value=empty_candles):
            with self.assertRaisesRegex(RuntimeError, "No 15m candle data"):
                market_predictor.download_clean_df("BTC-USD", "15m", "1d")

        with self.assertRaisesRegex(ValueError, "Insufficient confirmed"):
            market_predictor.calculate_metrics(
                pd.DataFrame(columns=["Close", "High", "Low"]),
                pd.DataFrame(columns=["Close"]),
            )
        with self.assertRaisesRegex(RuntimeError, "no BTC-USD spot candles"):
            data_engine._confirmed_candles(empty_candles, "4h")


if __name__ == "__main__":
    unittest.main()
