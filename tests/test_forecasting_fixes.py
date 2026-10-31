import csv
import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import app, auto_run, evaluator, market_predictor, prediction_logs


def valid_market_forecast() -> dict:
    return {
        "predicted_direction": "BULLISH",
        "confidence_score": 7,
        "expected_high_next_2h": 110.0,
        "expected_low_next_2h": 90.0,
        "projected_price_in_2h": 105.0,
        "predicted_scenario": "BTC is expected to rise over the next two hours.",
    }


def forecast_row(issue_time: pd.Timestamp, direction: str = "BULLISH") -> dict[str, str]:
    return {
        "forecast_id": "forecast-test-001",
        "forecast_issue_timestamp": issue_time.isoformat(),
        "ticker": "BTC-USD",
        "starting_btc_price": "100",
        "source_candle_timestamp": (issue_time - pd.Timedelta(minutes=15)).isoformat(),
        "forecast_horizon_hours": "2",
        "intended_evaluation_timestamp": (
            issue_time + pd.Timedelta(hours=2)
        ).isoformat(),
        "predicted_direction": direction,
        "expected_high": "110",
        "expected_low": "90",
        "projected_price": "105",
        "confidence_score": "7",
        "scenario": "Test forecast scenario.",
        "evaluation_status": "PENDING",
        "actual_btc_close": "",
        "actual_close_timestamp": "",
        "actual_high": "",
        "actual_low": "",
        "direction_correct": "",
        "final_close_in_predicted_range": "",
        "full_period_range_inside_predicted_range": "",
        "projected_price_absolute_error": "",
        "projected_price_percentage_error": "",
        "evaluated_at_utc": "",
        "evaluation_note": "",
        "legacy_source_file": "",
        "legacy_source_row": "",
    }


def candle_data(issue_time: pd.Timestamp, close_price: float = 108.0):
    index_15m = pd.date_range(issue_time, periods=8, freq="15min")
    candles_15m = pd.DataFrame(
        {
            "Open": [100.0] * 8,
            "High": [109.0] * 8,
            "Low": [95.0] * 8,
            "Close": [101.0] * 7 + [close_price],
        },
        index=index_15m,
    )
    index_1m = pd.date_range(issue_time, periods=120, freq="min")
    candles_1m = pd.DataFrame(
        {
            "Open": [100.0] * 120,
            "High": [109.0] * 120,
            "Low": [95.0] * 120,
            "Close": [101.0] * 119 + [close_price],
        },
        index=index_1m,
    )
    return candles_15m, candles_1m


class ForecastLoggingAndEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_gemini_forecast_is_normalized_and_invalid_values_rejected(self) -> None:
        valid = valid_market_forecast()
        valid["predicted_direction"] = " bullish "
        self.assertEqual(
            market_predictor.validate_prediction(valid)["predicted_direction"],
            "BULLISH",
        )

        invalid_values = []
        missing = valid_market_forecast()
        del missing["predicted_scenario"]
        invalid_values.append((missing, "missing required fields"))
        invalid_direction = valid_market_forecast()
        invalid_direction["predicted_direction"] = "UP"
        invalid_values.append((invalid_direction, "unsupported"))
        invalid_price = valid_market_forecast()
        invalid_price["projected_price_in_2h"] = 0
        invalid_values.append((invalid_price, "positive number"))
        nonnumeric = valid_market_forecast()
        nonnumeric["expected_high_next_2h"] = "110"
        invalid_values.append((nonnumeric, "positive number"))
        inverted_range = valid_market_forecast()
        inverted_range["expected_low_next_2h"] = 120
        invalid_values.append((inverted_range, "low price exceeds"))
        invalid_confidence = valid_market_forecast()
        invalid_confidence["confidence_score"] = 11
        invalid_values.append((invalid_confidence, "confidence_score"))
        for value, message in invalid_values:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                market_predictor.validate_prediction(value)

    def test_bearish_range_is_width_limited_and_anchored_at_current_price(self) -> None:
        prediction = valid_market_forecast()
        prediction.update(
            {
                "predicted_direction": "BEARISH",
                "expected_high_next_2h": 80000.0,
                "expected_low_next_2h": 79000.0,
                "projected_price_in_2h": 79100.0,
            }
        )

        result = market_predictor.validate_prediction(
            prediction, {"current_price": 79800.0}
        )

        self.assertEqual(result["expected_high_next_2h"], 79850.0)
        self.assertLessEqual(result["expected_low_next_2h"], 79800.0)
        self.assertLessEqual(
            result["expected_high_next_2h"] - result["expected_low_next_2h"],
            market_predictor.MAX_EXPECTED_RANGE_WIDTH,
        )

    def test_clamp_prediction_range_handles_directional_and_centered_ranges(self) -> None:
        bearish_low, bearish_high = market_predictor.clamp_prediction_range(
            "BEARISH", 79800.0, 79000.0, 80000.0
        )
        self.assertLessEqual(bearish_high, 79800.0 + 50.0)
        self.assertLessEqual(bearish_high - bearish_low, 450.0)

        bullish_low, bullish_high = market_predictor.clamp_prediction_range(
            "BULLISH", 79800.0, 79000.0, 81000.0
        )
        self.assertGreaterEqual(bullish_low, 79800.0 - 50.0)
        self.assertLessEqual(bullish_high - bullish_low, 450.0)

        sideways_low, sideways_high = market_predictor.clamp_prediction_range(
            "SIDEWAYS", 79800.0, 79000.0, 81000.0
        )
        self.assertEqual((sideways_low + sideways_high) / 2.0, 79800.0)
        self.assertLessEqual(sideways_high - sideways_low, 450.0)

    def test_clamp_prediction_range_rejects_invalid_values(self) -> None:
        for arguments in (
            ("BULLISH", 0.0, 90.0, 110.0),
            ("BEARISH", 100.0, 110.0, 90.0),
            ("SIDEWAYS", 100.0, 90.0, 110.0, 0.0),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                market_predictor.clamp_prediction_range(*arguments)

    def test_bullish_range_is_width_limited_and_anchored_at_current_price(self) -> None:
        prediction = valid_market_forecast()
        prediction.update(
            {
                "expected_high_next_2h": 81000.0,
                "expected_low_next_2h": 79000.0,
                "projected_price_in_2h": 80900.0,
            }
        )

        result = market_predictor.validate_prediction(
            prediction, {"current_price": 79800.0}
        )

        self.assertEqual(result["expected_low_next_2h"], 79750.0)
        self.assertGreaterEqual(result["expected_high_next_2h"], 79800.0)
        self.assertLessEqual(
            result["expected_high_next_2h"] - result["expected_low_next_2h"],
            market_predictor.MAX_EXPECTED_RANGE_WIDTH,
        )

    def test_sideways_and_wait_ranges_are_centered_and_width_limited(self) -> None:
        for update in (
            {"predicted_direction": "SIDEWAYS"},
            {"action": "WAIT"},
        ):
            prediction = valid_market_forecast()
            prediction.update(
                {
                    **update,
                    "expected_high_next_2h": 81000.0,
                    "expected_low_next_2h": 79000.0,
                }
            )
            with self.subTest(update=update):
                result = market_predictor.validate_prediction(
                    prediction, {"current_price": 79800.0}
                )
                self.assertEqual(
                    (result["expected_high_next_2h"] + result["expected_low_next_2h"])
                    / 2,
                    79800.0,
                )
                self.assertLessEqual(
                    result["expected_high_next_2h"]
                    - result["expected_low_next_2h"],
                    market_predictor.MAX_EXPECTED_RANGE_WIDTH,
                )

    def test_range_validation_rejects_missing_or_invalid_current_price(self) -> None:
        for metrics in ({}, {"current_price": float("nan")}, {"current_price": 0}):
            with self.subTest(metrics=metrics), self.assertRaisesRegex(
                ValueError, "Current BTC price"
            ):
                market_predictor.validate_prediction(valid_market_forecast(), metrics)

    def test_forecast_writes_unique_utc_two_hour_record_to_its_own_schema(self) -> None:
        log_file = self.directory / "btc_forecasts_2h.csv"
        issue_time = datetime.datetime(
            2026, 1, 1, 12, 0, tzinfo=datetime.timezone.utc
        )
        metrics = {
            "current_price": 100.0,
            "source_candle_timestamp": "2026-01-01T11:45:00+00:00",
        }
        market_predictor.log_prediction(
            "BTC-USD",
            metrics,
            valid_market_forecast(),
            log_file=log_file,
            issued_at=issue_time,
        )
        market_predictor.log_prediction(
            "BTC-USD",
            metrics,
            valid_market_forecast(),
            log_file=log_file,
            issued_at=issue_time,
        )
        with log_file.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, prediction_logs.BTC_FORECAST_FIELDS)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["forecast_id"], rows[1]["forecast_id"])
        self.assertEqual(rows[0]["forecast_issue_timestamp"], issue_time.isoformat())
        self.assertEqual(
            rows[0]["intended_evaluation_timestamp"],
            "2026-01-01T14:00:00+00:00",
        )
        self.assertEqual(rows[0]["forecast_horizon_hours"], "2")
        self.assertEqual(rows[0]["evaluation_status"], "PENDING")
        self.assertEqual(rows[0]["scenario"], valid_market_forecast()["predicted_scenario"])

    def test_source_candle_timestamp_records_the_candle_close(self) -> None:
        index = pd.date_range(
            "2026-01-01T00:00:00+00:00", periods=60, freq="15min"
        )
        closes = pd.Series(range(100, 160), index=index, dtype=float)
        micro = pd.DataFrame(
            {
                "Open": closes,
                "High": closes + 1,
                "Low": closes - 1,
                "Close": closes,
                "Volume": 1.0,
            },
            index=index,
        )
        macro = pd.DataFrame({"Close": closes})
        metrics = market_predictor.calculate_metrics(micro, macro)
        self.assertEqual(
            pd.Timestamp(metrics["source_candle_timestamp"]),
            index[-1] + pd.Timedelta(minutes=15),
        )

    def test_wrong_ticker_and_naive_timestamps_are_rejected(self) -> None:
        metrics = {
            "current_price": 100.0,
            "source_candle_timestamp": "2026-01-01T11:45:00+00:00",
        }
        with self.assertRaisesRegex(ValueError, "BTC-USD"):
            market_predictor.log_prediction(
                "ETH-USD", metrics, valid_market_forecast(),
                log_file=self.directory / "btc.csv",
                issued_at=datetime.datetime(2026, 1, 1, 12, tzinfo=datetime.timezone.utc),
            )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            prediction_logs.to_utc_iso(datetime.datetime(2026, 1, 1, 12))

    def test_advisory_and_btc_logs_are_separate_and_schema_checked(self) -> None:
        advisory_file = self.directory / "advisory_log.csv"
        btc_file = self.directory / "btc_forecasts_2h.csv"
        prediction_logs.initialize_advisory_log(advisory_file)
        prediction_logs.initialize_btc_forecast_log(btc_file)
        self.assertNotEqual(advisory_file, btc_file)
        with advisory_file.open(newline="", encoding="utf-8") as file:
            self.assertEqual(
                next(csv.reader(file)), prediction_logs.ADVISORY_LOG_FIELDS
            )
        with btc_file.open(newline="", encoding="utf-8") as file:
            self.assertEqual(next(csv.reader(file)), prediction_logs.BTC_FORECAST_FIELDS)
        self.assertEqual(app.HISTORY_FILE, prediction_logs.ADVISORY_LOG_FILE)
        self.assertEqual(auto_run.CSV_FILENAME, str(prediction_logs.ADVISORY_LOG_FILE))
        with self.assertRaisesRegex(ValueError, "unexpected CSV header"):
            prediction_logs.append_record(
                advisory_file,
                prediction_logs.BTC_FORECAST_FIELDS,
                {field: "" for field in prediction_logs.BTC_FORECAST_FIELDS},
            )

    def test_dashboard_pipeline_appends_only_to_advisory_log(self) -> None:
        advisory_file = self.directory / "advisory_log.csv"
        spot_data = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.DatetimeIndex(["2026-01-01T11:45:00+00:00"]),
        )
        decision = {
            "direction": "UP",
            "target_range_low": 95.0,
            "target_range_high": 110.0,
            "invalidation_price": 90.0,
            "confidence_score": 7,
            "market_regime": "Ranging",
            "reasoning": "Dashboard advisory test.",
        }
        with patch.object(app, "HISTORY_FILE", advisory_file):
            app.log_prediction(decision, {"spot": {"current_close": 100.0}}, spot_data)
        with advisory_file.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            row = next(reader)
            self.assertEqual(reader.fieldnames, prediction_logs.ADVISORY_LOG_FIELDS)
        self.assertEqual(row["direction"], "UP")
        self.assertEqual(row["forecast_horizon_hours"], "4")
        self.assertEqual(row["reasoning"], "Dashboard advisory test.")

    def test_legacy_records_are_backed_up_and_not_assumed_evaluable(self) -> None:
        logs = self.directory / "logs"
        backups = logs / "backups"
        logs.mkdir()
        backups.mkdir()
        old_market = logs / "market_predictions_log.csv"
        old_app = logs / "app_predictions_log.csv"
        old_shared = logs / "predictions_log.csv"
        with old_market.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=prediction_logs.LEGACY_MARKET_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp": "2026-01-01T12:00:00+00:00",
                    "forecast_issue_timestamp": "2026-01-01T12:00:00+00:00",
                    "source_candle_timestamp": "",
                    "target_timestamp": "",
                    "forecast_horizon_hours": "2",
                    "ticker": "BTC-USD",
                    "starting_btc_price": "100",
                    "predicted_direction": "BULLISH",
                    "confidence_score": "7",
                    "expected_low_2h": "90",
                    "expected_high_2h": "110",
                    "projected_price_2h": "105",
                    "scenario": "Old forecast",
                    "graded": "False",
                    "actual_price": "",
                    "actual_price_timestamp": "",
                    "actual_outcome": "Pending",
                }
            )
        with old_app.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=prediction_logs.APP_LOG_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp": "2026-01-01 10:00:00",
                    "direction": "UP",
                    "reasoning": "Old advisory",
                }
            )
        with old_shared.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(prediction_logs.LEGACY_APP_FIELDS)
            writer.writerow(
                [
                    "2026-01-01 10:00:00",
                    "BTC-USD",
                    "100",
                    "BULLISH",
                    "7",
                    "90",
                    "110",
                    "105",
                    "Legacy market-format row",
                ]
            )
        original_market_bytes = old_market.read_bytes()
        original_app_bytes = old_app.read_bytes()

        forecast_path = logs / "btc_forecasts_2h.csv"
        advisory_path = logs / "advisory_log.csv"
        with (
            patch.object(prediction_logs, "LOGS_DIR", logs),
            patch.object(prediction_logs, "BTC_FORECAST_FILE", forecast_path),
            patch.object(prediction_logs, "ADVISORY_LOG_FILE", advisory_path),
            patch.object(prediction_logs, "LEGACY_MARKET_LOG_FILE", old_market),
            patch.object(prediction_logs, "LEGACY_APP_LOG_FILE", old_app),
            patch.object(prediction_logs, "LEGACY_SHARED_LOG_FILE", old_shared),
        ):
            prediction_logs.initialize_btc_forecast_log()
            prediction_logs.initialize_advisory_log()

        with forecast_path.open(newline="", encoding="utf-8") as file:
            forecasts = list(csv.DictReader(file))
        with advisory_path.open(newline="", encoding="utf-8") as file:
            advisories = list(csv.DictReader(file))
        self.assertEqual(len(forecasts), 1)
        self.assertEqual(forecasts[0]["evaluation_status"], "NEEDS_REVIEW")
        self.assertEqual(forecasts[0]["source_candle_timestamp"], "")
        self.assertEqual(len(advisories), 1)
        self.assertEqual(advisories[0]["forecast_horizon_hours"], "")
        self.assertEqual(old_market.read_bytes(), original_market_bytes)
        self.assertEqual(old_app.read_bytes(), original_app_bytes)
        self.assertEqual(
            (backups / "market_predictions_log.csv.pre-phase1.bak").read_bytes(),
            original_market_bytes,
        )
        self.assertEqual(
            (backups / "app_predictions_log.csv.pre-phase1.bak").read_bytes(),
            original_app_bytes,
        )

    def test_target_timestamp_requires_aware_utc_and_two_hour_horizon(self) -> None:
        issue = pd.Timestamp("2026-01-01T12:00:00+00:00")
        row = forecast_row(issue)
        parsed_issue, target = evaluator._validate_forecast_timestamps(row)
        self.assertEqual(parsed_issue, issue)
        self.assertEqual(target, issue + pd.Timedelta(hours=2))
        row["forecast_issue_timestamp"] = "2026-01-01 12:00:00"
        with self.assertRaisesRegex(ValueError, "timezone"):
            evaluator._validate_forecast_timestamps(row)

    def test_direction_close_range_and_price_error_are_scored_separately(self) -> None:
        issue = pd.Timestamp("2026-01-01T12:00:00+00:00")
        candles_15m, candles_1m = candle_data(issue, close_price=108.0)
        result = evaluator.evaluate_forecast(
            forecast_row(issue, "BULLISH"), candles_15m, candles_1m
        )
        self.assertTrue(result["direction_correct"])
        self.assertTrue(result["final_close_in_predicted_range"])
        self.assertTrue(result["full_period_range_inside_predicted_range"])
        self.assertEqual(result["projected_price_absolute_error"], 3.0)
        self.assertAlmostEqual(result["projected_price_percentage_error"], 100 * 3 / 105)
        self.assertEqual(result["actual_close_timestamp"], (issue + pd.Timedelta(hours=2)).isoformat())

        wrong = evaluator.evaluate_forecast(
            forecast_row(issue, "BEARISH"), candles_15m, candles_1m
        )
        self.assertFalse(wrong["direction_correct"])

    def test_final_close_and_full_period_range_are_distinct(self) -> None:
        issue = pd.Timestamp("2026-01-01T12:00:00+00:00")
        candles_15m, candles_1m = candle_data(issue, close_price=108.0)
        candles_15m.iloc[2, candles_15m.columns.get_loc("High")] = 115.0
        result = evaluator.evaluate_forecast(
            forecast_row(issue), candles_15m, candles_1m
        )
        self.assertTrue(result["final_close_in_predicted_range"])
        self.assertFalse(result["full_period_range_inside_predicted_range"])

    def test_sideways_direction_is_not_applicable_and_range_is_scored(self) -> None:
        issue = pd.Timestamp("2026-01-01T12:00:00+00:00")
        candles_15m, candles_1m = candle_data(issue, close_price=108.0)
        result = evaluator.evaluate_forecast(
            forecast_row(issue, "SIDEWAYS"), candles_15m, candles_1m
        )
        self.assertEqual(result["direction_correct"], "")
        self.assertTrue(result["final_close_in_predicted_range"])

    def test_missing_or_misaligned_candles_are_not_graded(self) -> None:
        issue = pd.Timestamp("2026-01-01T12:00:00+00:00")
        candles_15m, candles_1m = candle_data(issue)
        candles_15m = candles_15m.drop(candles_15m.index[3])
        candles_1m = candles_1m.drop(candles_1m.index[45:60])
        with self.assertRaisesRegex(evaluator.EvaluationDataUnavailable, "Missing"):
            evaluator.evaluate_forecast(forecast_row(issue), candles_15m, candles_1m)

    def test_unaligned_expiry_uses_timestamped_minute_close(self) -> None:
        issue = pd.Timestamp("2026-01-01T12:05:30+00:00")
        target = issue + pd.Timedelta(hours=2)
        row = forecast_row(issue)
        candles_1m = pd.DataFrame(
            {
                "Open": [100.0] * 120,
                "High": [109.0] * 120,
                "Low": [95.0] * 120,
                "Close": [101.0] * 119 + [108.0],
            },
            index=pd.date_range(
                issue.floor("min"), periods=120, freq="min"
            ),
        )
        candles_15m = pd.DataFrame(
            columns=["Open", "High", "Low", "Close"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        result = evaluator.evaluate_forecast(row, candles_15m, candles_1m)
        self.assertEqual(
            result["actual_close_timestamp"],
            (target.floor("min")).isoformat(),
        )
        self.assertIn("latest minute close", result["evaluation_note"])

    def test_not_due_forecasts_are_left_pending_without_market_fetch(self) -> None:
        path = self.directory / "btc.csv"
        prediction_logs.initialize_btc_forecast_log(path)
        issue = pd.Timestamp.now(tz="UTC").floor("s")
        row = forecast_row(issue)
        with path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=prediction_logs.BTC_FORECAST_FIELDS).writerow(row)
        with patch.object(evaluator, "_fetch_evaluation_candles") as fetch:
            result = evaluator.grade_pending_predictions(
                path, now=issue + pd.Timedelta(hours=1)
            )
        fetch.assert_not_called()
        self.assertEqual(result["pending"], 1)
        with path.open(newline="", encoding="utf-8") as file:
            self.assertEqual(next(csv.DictReader(file))["evaluation_status"], "PENDING")

    def test_csv_is_updated_once_and_rag_upsert_uses_forecast_id(self) -> None:
        path = self.directory / "btc.csv"
        prediction_logs.initialize_btc_forecast_log(path)
        issue = pd.Timestamp("2026-01-01T12:00:00+00:00")
        row = forecast_row(issue)
        with path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=prediction_logs.BTC_FORECAST_FIELDS).writerow(row)
        candles_15m, candles_1m = candle_data(issue)
        now = issue + pd.Timedelta(hours=3)
        with patch.object(
            evaluator, "_fetch_evaluation_candles", return_value=(candles_15m, candles_1m)
        ), patch.object(evaluator, "store_memory") as store_memory:
            first = evaluator.grade_pending_predictions(path, now=now)
            second = evaluator.grade_pending_predictions(path, now=now)
        self.assertEqual(first["graded"], 1)
        self.assertEqual(first["memorized"], 1)
        self.assertEqual(second["graded"], 0)
        self.assertEqual(store_memory.call_count, 1)
        self.assertEqual(store_memory.call_args.args[0]["forecast_id"], row["forecast_id"])
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            saved = next(reader)
            self.assertEqual(reader.fieldnames, prediction_logs.BTC_FORECAST_FIELDS)
        self.assertEqual(saved["evaluation_status"], "EVALUATED")
        self.assertEqual(saved["actual_btc_close"], "108.0")
        self.assertEqual(saved["final_close_in_predicted_range"], "True")

    def test_rag_memory_upserts_by_stable_forecast_id(self) -> None:
        row = forecast_row(pd.Timestamp("2026-01-01T12:00:00+00:00"))
        row.update(
            {
                "actual_btc_close": "108",
                "actual_close_timestamp": "2026-01-01T14:00:00+00:00",
                "actual_high": "109",
                "actual_low": "95",
                "direction_correct": "True",
                "final_close_in_predicted_range": "True",
                "full_period_range_inside_predicted_range": "True",
                "evaluation_status": "EVALUATED",
            }
        )
        with patch.object(evaluator, "TRADE_MEMORIES") as collection:
            evaluator.store_memory(row)
            evaluator.store_memory(row)
        self.assertEqual(collection.upsert.call_count, 2)
        self.assertEqual(collection.upsert.call_args_list[0].kwargs["ids"], [
            row["forecast_id"]
        ])
        self.assertEqual(
            collection.upsert.call_args_list[1].kwargs["ids"],
            [row["forecast_id"]],
        )

    def test_unreliable_legacy_row_gets_needs_review_not_a_fake_grade(self) -> None:
        path = self.directory / "btc.csv"
        prediction_logs.initialize_btc_forecast_log(path)
        row = forecast_row(pd.Timestamp("2026-01-01T12:00:00+00:00"))
        row["source_candle_timestamp"] = ""
        row["starting_btc_price"] = ""
        with path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=prediction_logs.BTC_FORECAST_FIELDS).writerow(row)
        with patch.object(evaluator, "_fetch_evaluation_candles") as fetch:
            summary = evaluator.grade_pending_predictions(
                path, now=pd.Timestamp("2026-01-01T15:00:00+00:00")
            )
        fetch.assert_not_called()
        self.assertEqual(summary["needs_review"], 1)
        with path.open(newline="", encoding="utf-8") as file:
            saved = next(csv.DictReader(file))
        self.assertEqual(saved["evaluation_status"], "NEEDS_REVIEW")
        self.assertEqual(saved["actual_btc_close"], "")


if __name__ == "__main__":
    unittest.main()
