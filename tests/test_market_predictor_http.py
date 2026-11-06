import json
import datetime
import email.utils
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src import market_predictor


class MarketPredictorHttpTests(unittest.TestCase):
    def test_binance_spot_price_is_used_only_when_quote_is_fresh(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"symbol": "BTCUSDT", "price": "65000.25"}
        response.headers = {
            "Date": email.utils.format_datetime(
                datetime.datetime.now(datetime.timezone.utc),
                usegmt=True,
            )
        }

        with patch.object(market_predictor.requests, "get", return_value=response) as get:
            price, quoted_at = market_predictor.fetch_binance_spot_price()

        get.assert_called_once_with(
            market_predictor.BINANCE_SPOT_URL,
            params={"symbol": "BTCUSDT"},
            timeout=market_predictor.MARKET_DATA_TIMEOUT_SECONDS,
        )
        self.assertEqual(price, 65000.25)
        self.assertLessEqual(
            abs(
                (
                    datetime.datetime.now(datetime.timezone.utc) - quoted_at
                ).total_seconds()
            ),
            market_predictor.BINANCE_PRICE_MAX_AGE_SECONDS,
        )

    def test_binance_spot_price_rejects_stale_or_unverifiable_quotes(self):
        stale_response = Mock()
        stale_response.raise_for_status.return_value = None
        stale_response.json.return_value = {"price": "65000.25"}
        stale_response.headers = {
            "Date": email.utils.format_datetime(
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(seconds=market_predictor.BINANCE_PRICE_MAX_AGE_SECONDS + 10),
                usegmt=True,
            )
        }
        with (
            patch.object(market_predictor.requests, "get", return_value=stale_response),
            self.assertRaisesRegex(RuntimeError, "stale or future-dated"),
        ):
            market_predictor.fetch_binance_spot_price()

        stale_response.headers = {}
        with (
            patch.object(market_predictor.requests, "get", return_value=stale_response),
            self.assertRaisesRegex(RuntimeError, "no Date header"),
        ):
            market_predictor.fetch_binance_spot_price()

    def test_run_predictor_uses_live_binance_quote_as_forecast_baseline(self):
        execution_timestamp = datetime.datetime(
            2026, 1, 1, 12, 0, tzinfo=datetime.timezone.utc
        )
        metrics = {
            "current_price": 99.0,
            "source_candle_timestamp": "2026-01-01T11:45:00+00:00",
        }
        forecast = {
            "predicted_direction": "BULLISH",
            "confidence_score": 7,
            "expected_high_next_2h": 110.0,
            "expected_low_next_2h": 90.0,
            "projected_price_in_2h": 105.0,
            "predicted_scenario": "BTC may rise over the next two hours.",
        }
        with (
            patch.object(
                market_predictor,
                "generate_dual_chart",
                return_value=(object(), object(), "chart.png"),
            ),
            patch.object(market_predictor, "calculate_metrics", return_value=metrics),
            patch.object(
                market_predictor,
                "fetch_binance_spot_price",
                return_value=(100.123456, execution_timestamp),
            ),
            patch.object(
                market_predictor,
                "predict_next_2_hours",
                return_value=forecast,
            ),
            patch.object(market_predictor, "log_prediction", return_value={"saved": True}) as log,
        ):
            result = market_predictor.run_predictor()

        self.assertEqual(result, {"saved": True})
        self.assertEqual(metrics["current_price"], 100.123456)
        self.assertEqual(metrics["btc_price_at_prediction"], 100.123456)
        self.assertEqual(metrics["execution_timestamp"], execution_timestamp.isoformat())
        self.assertEqual(log.call_args.kwargs["issued_at"], execution_timestamp)

    def test_gemini_request_uses_the_shared_client_factory(self):
        prediction = {
            "predicted_direction": "BULLISH",
            "confidence_score": 7,
            "expected_high_next_2h": 110.0,
            "expected_low_next_2h": 90.0,
            "projected_price_in_2h": 105.0,
            "predicted_scenario": "BTC may rise over the next two hours.",
        }
        client = Mock()
        client.models.generate_content.return_value = Mock(
            text=json.dumps(prediction)
        )

        with (
            patch.object(
                market_predictor.gemini_utils,
                "create_client",
                return_value=client,
            ) as factory,
            patch.object(Image, "open", return_value=object()),
        ):
            result = market_predictor.predict_next_2_hours(
                "forecast.png",
                {"current_price": 100.0},
                "BTC-USD",
            )

        factory.assert_called_once_with()
        self.assertEqual(result["predicted_direction"], "BULLISH")

    def test_two_hour_prompt_includes_master_playbook_and_two_hour_lessons(self):
        prediction = {
            "predicted_direction": "BULLISH",
            "confidence_score": 7,
            "expected_high_next_2h": 110.0,
            "expected_low_next_2h": 90.0,
            "projected_price_in_2h": 105.0,
            "predicted_scenario": "BTC may rise over the next two hours.",
        }
        client = Mock()
        client.models.generate_content.return_value = Mock(text=json.dumps(prediction))
        with (
            patch.object(market_predictor.gemini_utils, "create_client", return_value=client),
            patch.object(Image, "open", return_value=object()),
            patch(
                "src.pro_strategist.load_playbook",
                return_value="EVOLVED MASTER RULES",
            ),
            patch(
                "src.evaluator.retrieve_past_lessons",
                return_value="2H LOSS LESSONS",
            ) as lessons,
        ):
            market_predictor.predict_next_2_hours(
                "forecast.png",
                {"current_price": 100.0, "macro_trend": "BULLISH"},
                "BTC-USD",
            )

        lessons.assert_called_once_with("BULLISH", forecast_horizon_hours=2)
        contents = client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("EVOLVED MASTER RULES", contents[0])
        self.assertIn("2H LOSS LESSONS", contents[1])

    def test_gemini_busy_response_retries_with_exponential_backoff(self):
        prediction = {
            "predicted_direction": "BULLISH",
            "confidence_score": 7,
            "expected_high_next_2h": 110.0,
            "expected_low_next_2h": 90.0,
            "projected_price_in_2h": 105.0,
            "predicted_scenario": "BTC may rise over the next two hours.",
        }
        client = Mock()
        client.models.generate_content.side_effect = [
            RuntimeError("504 DEADLINE_EXCEEDED"),
            Mock(text=json.dumps(prediction)),
        ]

        with (
            patch.object(market_predictor.gemini_utils, "create_client", return_value=client),
            patch.object(Image, "open", return_value=object()),
            patch.object(market_predictor.gemini_utils.time, "sleep") as sleep,
            patch("builtins.print") as print_call,
        ):
            result = market_predictor.predict_next_2_hours(
                "forecast.png",
                {"current_price": 100.0},
                "BTC-USD",
            )

        self.assertEqual(result["predicted_direction"], "BULLISH")
        self.assertEqual(client.models.generate_content.call_count, 2)
        sleep.assert_called_once_with(5)
        self.assertEqual(
            [call.args[0] for call in print_call.call_args_list],
            [
                "[API] Gemini request attempt 1/3 failed: "
                "type=RuntimeError, message=504 DEADLINE_EXCEEDED; "
                "retrying in 5 seconds."
            ],
        )

    def test_retryable_errors_stop_after_three_attempts(self):
        client = Mock()
        client.models.generate_content.side_effect = [
            RuntimeError("HTTP 503 unavailable"),
            RuntimeError("HTTP 429 too many requests"),
            RuntimeError("HTTP 504 deadline exceeded"),
        ]

        with (
            patch.object(market_predictor.gemini_utils, "create_client", return_value=client),
            patch.object(Image, "open", return_value=object()),
            patch.object(market_predictor.gemini_utils.time, "sleep") as sleep,
            self.assertRaisesRegex(
                RuntimeError,
                "Gemini API request failed: type=RuntimeError, .*"
                "message=HTTP 504 deadline exceeded",
            ),
        ):
            market_predictor.predict_next_2_hours(
                "forecast.png",
                {"current_price": 100.0},
                "BTC-USD",
            )

        self.assertEqual(client.models.generate_content.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [5, 10],
        )

    def test_invalid_json_and_empty_gemini_response_are_rejected(self):
        for response, error in (
            (Mock(text="{invalid"), "not a valid JSON forecast"),
            (Mock(text=""), "response was empty"),
        ):
            with self.subTest(error=error):
                client = Mock()
                client.models.generate_content.return_value = response
                with (
                    patch.object(
                        market_predictor.gemini_utils,
                        "create_client",
                        return_value=client,
                    ),
                    patch.object(Image, "open", return_value=object()),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    market_predictor.predict_next_2_hours(
                        "forecast.png",
                        {"current_price": 100.0},
                        "BTC-USD",
                    )

    def test_http_statuses_and_connection_timeouts_are_retryable(self):
        for status_code in (429, 502, 503, 504):
            error = RuntimeError("transient API error")
            error.code = status_code
            with self.subTest(status_code=status_code):
                self.assertTrue(market_predictor.gemini_utils._is_retryable(error))

        self.assertTrue(
            market_predictor.gemini_utils._is_retryable(
                TimeoutError("connection timed out")
            )
        )

    def test_permanent_gemini_error_is_not_retried_and_is_diagnosed(self):
        client = Mock()
        client.models.generate_content.side_effect = RuntimeError(
            "invalid API key: sensitive-value"
        )

        with (
            patch.object(market_predictor.gemini_utils, "create_client", return_value=client),
            patch.object(Image, "open", return_value=object()),
            patch.object(market_predictor.gemini_utils.time, "sleep") as sleep,
            patch("builtins.print") as print_call,
            self.assertRaisesRegex(
                RuntimeError,
                "Gemini API request failed: type=RuntimeError, .*"
                "message=invalid API key: sensitive-value",
            ),
        ):
            market_predictor.predict_next_2_hours(
                "forecast.png",
                {"current_price": 100.0},
                "BTC-USD",
            )

        client.models.generate_content.assert_called_once()
        sleep.assert_not_called()
        print_call.assert_called_once_with(
            "[API] Gemini request failed on attempt 1/3: "
            "type=RuntimeError, message=invalid API key: sensitive-value"
        )

    def test_market_data_download_has_an_explicit_timeout(self):
        with (
            patch.object(
                market_predictor.yf,
                "download",
                return_value=market_predictor.pd.DataFrame(),
            ) as download,
            self.assertRaisesRegex(RuntimeError, "No 15m candle data"),
        ):
            market_predictor.download_clean_df("BTC-USD", "15m", "3d")

        self.assertEqual(
            download.call_args.kwargs["timeout"],
            market_predictor.MARKET_DATA_TIMEOUT_SECONDS,
        )

    def test_gemini_connection_timeout_is_retried_then_reported_safely(self):
        client = Mock()
        client.models.generate_content.side_effect = [
            TimeoutError("read timed out"),
            TimeoutError("connection timeout"),
            TimeoutError("read timed out"),
        ]

        with (
            patch.object(market_predictor.gemini_utils, "create_client", return_value=client),
            patch.object(Image, "open", return_value=object()),
            patch.object(market_predictor.gemini_utils.time, "sleep") as sleep,
            self.assertRaisesRegex(
                RuntimeError,
                "Gemini API request failed: type=TimeoutError, .*"
                "message=read timed out",
            ),
        ):
            market_predictor.predict_next_2_hours(
                "forecast.png",
                {"current_price": 100.0},
                "BTC-USD",
            )
        self.assertEqual(client.models.generate_content.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10])


if __name__ == "__main__":
    unittest.main()
