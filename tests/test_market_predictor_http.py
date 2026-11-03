import json
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src import market_predictor


class MarketPredictorHttpTests(unittest.TestCase):
    def test_gemini_request_uses_120_second_timeout_and_sdk_retries_disabled(self):
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
            patch.object(market_predictor.genai, "Client", return_value=client) as factory,
            patch.object(Image, "open", return_value=object()),
        ):
            result = market_predictor.predict_next_2_hours(
                "forecast.png",
                {"current_price": 100.0},
                "BTC-USD",
            )

        http_options = factory.call_args.kwargs["http_options"]
        self.assertEqual(
            http_options.timeout,
            120_000,
        )
        self.assertLessEqual(http_options.timeout, 120_000)
        self.assertEqual(http_options.retry_options.attempts, 1)
        self.assertEqual(result["predicted_direction"], "BULLISH")

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
            patch.object(market_predictor.genai, "Client", return_value=client),
            patch.object(Image, "open", return_value=object()),
            patch.object(market_predictor.time, "sleep") as sleep,
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
                "[API] Gemini 2H attempt 1/3 failed (RuntimeError); retrying in 5 seconds."
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
            patch.object(market_predictor.genai, "Client", return_value=client),
            patch.object(Image, "open", return_value=object()),
            patch.object(market_predictor.time, "sleep") as sleep,
            self.assertRaisesRegex(
                RuntimeError,
                "Gemini 2-hour forecast request failed after 3 attempt\\(s\\) \\(RuntimeError\\)",
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

    def test_http_statuses_and_connection_timeouts_are_retryable(self):
        for status_code in (429, 502, 503, 504):
            error = RuntimeError("transient API error")
            error.code = status_code
            with self.subTest(status_code=status_code):
                self.assertTrue(market_predictor._is_retryable_gemini_error(error))

        self.assertTrue(
            market_predictor._is_retryable_gemini_error(
                TimeoutError("connection timed out")
            )
        )

    def test_permanent_gemini_error_is_not_retried_or_logged_with_error_text(self):
        client = Mock()
        client.models.generate_content.side_effect = RuntimeError(
            "invalid API key: sensitive-value"
        )

        with (
            patch.object(market_predictor.genai, "Client", return_value=client),
            patch.object(Image, "open", return_value=object()),
            patch.object(market_predictor.time, "sleep") as sleep,
            patch("builtins.print") as print_call,
            self.assertRaisesRegex(
                RuntimeError,
                "Gemini 2-hour forecast request failed after 1 attempt",
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
            "[API] Gemini 2H attempt 1/3 failed (RuntimeError); no retry will be made."
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
            patch.object(market_predictor.genai, "Client", return_value=client),
            patch.object(Image, "open", return_value=object()),
            patch.object(market_predictor.time, "sleep") as sleep,
            self.assertRaisesRegex(
                RuntimeError,
                "Gemini 2-hour forecast request failed after 3 attempt",
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
