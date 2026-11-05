import json
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src import market_predictor


class MarketPredictorHttpTests(unittest.TestCase):
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
