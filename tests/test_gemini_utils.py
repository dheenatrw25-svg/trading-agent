import os
import unittest
from unittest.mock import Mock, patch

from src import gemini_utils


class GeminiAPIError(Exception):
    def __init__(self, code: int, status: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message


class GeminiUtilsTests(unittest.TestCase):
    def test_client_loads_gemini_key_and_disables_sdk_retries(self) -> None:
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True),
            patch.object(gemini_utils, "load_dotenv") as load_dotenv,
            patch.object(gemini_utils.genai, "Client") as client_factory,
        ):
            gemini_utils.create_client()

        load_dotenv.assert_called_once_with(
            dotenv_path=gemini_utils.PROJECT_ROOT / ".env"
        )
        client_factory.assert_called_once()
        args = client_factory.call_args.kwargs
        self.assertEqual(args["api_key"], "test-key")
        self.assertEqual(args["http_options"].timeout, 120_000)
        self.assertEqual(args["http_options"].retry_options.attempts, 1)

    def test_google_api_key_is_supported_as_fallback(self) -> None:
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "fallback-key"}, clear=True),
            patch.object(gemini_utils, "load_dotenv"),
            patch.object(gemini_utils.genai, "Client") as client_factory,
        ):
            gemini_utils.create_client()

        self.assertEqual(client_factory.call_args.kwargs["api_key"], "fallback-key")

    def test_successful_response_is_returned(self) -> None:
        response = object()
        client = Mock()
        client.models.generate_content.return_value = response

        self.assertIs(
            gemini_utils.generate_content_with_retry(client, model="test-model"),
            response,
        )
        client.models.generate_content.assert_called_once_with(model="test-model")

    def test_empty_and_blocked_responses_are_rejected(self) -> None:
        for response, expected in (
            (None, "returned no response"),
            (Mock(text="  "), "response was empty"),
            (
                Mock(
                    text=None,
                    prompt_feedback=Mock(block_reason="SAFETY"),
                    candidates=[],
                ),
                "block_reason=SAFETY",
            ),
            (
                Mock(
                    text=None,
                    prompt_feedback=None,
                    candidates=[Mock(finish_reason="SAFETY")],
                ),
                "finish_reason=SAFETY",
            ),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(
                ValueError, expected
            ):
                gemini_utils.response_text(response, "test forecast")

    def test_permanent_client_errors_preserve_code_status_and_message(self) -> None:
        for code, status in (
            (400, "INVALID_ARGUMENT"),
            (401, "UNAUTHENTICATED"),
            (403, "PERMISSION_DENIED"),
        ):
            error = GeminiAPIError(code, status, "request was rejected")
            client = Mock()
            client.models.generate_content.side_effect = error

            with (
                self.subTest(code=code),
                patch.object(gemini_utils.time, "sleep") as sleep,
                patch("builtins.print") as print_call,
                self.assertRaisesRegex(
                    RuntimeError,
                    rf"code={code}, status={status}, message=request was rejected",
                ),
            ):
                gemini_utils.generate_content_with_retry(client, model="test-model")

            client.models.generate_content.assert_called_once()
            sleep.assert_not_called()
            self.assertIn(f"code={code}", print_call.call_args.args[0])
            self.assertIn(f"status={status}", print_call.call_args.args[0])
            self.assertIn("request was rejected", print_call.call_args.args[0])

    def test_daily_quota_429_is_not_retried(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = GeminiAPIError(
            429,
            "RESOURCE_EXHAUSTED",
            "Daily quota exceeded for this project",
        )

        with (
            patch.object(gemini_utils.time, "sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "Daily quota exceeded"),
        ):
            gemini_utils.generate_content_with_retry(client, model="test-model")

        client.models.generate_content.assert_called_once()
        sleep.assert_not_called()

    def test_503_retries_with_backoff_and_preserves_diagnostics(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = [
            GeminiAPIError(503, "UNAVAILABLE", "server is busy"),
            Mock(text='{"ok": true}'),
        ]

        with (
            patch.object(gemini_utils.time, "sleep") as sleep,
            patch("builtins.print") as print_call,
        ):
            response = gemini_utils.generate_content_with_retry(
                client, model="test-model"
            )

        self.assertEqual(response.text, '{"ok": true}')
        self.assertEqual(client.models.generate_content.call_count, 2)
        sleep.assert_called_once_with(5)
        log_message = print_call.call_args.args[0]
        self.assertIn("code=503", log_message)
        self.assertIn("status=UNAVAILABLE", log_message)
        self.assertIn("message=server is busy", log_message)

    def test_api_keys_are_redacted_from_error_logs(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = RuntimeError(
            "request failed with test-secret-key"
        )
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret-key"}),
            patch.object(gemini_utils.time, "sleep"),
            patch("builtins.print") as print_call,
            self.assertRaises(RuntimeError),
        ):
            gemini_utils.generate_content_with_retry(client, model="test-model")

        self.assertNotIn("test-secret-key", print_call.call_args.args[0])
        self.assertIn("[REDACTED]", print_call.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
