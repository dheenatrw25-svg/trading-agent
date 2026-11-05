import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from src import gemini_utils, trading_agent


class TradingAgentGeminiTests(unittest.TestCase):
    def test_trade_decision_uses_shared_client_and_request_without_temperature(self):
        decision = {
            "action": "WAIT",
            "setup_grade": "C",
            "confidence_score": 2,
            "entry_price": 100.0,
            "stop_loss": 90.0,
            "take_profit": 110.0,
            "thesis": "No trade setup.",
        }
        client = Mock()
        client.models.generate_content.return_value = Mock(
            text=json.dumps(decision)
        )

        with tempfile.TemporaryDirectory() as directory:
            chart_path = Path(directory) / "chart.png"
            Image.new("RGB", (4, 4)).save(chart_path)
            with (
                patch.object(trading_agent, "get_genai_client", return_value=client),
                patch.object(gemini_utils, "generate_content_with_retry", wraps=gemini_utils.generate_content_with_retry) as request,
            ):
                result = trading_agent.ask_gemini(
                    chart_path, {"last_15m_close": 100.0}, "BTC-USD"
                )

        self.assertEqual(result, decision)
        self.assertEqual(request.call_args.kwargs["model"], "gemini-3.0-flash")
        self.assertNotIn(
            "temperature",
            request.call_args.kwargs["config"].model_dump(exclude_none=True),
        )

    def test_trade_decision_rejects_empty_and_invalid_json(self):
        for response, error in (
            (Mock(text=""), "response was empty"),
            (Mock(text="{invalid"), "not valid JSON"),
        ):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                client = Mock()
                client.models.generate_content.return_value = response
                chart_path = Path(directory) / "chart.png"
                Image.new("RGB", (4, 4)).save(chart_path)
                with (
                    patch.object(trading_agent, "get_genai_client", return_value=client),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    trading_agent.ask_gemini(
                        chart_path, {"last_15m_close": 100.0}, "BTC-USD"
                    )


if __name__ == "__main__":
    unittest.main()
