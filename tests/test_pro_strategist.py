import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from PIL import Image

from src import data_engine, pro_strategist


def valid_advisory(direction: str = "UP") -> dict:
    return {
        "direction": direction,
        "target_range_low": 110.0,
        "target_range_high": 120.0,
        "invalidation_price": 90.0,
        "confidence_score": 8,
        "market_regime": "Breakout attempt",
        "reasoning": "Completed candle structure supports a directional bias.",
    }


class PlaybookTests(unittest.TestCase):
    def test_playbook_loads_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trading_playbook.md"
            path.write_text("# Test playbook", encoding="utf-8")
            with patch.object(pro_strategist, "PROJECT_ROOT", Path(temporary_directory)):
                self.assertEqual(pro_strategist.load_playbook(), "# Test playbook")

    def test_missing_and_empty_playbooks_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(pro_strategist, "PROJECT_ROOT", root):
                with self.assertRaisesRegex(FileNotFoundError, "was not found"):
                    pro_strategist.load_playbook()
                (root / "trading_playbook.md").write_text(" \n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "is empty"):
                    pro_strategist.load_playbook()

    def test_gemini_request_contains_playbook_context_lessons_and_chart(self) -> None:
        advisory_response = Mock(text=json.dumps(valid_advisory()))
        client = Mock()
        client.models.generate_content.return_value = advisory_response
        market_context = {
            "spot": {"current_close": 100.0},
            "macro": {"dxy_daily_change_pct": 0.0},
            "derivatives": {"annualized_funding_rate_pct": 0.0},
            "price_action": {"latest_completed_candle": {"close": 100.0}},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "chart.png"
            Image.new("RGB", (4, 4)).save(image_path)
            with (
                patch.object(pro_strategist, "load_playbook", return_value="PLAYBOOK TEXT"),
                patch.object(
                    pro_strategist, "retrieve_past_lessons", return_value="LESSON TEXT"
                ),
                patch.object(pro_strategist.genai, "Client", return_value=client),
            ):
                result = pro_strategist.generate_trading_advisory(
                    market_context, str(image_path)
                )

        self.assertEqual(result["direction"], "UP")
        request = client.models.generate_content.call_args.kwargs
        contents = request["contents"]
        self.assertIn("PLAYBOOK TEXT", contents[1])
        self.assertIn('"current_close": 100.0', contents[2])
        self.assertIn("LESSON TEXT", contents[3])
        self.assertIsInstance(contents[4], Image.Image)
        self.assertEqual(request["model"], "gemini-3.8-flash")


class DirectionAndRiskTests(unittest.TestCase):
    def test_advisory_accepts_only_up_and_down_directions(self) -> None:
        for direction in ("UP", "DOWN"):
            self.assertEqual(
                pro_strategist.validate_advisory(valid_advisory(direction))["direction"],
                direction,
            )
        for direction in ("NEUTRAL", "WAIT", "BUY", "up"):
            with self.subTest(direction=direction), self.assertRaisesRegex(
                ValueError, "unsupported direction"
            ):
                pro_strategist.validate_advisory(valid_advisory(direction))

    def test_risk_reward_requires_positive_sided_levels_and_minimum_ratio(self) -> None:
        valid = pro_strategist.calculate_risk_reward(
            100, "UP", 110, 120, 90
        )
        self.assertTrue(valid["actionable"])
        self.assertEqual(valid["risk"], 10)
        self.assertEqual(valid["reward"], 15)
        self.assertEqual(valid["risk_reward_ratio"], 1.5)

        valid_down = pro_strategist.calculate_risk_reward(
            100, "DOWN", 80, 90, 110
        )
        self.assertTrue(valid_down["actionable"])
        self.assertEqual(valid_down["risk_reward_ratio"], 1.5)

        below_threshold = pro_strategist.calculate_risk_reward(
            100, "UP", 105, 115, 90
        )
        self.assertFalse(below_threshold["actionable"])
        self.assertAlmostEqual(below_threshold["risk_reward_ratio"], 1.0)

        invalid_levels = pro_strategist.calculate_risk_reward(
            100, "UP", 90, 110, 105
        )
        self.assertFalse(invalid_levels["actionable"])
        self.assertIsNone(invalid_levels["risk_reward_ratio"])

    def test_strong_dxy_caps_up_confidence(self) -> None:
        context = {
            "spot": {"current_close": 100.0},
            "macro": {"dxy_daily_change_pct": 0.40},
            "derivatives": {"annualized_funding_rate_pct": 0.0},
        }
        result = pro_strategist._apply_python_strategy_checks(
            valid_advisory(), context
        )
        self.assertLessEqual(result["confidence_score"], 5)
        self.assertIn("Strong-dollar headwind", result["reasoning"])

    def test_crowded_funding_reduces_up_confidence_by_two(self) -> None:
        context = {
            "spot": {"current_close": 100.0},
            "macro": {"dxy_daily_change_pct": 0.0},
            "derivatives": {
                "annualized_funding_rate_pct": 25.1,
                "open_interest_trend": "rising",
            },
            "price_action": {
                "volume_comparison": {"volume_state": "flat"},
            },
        }
        result = pro_strategist._apply_python_strategy_checks(
            valid_advisory(), context
        )
        self.assertEqual(result["confidence_score"], 6)
        self.assertIn("reduced by at least two points", result["reasoning"])

    def test_unclear_non_actionable_setup_keeps_direction_and_low_confidence(self) -> None:
        advisory = valid_advisory("DOWN")
        advisory["target_range_low"] = 105.0
        advisory["target_range_high"] = 110.0
        advisory["invalidation_price"] = 95.0
        context = {
            "spot": {"current_close": 100.0},
            "macro": {},
            "derivatives": {},
        }
        result = pro_strategist._apply_python_strategy_checks(advisory, context)
        self.assertEqual(result["direction"], "DOWN")
        self.assertLessEqual(result["confidence_score"], 4)
        self.assertIn("not an actionable setup", result["reasoning"])


class MarketContextTests(unittest.TestCase):
    def test_unavailable_historical_open_interest_is_not_called_a_trend(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [{"timestamp": 1, "sumOpenInterest": "10"}]
        with patch.object(data_engine.requests, "get", return_value=response):
            context = data_engine._fetch_open_interest_trend("BTCUSDT")
        self.assertEqual(context["open_interest_trend"], "unavailable")
        self.assertIsNone(context["open_interest_change_pct"])
        self.assertIn("fewer than two", context["open_interest_trend_reason"])

    def test_reliable_historical_open_interest_change_is_measured(self) -> None:
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [
            {"timestamp": now_ms - 8 * 3_600_000, "sumOpenInterest": "100"},
            {"timestamp": now_ms - 4 * 3_600_000, "sumOpenInterest": "102"},
        ]
        with patch.object(data_engine.requests, "get", return_value=response):
            context = data_engine._fetch_open_interest_trend("BTCUSDT")
        self.assertEqual(context["open_interest_trend"], "rising")
        self.assertAlmostEqual(context["open_interest_change_pct"], 2.0)
        self.assertAlmostEqual(context["open_interest_change_interval_hours"], 4.0)

    def test_price_action_context_uses_completed_candles_and_actual_extremes(self) -> None:
        timestamps = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").floor("4h") - pd.Timedelta(hours=4),
            periods=25,
            freq="4h",
        )
        closes = [100.0 + index for index in range(25)]
        candles = pd.DataFrame(
            {
                "Open": closes,
                "High": [value + 2 for value in closes],
                "Low": [value - 2 for value in closes],
                "Close": closes,
                "Volume": [100.0] * 24 + [200.0],
            },
            index=timestamps,
        )
        context = data_engine._build_price_action_context(candles)
        self.assertEqual(context["latest_completed_candle"]["close"], closes[-1])
        self.assertEqual(len(context["recent_completed_candles"]), 20)
        self.assertEqual(
            context["recent_completed_candles"][-1]["close"], closes[-1]
        )
        self.assertEqual(context["volume_comparison"]["volume_state"], "expanding")
        recent_candles = candles.tail(20)
        self.assertEqual(
            context["recent_swing_high"]["price"], max(recent_candles["High"])
        )
        self.assertEqual(
            context["recent_swing_low"]["price"], min(recent_candles["Low"])
        )
        self.assertTrue(context["latest_completed_candle"]["close_timestamp"])


if __name__ == "__main__":
    unittest.main()
