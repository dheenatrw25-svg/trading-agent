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
            "current_price": 100.0,
            "technical_indicators": {
                "rsi_15m": 72.0,
                "dxy_macro_trend": "+0.0%",
            },
            "institutional_order_flow": {
                "largest_buy_wall_price": 99.0,
                "largest_buy_wall_usd": "$45,000,000",
                "largest_sell_wall_price": 101.0,
                "largest_sell_wall_usd": "$52,000,000",
                "cvd_status": "Neutral/Choppy",
                "open_interest_trend": "Unavailable",
            },
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
        payload = json.loads(contents[2])
        self.assertEqual(
            list(payload),
            ["current_price", "technical_indicators", "institutional_order_flow"],
        )
        self.assertEqual(payload["current_price"], 100.0)
        supporting_context = json.loads(contents[3].split("\n", 1)[1])
        self.assertEqual(
            supporting_context["price_action"]["latest_completed_candle"]["close"],
            100.0,
        )
        self.assertIn("LESSON TEXT", contents[4])
        self.assertIsInstance(contents[5], Image.Image)
        self.assertEqual(request["model"], "gemini-3.8-flash")

    def test_system_prompt_explains_wall_and_taker_divergence_rules(self) -> None:
        self.assertIn("Do not set a target beyond a nearby opposing wall", pro_strategist.PRO_TRADER_PROMPT)
        self.assertIn("never invent a fixed offset", pro_strategist.PRO_TRADER_PROMPT)
        self.assertIn("do not claim that anyone is lying", pro_strategist.PRO_TRADER_PROMPT)
        self.assertIn("confidence 4/10 or lower", pro_strategist.PRO_TRADER_PROMPT)


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
    def test_taker_pressure_aggregates_recent_buy_and_sell_volume(self) -> None:
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [
            {
                "timestamp": now_ms - (11 - index) * 15 * 60_000,
                "buyVol": "2",
                "sellVol": "1",
            }
            for index in range(12)
        ]
        with patch.object(data_engine.requests, "get", return_value=response) as get:
            context = data_engine._fetch_taker_pressure("BTCUSDT")

        get.assert_called_once_with(
            "https://fapi.binance.com/futures/data/takerlongshortRatio",
            params={"symbol": "BTCUSDT", "period": "15m", "limit": 12},
            timeout=data_engine.REQUEST_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            context["taker_pressure_status"],
            "Aggressive Bid Lifting (Bullish CVD)",
        )
        self.assertEqual(context["taker_buy_volume_btc"], 24)
        self.assertEqual(context["taker_sell_volume_btc"], 12)
        self.assertEqual(context["net_taker_volume_btc"], 12)
        self.assertEqual(context["taker_buy_sell_ratio"], 2)
        self.assertEqual(context["taker_pressure_window_hours"], 3)

    def test_taker_pressure_reports_selling_and_neutral_conditions(self) -> None:
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        cases = [
            (1, 2, "Aggressive Ask Hitting (Bearish CVD)"),
            (1, 1, "Neutral/Choppy"),
        ]
        for buy_volume, sell_volume, expected_status in cases:
            with self.subTest(expected_status=expected_status):
                response = Mock()
                response.raise_for_status.return_value = None
                response.json.return_value = [
                    {
                        "timestamp": now_ms - (11 - index) * 15 * 60_000,
                        "buyVol": str(buy_volume),
                        "sellVol": str(sell_volume),
                    }
                    for index in range(12)
                ]
                with patch.object(
                    data_engine.requests, "get", return_value=response
                ):
                    context = data_engine._fetch_taker_pressure("BTCUSDT")
                self.assertEqual(
                    context["taker_pressure_status"], expected_status
                )

    def test_taker_pressure_marks_short_or_stale_history_unavailable(self) -> None:
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        cases = [
            [
                {"timestamp": now_ms, "buyVol": "1", "sellVol": "1"}
                for _ in range(7)
            ],
            [
                {
                    "timestamp": now_ms - 60 * 60_000 - (7 - index) * 15 * 60_000,
                    "buyVol": "1",
                    "sellVol": "1",
                }
                for index in range(8)
            ],
        ]
        for history in cases:
            with self.subTest(history_length=len(history)):
                response = Mock()
                response.raise_for_status.return_value = None
                response.json.return_value = history
                with patch.object(
                    data_engine.requests, "get", return_value=response
                ):
                    context = data_engine._fetch_taker_pressure("BTCUSDT")
                self.assertEqual(context["taker_pressure_status"], "Unavailable")
                self.assertIsNone(context["net_taker_volume_btc"])

    def test_compiled_market_context_includes_order_book_walls(self) -> None:
        order_book = {
            "symbol": "BTCUSDT",
            "depth_limit": 1000,
            "largest_buy_wall": {
                "price_usd": 64900.0,
                "quantity_btc": 2.0,
                "notional_usd": 129800.0,
            },
            "largest_sell_wall": {
                "price_usd": 65200.0,
                "quantity_btc": 1.2,
                "notional_usd": 78240.0,
            },
        }
        spot_data = pd.DataFrame({"Close": [100.0]})
        confirmed_spot_data = pd.DataFrame({"Close": [100.0]})
        with (
            patch.object(data_engine, "_build_price_action_context", return_value={}),
            patch.object(
                data_engine, "_confirmed_candles", return_value=confirmed_spot_data
            ),
            patch.object(
                data_engine,
                "fetch_macro_context",
                return_value={"dxy_daily_change_pct": 0.5},
            ),
            patch.object(
                data_engine,
                "fetch_derivatives_data",
                return_value={
                    "taker_pressure_status": "Neutral/Choppy",
                },
            ),
            patch.object(data_engine, "fetch_order_book_walls", return_value=order_book),
            patch.object(
                data_engine,
                "_fetch_15m_indicators",
                return_value={"current_price": 100.5, "rsi_15m": 72.0},
            ),
        ):
            context = data_engine.compile_market_context(spot_data)

        self.assertEqual(context["order_book"], order_book)
        self.assertEqual(
            context["derivatives"]["taker_pressure_status"], "Neutral/Choppy"
        )
        self.assertEqual(context["current_price"], 100.5)
        self.assertEqual(
            context["technical_indicators"],
            {"rsi_15m": 72.0, "dxy_macro_trend": "+0.5%"},
        )
        self.assertEqual(
            context["institutional_order_flow"],
            {
                "largest_buy_wall_price": 64900.0,
                "largest_buy_wall_usd": "$129,800",
                "largest_sell_wall_price": 65200.0,
                "largest_sell_wall_usd": "$78,240",
                "cvd_status": "Neutral/Choppy",
                "open_interest_trend": "Unavailable",
            },
        )
        self.assertEqual(
            data_engine.build_ai_prompt_payload(context),
            {
                "current_price": 100.5,
                "technical_indicators": {
                    "rsi_15m": 72.0,
                    "dxy_macro_trend": "+0.5%",
                },
                "institutional_order_flow": context["institutional_order_flow"],
            },
        )

    def test_institutional_order_flow_maps_taker_and_open_interest_statuses(self) -> None:
        result = data_engine._build_institutional_order_flow(
            {
                "largest_buy_wall": {"price_usd": 81000.0, "notional_usd": 45_000_000},
                "largest_sell_wall": {"price_usd": 81500.0, "notional_usd": 52_000_000},
            },
            {
                "taker_pressure_status": "Aggressive Ask Hitting (Bearish CVD)",
                "open_interest_trend": "falling",
            },
        )
        self.assertEqual(result["largest_buy_wall_price"], 81000.0)
        self.assertEqual(result["largest_buy_wall_usd"], "$45,000,000")
        self.assertEqual(result["largest_sell_wall_price"], 81500.0)
        self.assertEqual(result["largest_sell_wall_usd"], "$52,000,000")
        self.assertEqual(
            result["cvd_status"], "Aggressive Market Selling (Bearish CVD)"
        )
        self.assertEqual(
            result["open_interest_trend"],
            "Falling OI (Position Closing/Reversal Risk)",
        )

    def test_15_minute_indicators_use_only_completed_candles(self) -> None:
        now = pd.Timestamp.now(tz="UTC").floor("15min")
        timestamps = pd.date_range(
            end=now - pd.Timedelta(minutes=15),
            periods=40,
            freq="15min",
        )
        close = [100.0 + index for index in range(40)]
        candles = pd.DataFrame(
            {
                "Open": close,
                "High": [price + 1 for price in close],
                "Low": [price - 1 for price in close],
                "Close": close,
                "Volume": [1.0] * len(close),
            },
            index=timestamps,
        )
        with patch.object(data_engine.yf, "download", return_value=candles) as download:
            indicators = data_engine._fetch_15m_indicators()

        download.assert_called_once_with(
            "BTC-USD",
            period="5d",
            interval="15m",
            progress=False,
        )
        self.assertEqual(indicators["current_price"], close[-1])
        self.assertEqual(indicators["rsi_15m"], 100.0)

    def test_order_book_walls_use_depth_1000_and_report_largest_levels(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "bids": [["65000", "1.5"], ["64900", "2.0"], ["64800", "0.5"]],
            "asks": [["65100", "0.8"], ["65200", "1.2"], ["65300", "0.4"]],
        }
        with patch.object(data_engine.requests, "get", return_value=response) as get:
            context = data_engine.fetch_order_book_walls("BTCUSDT")

        get.assert_called_once_with(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": "BTCUSDT", "limit": 1000},
            timeout=data_engine.REQUEST_TIMEOUT_SECONDS,
        )
        self.assertEqual(context["depth_limit"], 1000)
        self.assertEqual(
            context["largest_buy_wall"],
            {
                "price_usd": 64900.0,
                "quantity_btc": 2.0,
                "notional_usd": 129800.0,
            },
        )
        self.assertEqual(
            context["largest_sell_wall"],
            {
                "price_usd": 65200.0,
                "quantity_btc": 1.2,
                "notional_usd": 78240.0,
            },
        )
        self.assertTrue(context["snapshot_timestamp_utc"])
        self.assertIn("cannot", context["interpretation_note"])

    def test_order_book_walls_reject_malformed_levels(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "bids": [["not-a-price", "1"]],
            "asks": [["65100", "1"]],
        }
        with patch.object(data_engine.requests, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "non-numeric bid level"):
                data_engine.fetch_order_book_walls("BTCUSDT")

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
