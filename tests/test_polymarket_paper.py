import csv
import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src import auto_run, evaluator, market_predictor, polymarket_client, polymarket_paper


UTC = datetime.timezone.utc
WINDOW_START = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def market_for(horizon_hours: int = 4) -> dict:
    window_end = WINDOW_START + datetime.timedelta(hours=horizon_hours)
    return {
        "id": f"btc-window-{horizon_hours}h",
        "question": "Bitcoin Up or Down - test window?",
        "description": (
            "This market resolves Up if the Chainlink Bitcoin price at the end "
            "of the period is higher than the Bitcoin price at the beginning. "
            "It resolves Down if the closing price is lower."
        ),
        "resolutionSource": "https://data.chain.link/streams/btc-usd",
        "startDate": WINDOW_START.isoformat(),
        "endDate": window_end.isoformat(),
        "active": True,
        "closed": False,
        "outcomes": json.dumps(["Up", "Down"]),
        "clobTokenIds": json.dumps(["up-token", "down-token"]),
        "feesEnabled": True,
        "feeRate": 0.07,
    }


def advisory_row(direction: str = "UP") -> dict:
    return {
        "forecast_issue_timestamp": WINDOW_START.isoformat(),
        "target_timestamp": (WINDOW_START + datetime.timedelta(hours=4)).isoformat(),
        "forecast_horizon_hours": "4",
        "direction": direction,
    }


def forecast_row(horizon_hours: int = 2, direction: str = "BULLISH") -> dict:
    return {
        "forecast_id": f"forecast-{horizon_hours}h",
        "forecast_issue_timestamp": WINDOW_START.isoformat(),
        "intended_evaluation_timestamp": (
            WINDOW_START + datetime.timedelta(hours=horizon_hours)
        ).isoformat(),
        "forecast_horizon_hours": str(horizon_hours),
        "predicted_direction": direction,
    }


def order_book(asks=None, bids=None) -> dict:
    return {
        "asks": asks if asks is not None else [{"price": "0.5", "size": "100"}],
        "bids": bids if bids is not None else [{"price": "0.49", "size": "100"}],
    }


class MockPolymarketClient:
    def __init__(self, markets=None, books=None, prices=None, resolution=None):
        self.markets = markets or []
        self.books = books or {"up-token": order_book(), "down-token": order_book()}
        self.prices = prices or {"up-token": 0.5, "down-token": 0.5}
        self.resolution = resolution or {}
        self.book_calls = []

    def find_markets(self, window_start, window_end):
        return self.markets

    def get_order_book(self, token_id):
        self.book_calls.append(token_id)
        return self.books[token_id]

    def get_price(self, token_id, side="BUY"):
        if side != "BUY":
            raise AssertionError("Entry must use the CLOB ask.")
        return self.prices[token_id]

    def get_market(self, market_id):
        return self.resolution[market_id]


class PolymarketPaperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.state_path = self.directory / "paper_state.json"
        self.trades_path = self.directory / "paper_trades.csv"
        self.results_path = self.directory / "paper_results.csv"
        self.now = WINDOW_START + datetime.timedelta(seconds=5)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def process(self, row, kind, api, now=None):
        return polymarket_paper.process_forecast(
            row,
            kind,
            client=api,
            state_path=self.state_path,
            trades_path=self.trades_path,
            now=now or self.now,
        )

    def test_exact_four_hour_window_enters_at_ask_and_keeps_separate_bankroll(self):
        api = MockPolymarketClient(markets=[market_for(4)])
        result = self.process(advisory_row(), "advisory", api)
        self.assertEqual(result["status"], "entered")
        trade = result["trade"]
        self.assertEqual(trade["market_id"], "btc-window-4h")
        self.assertEqual(trade["outcome_bought"], "Up")
        self.assertEqual(trade["entry_odds"], 0.5)
        self.assertEqual(trade["stake"], 5.0)
        self.assertEqual(trade["shares"], 10.0)
        self.assertEqual(trade["spread"], 0.01)
        self.assertEqual(trade["fee"], 0.175)
        self.assertEqual(trade["virtual_balance"], 94.83)
        self.assertEqual(api.book_calls, ["up-token"])

        with self.state_path.open(encoding="utf-8") as file:
            state = json.load(file)
        self.assertEqual(state["starting_balance"], 100.0)
        self.assertEqual(len(state["open_positions"]), 1)
        with self.trades_path.open(newline="", encoding="utf-8") as file:
            logged = next(csv.DictReader(file))
        self.assertEqual(logged["forecast_reference"], f"advisory:{WINDOW_START.isoformat()}")

    def test_bearish_forecast_buys_down_token(self):
        api = MockPolymarketClient(markets=[market_for(2)])
        result = self.process(forecast_row(2, "BEARISH"), "forecast", api)
        self.assertEqual(result["status"], "entered")
        self.assertEqual(result["trade"]["outcome_bought"], "Down")
        self.assertEqual(api.book_calls, ["down-token"])

    def test_sideways_forecast_is_logged_as_non_tradable_skip(self):
        api = MockPolymarketClient(markets=[market_for(2)])
        result = self.process(forecast_row(2, "SIDEWAYS"), "forecast", api)

        self.assertEqual(result["status"], "skipped")
        self.assertIn("SIDEWAYS", result["reason"])
        self.assertEqual(api.book_calls, [])
        with self.trades_path.open(newline="", encoding="utf-8") as file:
            logged = next(csv.DictReader(file))
        self.assertEqual(logged["status"], "SKIPPED")
        self.assertIn("SIDEWAYS", logged["skip_reason"])

    def test_two_hour_forecast_never_substitutes_hourly_or_four_hour_market(self):
        quarter_hour = market_for(1)
        quarter_hour["id"] = "btc-window-15m"
        quarter_hour["endDate"] = (
            WINDOW_START + datetime.timedelta(minutes=15)
        ).isoformat()
        hourly = market_for(1)
        four_hour = market_for(4)
        api = MockPolymarketClient(markets=[quarter_hour, hourly, four_hour])
        result = self.process(forecast_row(2), "forecast", api)
        self.assertEqual(result["status"], "skipped")
        self.assertIn("exact 2H measurement window", result["reason"])
        self.assertEqual(api.book_calls, [])
        with self.trades_path.open(newline="", encoding="utf-8") as file:
            logged = next(csv.DictReader(file))
        self.assertEqual(logged["status"], "SKIPPED")
        self.assertTrue(logged["skip_reason"])

    def test_missing_market_is_logged_with_reason(self):
        result = self.process(
            advisory_row(), "advisory", MockPolymarketClient(markets=[])
        )
        self.assertEqual(result["status"], "skipped")
        self.assertIn("No verified BTC Up/Down market", result["reason"])
        with self.trades_path.open(newline="", encoding="utf-8") as file:
            logged = next(csv.DictReader(file))
        self.assertEqual(logged["status"], "SKIPPED")
        self.assertIn("resolution rules", logged["skip_reason"])

    def test_stale_forecast_is_skipped(self):
        result = self.process(
            advisory_row(),
            "advisory",
            MockPolymarketClient(markets=[market_for(4)]),
            now=WINDOW_START + datetime.timedelta(minutes=2),
        )
        self.assertEqual(result["status"], "skipped")
        self.assertIn("stale", result["reason"])

    def test_invalid_price_quote_is_skipped(self):
        api = MockPolymarketClient(markets=[market_for(4)], prices={"up-token": 0.6})
        result = self.process(advisory_row(), "advisory", api)
        self.assertEqual(result["status"], "skipped")
        self.assertIn("does not agree", result["reason"])

    def test_insufficient_ask_liquidity_is_skipped(self):
        api = MockPolymarketClient(
            markets=[market_for(4)],
            books={
                "up-token": order_book(
                    asks=[{"price": "0.5", "size": "2"}]
                )
            },
        )
        result = self.process(advisory_row(), "advisory", api)
        self.assertEqual(result["status"], "skipped")
        self.assertIn("Insufficient ask liquidity", result["reason"])

    def test_buy_walks_asks_and_calculates_vwap_fee_and_spread(self):
        quote = polymarket_paper._simulate_buy(
            order_book(
                asks=[
                    {"price": "0.6", "size": "10"},
                    {"price": "0.5", "size": "5"},
                ],
                bids=[{"price": "0.49", "size": "20"}],
            ),
            5,
            0.07,
        )
        self.assertAlmostEqual(quote["shares"], 5 + 2.5 / 0.6)
        self.assertAlmostEqual(quote["entry_odds"], 5 / quote["shares"])
        self.assertAlmostEqual(quote["fee"], 0.1575)
        self.assertAlmostEqual(quote["spread"], 0.01)

    def test_entry_is_skipped_when_stake_plus_fee_exceeds_virtual_balance(self):
        polymarket_paper._save_state(
            self.state_path,
            {
                "starting_balance": 5.0,
                "current_balance": 5.0,
                "open_positions": [],
                "entered_market_ids": [],
                "processed_forecasts": [],
            },
        )
        result = self.process(
            advisory_row(), "advisory", MockPolymarketClient(markets=[market_for(4)])
        )
        self.assertEqual(result["status"], "skipped")
        self.assertIn("Insufficient virtual balance", result["reason"])

    def test_malformed_order_book_levels_are_rejected(self):
        with self.assertRaisesRegex(
            polymarket_paper.PaperTradeError, "invalid asks price or size"
        ):
            polymarket_paper._simulate_buy(
                order_book(
                    asks=[{"price": "1.1", "size": "10"}],
                ),
                5,
                0.07,
            )

    def test_duplicate_market_entry_is_prevented(self):
        api = MockPolymarketClient(markets=[market_for(4)])
        first = self.process(advisory_row(), "advisory", api)
        second_row = advisory_row("DOWN")
        second_row["forecast_issue_timestamp"] = (
            WINDOW_START + datetime.timedelta(seconds=1)
        ).isoformat()
        second_row["target_timestamp"] = (
            WINDOW_START + datetime.timedelta(hours=4, seconds=1)
        ).isoformat()
        second = self.process(second_row, "advisory", api)
        self.assertEqual(first["status"], "entered")
        self.assertEqual(second["status"], "skipped")
        self.assertIn("already entered", second["reason"])
        self.assertEqual(api.book_calls, ["up-token"])

    def test_position_waits_for_confirmed_final_resolution_and_records_pnl(self):
        market = market_for(4)
        not_resolved = dict(
            market,
            closed=False,
            umaResolutionStatus="proposed",
            outcomePrices=json.dumps(["0.98", "0.02"]),
        )
        resolved = dict(
            market,
            closed=True,
            umaResolutionStatus="resolved",
            outcomePrices=json.dumps(["1", "0"]),
        )
        api = MockPolymarketClient(
            markets=[market],
            resolution={market["id"]: not_resolved},
        )
        entered = self.process(advisory_row(), "advisory", api)
        self.assertEqual(entered["status"], "entered")

        after_expiry = WINDOW_START + datetime.timedelta(hours=5)
        pending = polymarket_paper.settle_open_positions(
            client=api,
            state_path=self.state_path,
            results_path=self.results_path,
            now=after_expiry,
        )
        self.assertEqual(pending, {"settled": 0, "pending": 1})
        self.assertFalse(self.results_path.exists())

        api.resolution[market["id"]] = resolved
        settled = polymarket_paper.settle_open_positions(
            client=api,
            state_path=self.state_path,
            results_path=self.results_path,
            now=after_expiry,
        )
        self.assertEqual(settled, {"settled": 1, "pending": 0})
        with self.results_path.open(newline="", encoding="utf-8") as file:
            result = next(csv.DictReader(file))
        self.assertEqual(result["resolved_outcome"], "Up")
        self.assertAlmostEqual(float(result["payout"]), 10.0)
        self.assertAlmostEqual(float(result["pnl"]), 4.825)
        self.assertAlmostEqual(float(result["virtual_balance"]), 104.83)
        with self.state_path.open(encoding="utf-8") as file:
            state = json.load(file)
        self.assertEqual(state["open_positions"], [])
        self.assertEqual(state["current_balance"], 104.83)

    def test_unresolved_closed_market_does_not_settle(self):
        market = market_for(4)
        unresolved = dict(
            market,
            closed=True,
            umaResolutionStatus="proposed",
            outcomePrices=json.dumps(["1", "0"]),
        )
        api = MockPolymarketClient(
            markets=[market],
            resolution={market["id"]: unresolved},
        )
        self.process(advisory_row(), "advisory", api)
        result = polymarket_paper.settle_open_positions(
            client=api,
            state_path=self.state_path,
            results_path=self.results_path,
            now=WINDOW_START + datetime.timedelta(hours=5),
        )
        self.assertEqual(result, {"settled": 0, "pending": 1})
        self.assertFalse(self.results_path.exists())

    def test_client_uses_only_public_get_endpoints(self):
        session = Mock()
        response = Mock()
        response.json.side_effect = [
            [{"id": "market"}],
            order_book(),
            {"price": "0.5"},
            {"id": "market"},
        ]
        response.raise_for_status.return_value = None
        session.get.return_value = response
        client = polymarket_client.PolymarketClient(session=session)

        self.assertEqual(
            client.find_markets(WINDOW_START, WINDOW_START + datetime.timedelta(hours=4)),
            [{"id": "market"}],
        )
        client.get_order_book("up-token")
        self.assertEqual(client.get_price("up-token", "BUY"), 0.5)
        self.assertEqual(client.get_market("market"), {"id": "market"})
        urls = [call.args[0] for call in session.get.call_args_list]
        self.assertIn("https://gamma-api.polymarket.com/markets", urls[0])
        self.assertIn("https://clob.polymarket.com/book", urls[1])
        self.assertIn("https://clob.polymarket.com/price", urls[2])
        self.assertIn("https://gamma-api.polymarket.com/markets/market", urls[3])

    def test_scheduler_sends_new_two_hour_forecast_to_paper_engine_and_evaluator(self):
        forecast = forecast_row(2)
        with (
            patch.object(market_predictor, "run_predictor", return_value=forecast),
            patch.object(
                polymarket_paper,
                "process_forecast",
                return_value={"status": "entered"},
            ) as process_forecast,
            patch.object(
                evaluator,
                "grade_pending_predictions",
                return_value={
                    "graded": 0,
                    "pending": 0,
                    "needs_review": 0,
                    "errors": 0,
                },
            ) as grade_forecasts,
            patch.object(
                evaluator,
                "grade_pending_advisories",
                return_value={
                    "graded": 0,
                    "pending": 0,
                    "needs_review": 0,
                    "errors": 0,
                },
            ),
        ):
            auto_run.run_market_prediction()

        process_forecast.assert_called_once_with(forecast, "forecast")
        grade_forecasts.assert_called_once_with()

    def test_scheduler_sends_new_four_hour_advisory_to_paper_engine(self):
        source_time = WINDOW_START - datetime.timedelta(minutes=15)
        spot_data = Mock()
        spot_data.index = [pd.Timestamp(source_time)]
        advisory = {
            "direction": "UP",
            "target_range_low": 101.0,
            "target_range_high": 110.0,
            "target_price": 105.0,
            "invalidation_price": 95.0,
            "confidence_score": 7,
            "market_regime": "trend",
            "reasoning": "test advisory",
        }
        with (
            patch.object(auto_run.data_engine, "fetch_spot_data", return_value=spot_data),
            patch.object(
                auto_run.data_engine,
                "compile_market_context",
                return_value={"spot": {"current_close": 100.0}},
            ),
            patch.object(auto_run.chart_engine, "generate_pro_chart"),
            patch.object(
                auto_run.pro_strategist,
                "generate_trading_advisory",
                return_value=advisory,
            ),
            patch.object(auto_run, "_validate_advisory", return_value=True),
            patch.object(auto_run, "initialize_csv"),
            patch.object(auto_run, "utc_now", return_value=WINDOW_START),
            patch.object(auto_run, "append_record"),
            patch.object(auto_run, "append_forecast_result"),
            patch.object(
                auto_run.pro_strategist,
                "calculate_risk_reward",
                return_value={"actionable": True},
            ),
            patch.object(
                polymarket_paper,
                "process_forecast",
                return_value={"status": "entered"},
            ) as process_forecast,
        ):
            auto_run.run_agent_pipeline()

        args, _ = process_forecast.call_args
        self.assertEqual(args[0]["direction"], "UP")
        self.assertEqual(args[0]["forecast_horizon_hours"], 4)
        self.assertEqual(args[1], "advisory")


if __name__ == "__main__":
    unittest.main()
