import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pj22_btc.mexc_downloader import SettingsError  # noqa: E402
from pj22_btc.sentiment_research import (  # noqa: E402
    VALID_TARGET_COLUMNS,
    build_backtest,
    build_backtest_report_html,
    build_follow_trades,
    build_rules_recommendation,
    group_by_sentiment,
    load_sentiment_research_config,
    recommend_action,
)


class SentimentResearchTests(unittest.TestCase):
    def test_build_follow_trades_uses_selected_target_column(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "source_date": "2025-09-02",
                    "sentiment": 4,
                    "next_body": -5.0,
                    "next_open_to_open": 7.0,
                },
                {
                    "source_date": "2025-09-03",
                    "sentiment": -3,
                    "next_body": 9.0,
                    "next_open_to_open": -2.0,
                },
            ]
        )

        trades = build_follow_trades(
            source,
            quantity=2,
            target_column="next_open_to_open",
        )

        self.assertEqual(trades["direction"].tolist(), ["LONG", "SHORT"])
        self.assertEqual(trades["target_move"].tolist(), [7.0, -2.0])
        self.assertEqual(trades["pnl"].tolist(), [14.0, 4.0])
        self.assertEqual(trades["target_column"].unique().tolist(), ["next_open_to_open"])

    def test_group_by_sentiment_returns_full_range_with_pnl_counts(self) -> None:
        trades = pd.DataFrame(
            [
                {"sentiment": -1, "pnl": 5.0},
                {"sentiment": -1, "pnl": -2.0},
                {"sentiment": 2, "pnl": 3.0},
            ]
        )

        grouped = group_by_sentiment(trades)

        self.assertEqual(grouped["sentiment"].tolist(), list(range(-10, 11)))
        minus_one = grouped[grouped["sentiment"] == -1].iloc[0]
        self.assertEqual(int(minus_one["count_pos"]), 1)
        self.assertEqual(int(minus_one["count_neg"]), 1)
        self.assertEqual(float(minus_one["total_pnl"]), 3.0)
        self.assertEqual(int(minus_one["trades"]), 2)
        zero = grouped[grouped["sentiment"] == 0].iloc[0]
        self.assertEqual(int(zero["trades"]), 0)

    def test_rules_recommendation_uses_nearest_nonzero_total_pnl_for_zero(self) -> None:
        total_pnl = pd.Series(
            {
                -2: -10.0,
                -1: 100.0,
                0: 0.0,
                1: -50.0,
                2: 20.0,
            }
        )

        self.assertEqual(recommend_action(total_pnl, 0), "follow")
        self.assertEqual(recommend_action(total_pnl, -2), "invert")

    def test_build_rules_recommendation_creates_one_rule_per_sentiment(self) -> None:
        grouped = pd.DataFrame(
            {
                "sentiment": list(range(-10, 11)),
                "total_pnl": [-1.0] * 10 + [0.0] + [2.0] * 10,
            }
        )

        rules = build_rules_recommendation(grouped)

        self.assertEqual(len(rules), 21)
        self.assertEqual(rules[0], {"min": -10, "max": -10, "action": "invert"})
        self.assertEqual(rules[10], {"min": 0, "max": 0, "action": "follow"})
        self.assertEqual(rules[-1], {"min": 10, "max": 10, "action": "follow"})

    def test_build_backtest_applies_rules_to_selected_target_column(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "source_date": "2025-09-02",
                    "sentiment": 5,
                    "next_body": 1.0,
                    "next_open_to_open": 10.0,
                },
                {
                    "source_date": "2025-09-03",
                    "sentiment": -3,
                    "next_body": 1.0,
                    "next_open_to_open": -4.0,
                },
                {
                    "source_date": "2025-09-04",
                    "sentiment": 0,
                    "next_body": 1.0,
                    "next_open_to_open": 99.0,
                },
            ]
        )
        rules = [
            {"min": 5, "max": 5, "action": "invert"},
            {"min": -3, "max": -3, "action": "follow"},
            {"min": 0, "max": 0, "action": "skip"},
        ]

        result = build_backtest(
            source,
            quantity=1,
            rules=rules,
            target_column="next_open_to_open",
        )

        self.assertEqual(result["direction"].tolist(), ["SHORT", "SHORT"])
        self.assertEqual(result["target_move"].tolist(), [10.0, -4.0])
        self.assertEqual(result["pnl"].tolist(), [-10.0, 4.0])
        self.assertEqual(result["cum_pnl"].tolist(), [-10.0, -6.0])

    def test_backtest_report_includes_weekly_and_monthly_pnl_after_top_charts(self) -> None:
        result = pd.DataFrame(
            [
                {
                    "source_date": "2025-09-02",
                    "sentiment": 5,
                    "action": "follow",
                    "direction": "LONG",
                    "target_column": "next_body",
                    "target_move": 10.0,
                    "quantity": 1,
                    "pnl": 10.0,
                    "cum_pnl": 10.0,
                },
                {
                    "source_date": "2025-09-09",
                    "sentiment": -2,
                    "action": "invert",
                    "direction": "LONG",
                    "target_column": "next_body",
                    "target_move": -4.0,
                    "quantity": 1,
                    "pnl": -4.0,
                    "cum_pnl": 6.0,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_html = Path(tmp) / "report.html"
            build_backtest_report_html(
                result,
                symbol="BTCUSDT",
                model_key="fake_model",
                target_column="next_body",
                rules_yaml=Path("rules_next_body.yaml"),
                output_html=output_html,
            )

            html = (
                output_html.read_text(encoding="utf-8")
                .encode("latin1", "backslashreplace")
                .decode("unicode_escape")
            )

        trade_index = html.index("P/L по сделкам")
        equity_index = html.index("Накопленная прибыль (equity)")
        weekly_index = html.index("P/L по неделям")
        monthly_index = html.index("P/L по месяцам")
        drawdown_index = html.index("Drawdown от максимума")
        self.assertLess(trade_index, weekly_index)
        self.assertLess(equity_index, weekly_index)
        self.assertLess(weekly_index, drawdown_index)
        self.assertLess(monthly_index, drawdown_index)
        self.assertIn("Backtest: статистика стратегии", html)
        self.assertIn("Backtest: ключевые коэффициенты", html)
        self.assertIn("Прогноз на следующий месяц", html)

    def test_load_sentiment_research_config_uses_reports_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = root / "settings.yaml"
            settings.write_text(
                "\n".join(
                    [
                        "mexc:",
                        "  symbol: BTCUSDT",
                        "daily:",
                        "  sqlite_path: data/daily_msk.db",
                        "news:",
                        "  markdown_dir: data/news_md",
                        "sentiment:",
                        "  output_dir: data/sentiment/BTCUSDT",
                        "  models:",
                        "    fake_model:",
                        "      enabled: true",
                        "      ollama_model: fake:1",
                        "reports:",
                        "  output_dir: reports/sentiment/BTCUSDT",
                        "  quantity: 3",
                        "  date_from: 2025-09-01",
                        "  date_to: 2025-10-01",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_sentiment_research_config(settings)

            self.assertEqual(config.symbol, "BTCUSDT")
            self.assertEqual(config.output_dir, root / "reports" / "sentiment" / "BTCUSDT")
            self.assertEqual(config.quantity, 3)
            self.assertEqual(config.target_column, "next_body")
            self.assertEqual(config.model_pkl("fake_model"), root / "data" / "sentiment" / "BTCUSDT" / "fake_model" / "sentiment_scores.pkl")
            self.assertEqual(config.selected_model_keys(None), ["fake_model"])

    def test_load_sentiment_research_config_rejects_unknown_target_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = root / "settings.yaml"
            settings.write_text(
                "\n".join(
                    [
                        "mexc:",
                        "  symbol: BTCUSDT",
                        "daily:",
                        "  sqlite_path: data/daily_msk.db",
                        "news:",
                        "  markdown_dir: data/news_md",
                        "sentiment:",
                        "  output_dir: data/sentiment/BTCUSDT",
                        "  models:",
                        "    fake_model:",
                        "      enabled: true",
                        "      ollama_model: fake:1",
                        "reports:",
                        "  target_column: body",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SettingsError, "target_column"):
                load_sentiment_research_config(settings)

    def test_valid_target_columns_are_the_two_supported_market_features(self) -> None:
        self.assertEqual(VALID_TARGET_COLUMNS, {"next_body", "next_open_to_open"})


if __name__ == "__main__":
    unittest.main()
