import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pj22_btc.walk_forward.core import (  # noqa: E402
    load_walk_forward_config,
    run_walk_forward_day,
    run_walk_forward_model,
    save_global_summary,
    save_model_outputs,
)
from pj22_btc.walk_forward.report import build_report  # noqa: E402
from scripts.run_walk_forward import parse_cli_date, parse_model_keys  # noqa: E402


class WalkForwardTests(unittest.TestCase):
    def test_cli_helpers_parse_models_and_dates(self) -> None:
        self.assertEqual(parse_model_keys(" gemma3_12b, qwen3_14b "), ["gemma3_12b", "qwen3_14b"])
        self.assertIsNone(parse_model_keys(None))
        self.assertEqual(parse_cli_date("2025-09-01"), date(2025, 9, 1))

    def test_load_walk_forward_config_uses_project_settings_and_model_pkls(self) -> None:
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
                        "  target_column: next_open_to_open",
                        "  date_from: 2025-09-01",
                        "  date_to: 2025-10-01",
                        "walk_forward:",
                        "  output_dir: reports/walk_forward",
                        "  train_months: 4",
                        "  min_train_rows: 7",
                        "  save_daily_artifacts: true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_walk_forward_config(settings)

            self.assertEqual(config.symbol, "BTCUSDT")
            self.assertEqual(config.output_dir, root / "reports" / "walk_forward")
            self.assertEqual(config.quantity, 3)
            self.assertEqual(config.target_column, "next_open_to_open")
            self.assertEqual(config.backtest_start_date, date(2025, 9, 1))
            self.assertEqual(config.backtest_end_date, date(2025, 10, 1))
            self.assertEqual(config.train_months, 4)
            self.assertEqual(config.min_train_rows, 7)
            self.assertTrue(config.save_daily_artifacts)
            self.assertEqual(
                config.model_pkl("fake_model"),
                root / "data" / "sentiment" / "BTCUSDT" / "fake_model" / "sentiment_scores.pkl",
            )
            self.assertEqual(config.selected_model_keys(None), ["fake_model"])

    def test_run_walk_forward_day_excludes_test_date_from_training_window(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "source_date": "2025-01-01",
                    "sentiment": 1,
                    "next_body": 10.0,
                    "next_open_to_open": 10.0,
                },
                {
                    "source_date": "2025-01-02",
                    "sentiment": 1,
                    "next_body": 10.0,
                    "next_open_to_open": 10.0,
                },
                {
                    "source_date": "2025-01-03",
                    "sentiment": 1,
                    "next_body": -1000.0,
                    "next_open_to_open": -1000.0,
                },
            ]
        )

        day = run_walk_forward_day(
            source,
            symbol="BTCUSDT",
            model_key="fake_model",
            quantity=1,
            target_column="next_body",
            test_date=date(2025, 1, 3),
            train_months=1,
            min_train_rows=2,
        )

        self.assertEqual(day.summary["status"], "ok")
        self.assertEqual(day.summary["train_rows"], 2)
        self.assertEqual(day.summary["test_rows"], 1)
        self.assertEqual(day.summary["train_end"], date(2025, 1, 2))
        self.assertEqual(day.trade["action"], "follow")
        self.assertEqual(day.trade["direction"], "LONG")
        self.assertEqual(day.trade["pnl"], -1000.0)

    def test_run_walk_forward_model_and_save_outputs_include_target_column_paths(self) -> None:
        source = pd.DataFrame(
            [
                {"source_date": "2025-01-01", "sentiment": 1, "next_body": 2.0},
                {"source_date": "2025-01-02", "sentiment": -1, "next_body": 3.0},
                {"source_date": "2025-01-03", "sentiment": 1, "next_body": 4.0},
            ]
        )

        result = run_walk_forward_model(
            source,
            symbol="BTCUSDT",
            model_key="fake_model",
            quantity=1,
            target_column="next_body",
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 3),
            train_months=1,
            min_train_rows=1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "wf"
            save_model_outputs(
                output_dir=output_dir,
                symbol="BTCUSDT",
                model_key="fake_model",
                target_column="next_body",
                result=result,
                save_daily_artifacts=True,
            )
            save_global_summary(
                output_dir=output_dir,
                target_column="next_body",
                daily_summaries=result.daily_summaries,
                model_summaries=[result.model_summary],
            )

            model_dir = output_dir / "BTCUSDT" / "fake_model" / "next_body"
            self.assertTrue((model_dir / "trades.csv").exists())
            self.assertTrue((model_dir / "trades.xlsx").exists())
            self.assertTrue((model_dir / "daily_summary.csv").exists())
            self.assertTrue((model_dir / "summary.json").exists())
            self.assertTrue((model_dir / "daily" / "2025-01-02" / "group_stats.xlsx").exists())
            self.assertTrue((model_dir / "daily" / "2025-01-02" / "rules.yaml").exists())
            self.assertTrue((output_dir / "summary_next_body.csv").exists())
            self.assertTrue((output_dir / "summary_next_body.xlsx").exists())
            self.assertTrue((output_dir / "model_summary_next_body.csv").exists())

            summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["symbol"], "BTCUSDT")
            self.assertEqual(summary["model_key"], "fake_model")
            self.assertEqual(summary["target_column"], "next_body")

    def test_build_report_uses_saved_summary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "wf"
            model_dir = output_dir / "BTCUSDT" / "fake_model" / "next_body"
            model_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "source_date": date(2025, 1, 2),
                        "symbol": "BTCUSDT",
                        "model_key": "fake_model",
                        "target_column": "next_body",
                        "pnl": 5.0,
                        "cum_pnl": 5.0,
                    }
                ]
            ).to_csv(model_dir / "trades.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "symbol": "BTCUSDT",
                        "model_key": "fake_model",
                        "target_column": "next_body",
                        "source_date": date(2025, 1, 2),
                        "status": "ok",
                        "pnl": 5.0,
                    }
                ]
            ).to_csv(output_dir / "summary_next_body.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "symbol": "BTCUSDT",
                        "model_key": "fake_model",
                        "target_column": "next_body",
                        "trades": 1,
                        "total_pnl": 5.0,
                        "winrate": 100.0,
                    }
                ]
            ).to_csv(output_dir / "model_summary_next_body.csv", index=False, encoding="utf-8-sig")

            html_path, xlsx_path = build_report(output_dir, target_column="next_body")

            self.assertEqual(html_path, output_dir / "walk_forward_report_next_body.html")
            self.assertEqual(xlsx_path, output_dir / "walk_forward_report_next_body.xlsx")
            self.assertTrue(html_path.exists())
            self.assertTrue(xlsx_path.exists())
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Walk-forward report", html)
            self.assertIn("fake_model", html)


if __name__ == "__main__":
    unittest.main()
