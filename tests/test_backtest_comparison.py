import tempfile
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pj22_btc.backtest_comparison import (  # noqa: E402
    ComparisonPair,
    build_report,
    discover_pairs,
    open_report_in_chrome,
    prepare_comparison,
)
from scripts.create_backtest_comparison_report import parse_model_keys, resolve_cli_path  # noqa: E402


class BacktestComparisonTests(unittest.TestCase):
    def test_discover_pairs_builds_paths_for_models_and_targets(self) -> None:
        reports_dir = Path("reports/sentiment/BTCUSDT")
        walk_dir = Path("reports/walk_forward")

        pairs = discover_pairs(
            symbol="BTCUSDT",
            reports_dir=reports_dir,
            walk_forward_dir=walk_dir,
            model_keys=["gemma3_12b", "qwen3_14b"],
            target_columns=["next_body", "next_open_to_open"],
        )

        self.assertEqual(len(pairs), 4)
        first = pairs[0]
        self.assertEqual(first.symbol, "BTCUSDT")
        self.assertEqual(first.model_key, "gemma3_12b")
        self.assertEqual(first.target_column, "next_body")
        self.assertEqual(
            first.ordinary_path,
            reports_dir
            / "gemma3_12b"
            / "backtest"
            / "sentiment_backtest_next_body_results.xlsx",
        )
        self.assertEqual(
            first.walk_path,
            walk_dir / "BTCUSDT" / "gemma3_12b" / "next_body" / "trades.xlsx",
        )

    def test_prepare_comparison_calculates_metrics_on_overlap_dates(self) -> None:
        pair = ComparisonPair(
            symbol="BTCUSDT",
            model_key="fake_model",
            target_column="next_body",
            ordinary_path=Path("ordinary.xlsx"),
            walk_path=Path("walk.xlsx"),
        )
        ordinary = pd.DataFrame(
            [
                _trade("2025-01-01", "follow", "LONG", 3.0),
                _trade("2025-01-02", "follow", "LONG", 10.0),
                _trade("2025-01-03", "invert", "SHORT", -4.0),
            ]
        )
        walk = pd.DataFrame(
            [
                _trade("2025-01-02", "follow", "LONG", 7.0),
                _trade("2025-01-03", "follow", "LONG", 5.0),
            ]
        )

        comparison = prepare_comparison(pair=pair, ordinary=ordinary, walk=walk)

        self.assertIsNone(comparison.error)
        self.assertEqual(comparison.metrics["start_date"], date(2025, 1, 2))
        self.assertEqual(comparison.metrics["end_date"], date(2025, 1, 3))
        self.assertEqual(comparison.metrics["overlap_rows"], 2)
        self.assertEqual(comparison.metrics["ordinary_total_pnl"], 6.0)
        self.assertEqual(comparison.metrics["walk_total_pnl"], 12.0)
        self.assertEqual(comparison.metrics["delta_pnl"], 6.0)
        self.assertEqual(comparison.metrics["ordinary_max_drawdown"], -4.0)
        self.assertEqual(comparison.metrics["walk_max_drawdown"], 0.0)
        self.assertEqual(comparison.metrics["ordinary_win_rate"], 50.0)
        self.assertEqual(comparison.metrics["walk_win_rate"], 100.0)
        self.assertEqual(comparison.metrics["signal_match_rate"], 50.0)

    def test_build_report_writes_target_sections_and_missing_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports_dir = root / "reports" / "sentiment" / "BTCUSDT"
            walk_dir = root / "reports" / "walk_forward"
            ordinary_path = (
                reports_dir
                / "fake_model"
                / "backtest"
                / "sentiment_backtest_next_body_results.xlsx"
            )
            walk_path = walk_dir / "BTCUSDT" / "fake_model" / "next_body" / "trades.xlsx"
            ordinary_path.parent.mkdir(parents=True)
            walk_path.parent.mkdir(parents=True)
            pd.DataFrame([_trade("2025-01-02", "follow", "LONG", 10.0)]).to_excel(
                ordinary_path,
                index=False,
            )
            pd.DataFrame([_trade("2025-01-02", "follow", "LONG", 8.0)]).to_excel(
                walk_path,
                index=False,
            )
            output_html = root / "reports" / "backtest_comparison" / "report.html"

            result = build_report(
                symbol="BTCUSDT",
                reports_dir=reports_dir,
                walk_forward_dir=walk_dir,
                output_html=output_html,
                model_keys=["fake_model"],
                target_columns=["next_body", "next_open_to_open"],
            )

            html = output_html.read_text(encoding="utf-8")
            self.assertEqual(result.output_html, output_html)
            self.assertEqual(len(result.comparisons), 1)
            self.assertEqual(len(result.errors), 1)
            self.assertIn("next_body", html)
            self.assertIn("next_open_to_open", html)
            self.assertIn("fake_model", html)
            self.assertIn("Ошибки и пропуски", html)
            self.assertIn("ordinary backtest", html)

    def test_open_report_in_chrome_uses_new_window_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chrome = root / "chrome.exe"
            html = root / "report.html"
            chrome.write_text("", encoding="utf-8")
            html.write_text("<html></html>", encoding="utf-8")
            commands: list[list[str]] = []

            open_report_in_chrome(
                html,
                chrome_path=chrome,
                popen=lambda command: commands.append(command),
            )

            self.assertEqual(commands, [[str(chrome), "--new-window", str(html)]])


class BacktestComparisonCliTests(unittest.TestCase):
    def test_parse_model_keys_handles_csv_and_empty_values(self) -> None:
        self.assertEqual(parse_model_keys(" gemma3_12b, qwen3_14b "), ["gemma3_12b", "qwen3_14b"])
        self.assertIsNone(parse_model_keys(None))
        self.assertIsNone(parse_model_keys(" , "))

    def test_resolve_cli_path_uses_project_root_for_relative_paths(self) -> None:
        root = Path(r"C:\project")

        self.assertEqual(
            resolve_cli_path(root, Path("reports/out.html")),
            root / "reports" / "out.html",
        )
        self.assertEqual(
            resolve_cli_path(root, Path(r"C:\absolute\out.html")),
            Path(r"C:\absolute\out.html"),
        )


def _trade(source_date: str, action: str, direction: str, pnl: float) -> dict[str, object]:
    return {
        "source_date": source_date,
        "sentiment": 1.0,
        "action": action,
        "direction": direction,
        "target_column": "next_body",
        "target_move": pnl,
        "quantity": 1,
        "pnl": pnl,
    }


if __name__ == "__main__":
    unittest.main()
