# Backtest Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native BTCUSDT report that compares ordinary sentiment backtests with walk-forward backtests for both supported target columns and opens the resulting HTML in Chrome.

**Architecture:** Add a focused `pj22_btc.backtest_comparison` module for pair discovery, trade normalization, overlap metrics, Plotly HTML rendering, report writing, and Chrome opening. Add a thin CLI script that reads existing project settings, selects models, invokes the module, and handles non-fatal Chrome-open failures.

**Tech Stack:** Python 3.13, pandas, plotly, openpyxl, unittest, existing `pj22_btc.sentiment_research`, existing `pj22_btc.walk_forward.core`, existing `pj22_btc.html_reports`.

---

## File Structure

- Create `src/pj22_btc/backtest_comparison.py`: reusable implementation for comparing saved ordinary and walk-forward artifacts.
- Create `scripts/create_backtest_comparison_report.py`: CLI entrypoint with settings/model parsing and default Chrome opening.
- Create `tests/test_backtest_comparison.py`: unit tests for pair discovery, metrics, HTML output with missing-file errors, Chrome command helper, and CLI model parsing.
- Modify `README.md`: document the new comparison command and output path.

### Task 1: Tests for comparison core

**Files:**
- Create: `tests/test_backtest_comparison.py`
- Later create: `src/pj22_btc/backtest_comparison.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest_comparison.py` with tests for expected paths, overlap metrics, report HTML sections, missing-file errors, and Chrome opening:

```python
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from pj22_btc.backtest_comparison import (
    ComparisonPair,
    build_report,
    discover_pairs,
    open_report_in_chrome,
    prepare_comparison,
)


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
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_backtest_comparison
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `pj22_btc.backtest_comparison` does not exist yet.

### Task 2: Implement comparison module

**Files:**
- Create: `src/pj22_btc/backtest_comparison.py`
- Test: `tests/test_backtest_comparison.py`

- [ ] **Step 1: Write minimal implementation**

Create `src/pj22_btc/backtest_comparison.py` with:

```python
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from pj22_btc.html_reports import DEFAULT_CHROME_PATH, build_chrome_command

TARGET_COLUMNS = ("next_body", "next_open_to_open")
DEFAULT_OUTPUT_HTML = Path("reports/backtest_comparison/backtest_vs_walk_forward.html")
REQUIRED_COLUMNS = {"source_date", "sentiment", "action", "direction", "quantity", "pnl"}


@dataclass(frozen=True)
class ComparisonPair:
    symbol: str
    model_key: str
    target_column: str
    ordinary_path: Path
    walk_path: Path


@dataclass
class PairComparison:
    pair: ComparisonPair
    ordinary: pd.DataFrame
    walk: pd.DataFrame
    metrics: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class ReportResult:
    output_html: Path
    comparisons: list[PairComparison]
    errors: list[dict[str, str]]
```

Then add functions for `discover_pairs`, `normalize_trades`, `prepare_comparison`, HTML rendering, `build_report`, and `open_report_in_chrome` matching the tests and the design spec.

- [ ] **Step 2: Run the core test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_backtest_comparison
```

Expected: PASS for `tests.test_backtest_comparison`.

### Task 3: CLI and CLI tests

**Files:**
- Modify: `tests/test_backtest_comparison.py`
- Create: `scripts/create_backtest_comparison_report.py`

- [ ] **Step 1: Add failing tests for CLI helpers**

Append to `tests/test_backtest_comparison.py`:

```python
from scripts.create_backtest_comparison_report import parse_model_keys, resolve_cli_path


class BacktestComparisonCliTests(unittest.TestCase):
    def test_parse_model_keys_handles_csv_and_empty_values(self) -> None:
        self.assertEqual(parse_model_keys(" gemma3_12b, qwen3_14b "), ["gemma3_12b", "qwen3_14b"])
        self.assertIsNone(parse_model_keys(None))
        self.assertIsNone(parse_model_keys(" , "))

    def test_resolve_cli_path_uses_project_root_for_relative_paths(self) -> None:
        root = Path(r"C:\project")
        self.assertEqual(resolve_cli_path(root, Path("reports/out.html")), root / "reports" / "out.html")
        self.assertEqual(resolve_cli_path(root, Path(r"C:\absolute\out.html")), Path(r"C:\absolute\out.html"))
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_backtest_comparison
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because the CLI script does not exist yet.

- [ ] **Step 3: Implement CLI**

Create `scripts/create_backtest_comparison_report.py` with parser options `--settings`, `--models`, `--reports-dir`, `--walk-forward-dir`, `--output-html`, `--chrome-path`, and `--no-open`. Use `load_sentiment_research_config` and `load_walk_forward_config` for defaults, call `build_report`, print output path and counts, and treat Chrome-open failures as non-fatal.

- [ ] **Step 4: Run the CLI helper tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_backtest_comparison
```

Expected: PASS for comparison and CLI helper tests.

### Task 4: Documentation and full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add a short section after the walk-forward report commands:

```markdown
Сравнение ordinary backtest и walk-forward по двум target-колонкам:

```powershell
.\.venv\Scripts\python.exe scripts\create_backtest_comparison_report.py
.\.venv\Scripts\python.exe scripts\create_backtest_comparison_report.py --models gemma3_12b,qwen3_14b --no-open
```

Итоговый HTML по умолчанию:

```text
reports/backtest_comparison/backtest_vs_walk_forward.html
```
```

- [ ] **Step 2: Run targeted tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_backtest_comparison
```

Expected: PASS.

- [ ] **Step 3: Run the full test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest
```

Expected: all tests pass. If unrelated pre-existing tests fail, capture exact failures and do not claim full-suite success.

- [ ] **Step 4: Run CLI smoke test without Chrome**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\create_backtest_comparison_report.py --no-open
```

Expected: script exits 0, prints the HTML path, and creates `reports/backtest_comparison/backtest_vs_walk_forward.html`. Missing individual model/target files may appear in the report errors section and are not fatal.

## Self-Review

Spec coverage:
- Native module and CLI are covered by Tasks 2 and 3.
- Two target columns in one HTML but separate sections are covered by Task 1 and Task 2.
- Chrome new-window opening and `--no-open` are covered by Tasks 1 and 3.
- Error table for missing artifacts is covered by Task 1 and Task 2.
- README documentation is covered by Task 4.

Placeholder scan: no `TBD`, `TODO`, or unspecified implementation task remains.

Type consistency: `ComparisonPair`, `PairComparison`, `ReportResult`, `discover_pairs`, `prepare_comparison`, `build_report`, `open_report_in_chrome`, `parse_model_keys`, and `resolve_cli_path` are used consistently across tests and implementation tasks.
