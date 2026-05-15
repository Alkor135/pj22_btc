r"""Собрать HTML-сравнение ordinary backtest и walk-forward.

Примеры запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\create_backtest_comparison_report.py
.\.venv\Scripts\python.exe scripts\create_backtest_comparison_report.py --models gemma3_12b,qwen3_14b
.\.venv\Scripts\python.exe scripts\create_backtest_comparison_report.py --no-open
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pj22_btc.backtest_comparison import (  # noqa: E402
    DEFAULT_OUTPUT_HTML,
    TARGET_COLUMNS,
    build_report,
    open_report_in_chrome,
)
from pj22_btc.html_reports import DEFAULT_CHROME_PATH  # noqa: E402
from pj22_btc.mexc_downloader import SettingsError  # noqa: E402
from pj22_btc.sentiment_research import load_sentiment_research_config  # noqa: E402
from pj22_btc.walk_forward.core import load_walk_forward_config  # noqa: E402


def parse_model_keys(value: str | None) -> list[str] | None:
    """Parse comma-separated model keys from CLI."""
    if value is None:
        return None
    keys = [item.strip() for item in value.split(",") if item.strip()]
    return keys or None


def resolve_cli_path(root: Path, path: Path) -> Path:
    """Resolve a CLI path relative to project root unless it is absolute."""
    return path if path.is_absolute() else root / path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Собрать сравнение ordinary sentiment backtest и walk-forward."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT / "settings.yaml",
        help="Путь к settings.yaml. По умолчанию используется файл в корне проекта.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Модели через запятую. Если не задано, берутся sentiment.models с enabled=true.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Папка ordinary sentiment-отчетов. По умолчанию reports.output_dir.",
    )
    parser.add_argument(
        "--walk-forward-dir",
        type=Path,
        default=None,
        help="Папка walk-forward результатов. По умолчанию walk_forward.output_dir.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=DEFAULT_OUTPUT_HTML,
        help="Итоговый HTML-файл.",
    )
    parser.add_argument(
        "--chrome-path",
        type=Path,
        default=DEFAULT_CHROME_PATH,
        help="Путь к chrome.exe для открытия отчета.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Не открывать итоговый HTML в Chrome.",
    )
    return parser


def main() -> int:
    """Build comparison HTML and optionally open it in Chrome."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        settings_path = resolve_cli_path(ROOT, args.settings)
        research_config = load_sentiment_research_config(settings_path)
        walk_config = load_walk_forward_config(settings_path)
        model_keys = research_config.selected_model_keys(parse_model_keys(args.models))
        reports_dir = (
            resolve_cli_path(ROOT, args.reports_dir)
            if args.reports_dir is not None
            else research_config.output_dir
        )
        walk_forward_dir = (
            resolve_cli_path(ROOT, args.walk_forward_dir)
            if args.walk_forward_dir is not None
            else walk_config.output_dir
        )
        output_html = resolve_cli_path(ROOT, args.output_html)

        result = build_report(
            symbol=research_config.symbol,
            reports_dir=reports_dir,
            walk_forward_dir=walk_forward_dir,
            output_html=output_html,
            model_keys=model_keys,
            target_columns=TARGET_COLUMNS,
        )

        print(f"HTML: {result.output_html}")
        print(f"Сопоставимых пар: {len(result.comparisons)}")
        print(f"Ошибок и пропусков: {len(result.errors)}")

        if not args.no_open:
            try:
                open_report_in_chrome(
                    result.output_html,
                    chrome_path=resolve_cli_path(ROOT, args.chrome_path),
                )
                print("Открыто в Chrome.")
            except OSError as exc:
                print(f"HTML создан, но Chrome не открыт: {exc}", file=sys.stderr)

    except (SettingsError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
