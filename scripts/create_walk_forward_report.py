r"""Собрать HTML/XLSX отчет по уже созданным walk-forward результатам.

Примеры запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\create_walk_forward_report.py
.\.venv\Scripts\python.exe scripts\create_walk_forward_report.py --target-column next_open_to_open
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

from pj22_btc.mexc_downloader import SettingsError  # noqa: E402
from pj22_btc.sentiment_research import VALID_TARGET_COLUMNS  # noqa: E402
from pj22_btc.walk_forward.core import load_walk_forward_config  # noqa: E402
from pj22_btc.walk_forward.report import build_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Собрать отчет по сохраненным walk-forward CSV."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT / "settings.yaml",
        help="Путь к settings.yaml. По умолчанию используется файл в корне проекта.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Папка результатов. По умолчанию walk_forward.output_dir.",
    )
    parser.add_argument(
        "--target-column",
        choices=sorted(VALID_TARGET_COLUMNS),
        default=None,
        help="Колонка движения. По умолчанию walk_forward.target_column.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Явный путь HTML-отчета.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=None,
        help="Явный путь XLSX-отчета.",
    )
    return parser


def main() -> int:
    """Build report from existing summary files."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_walk_forward_config(args.settings)
        target_column = args.target_column or config.target_column
        output_dir = args.output_dir or config.output_dir
        html_path, xlsx_path = build_report(
            output_dir,
            target_column=target_column,
            output_html=args.output_html,
            output_xlsx=args.output_xlsx,
        )
        print(f"HTML: {html_path}")
        print(f"Excel: {xlsx_path}")
    except (SettingsError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
