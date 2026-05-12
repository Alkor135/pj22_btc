r"""Открыть HTML-отчеты проекта в одном новом окне Google Chrome.

По умолчанию скрипт ищет все `*.html` рекурсивно в:

```text
reports/sentiment/BTCUSDT
```

Каждый найденный HTML передается Chrome отдельным аргументом, поэтому файлы
откроются в одном новом окне, каждый в своей вкладке.

Пример запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\open_html_reports.py
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

from pj22_btc.html_reports import (  # noqa: E402
    DEFAULT_CHROME_PATH,
    build_chrome_command,
    collect_html_reports,
    open_reports_in_chrome,
)

DEFAULT_REPORTS_ROOT = ROOT / "reports" / "sentiment" / "BTCUSDT"


def build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер аргументов."""
    parser = argparse.ArgumentParser(
        description="Открыть все HTML-отчеты в одном новом окне Chrome."
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Корневая папка поиска HTML. По умолчанию reports/sentiment/BTCUSDT.",
    )
    parser.add_argument(
        "--chrome",
        type=Path,
        default=DEFAULT_CHROME_PATH,
        help=f"Путь к chrome.exe. По умолчанию {DEFAULT_CHROME_PATH}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать найденные HTML и команду Chrome, но не открывать браузер.",
    )
    return parser


def main() -> int:
    """Находит HTML-отчеты и открывает их в новом окне Chrome."""
    parser = build_parser()
    args = parser.parse_args()

    reports_root = Path(args.reports_root)
    reports = collect_html_reports(reports_root)
    if not reports:
        print(f"HTML-файлы не найдены: {reports_root}")
        return 0

    print(f"Найдено HTML-файлов: {len(reports)}")
    for report in reports:
        try:
            label = report.relative_to(reports_root)
        except ValueError:
            label = report
        print(f"  [HTML] {label}")

    command = build_chrome_command(args.chrome, reports)
    if args.dry_run:
        print("\nКоманда Chrome:")
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
        return 0

    try:
        open_reports_in_chrome(args.chrome, reports)
    except FileNotFoundError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"\nОткрываю в новом окне Chrome: {args.chrome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
