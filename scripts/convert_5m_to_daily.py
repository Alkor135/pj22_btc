r"""Конвертация пятиминутных свечей BTCUSDT в дневные свечи по Москве.

Скрипт читает годовые SQLite DB с 5m-свечами из пути `storage.sqlite_dir`
в `settings.yaml`, переводит UTC-время свечей в московское время `MSK`
(`UTC+03:00`) и собирает дневные свечи по правилу:
`[21:00 предыдущего дня; 21:00 текущего дня)`.

Свеча ровно в `21:00 MSK` не входит в завершающуюся дневную свечу, а
относится к следующей дневной сессии.

Пример запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\convert_5m_to_daily.py
```

Пример запуска с явным путем к настройкам:

```powershell
.\.venv\Scripts\python.exe scripts\convert_5m_to_daily.py --settings settings.yaml
```
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pj22_btc.daily_converter import convert_5m_to_daily, load_daily_converter_config
from pj22_btc.mexc_downloader import SettingsError


def build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер аргументов для выбора файла настроек."""
    parser = argparse.ArgumentParser(
        description="Конвертировать 5m-свечи BTCUSDT в дневные свечи по MSK-сессии."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT / "settings.yaml",
        help="Путь к settings.yaml. По умолчанию используется файл в корне проекта.",
    )
    return parser


def main() -> int:
    """Загружает настройки, запускает конвертацию и печатает итоговую сводку."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_daily_converter_config(args.settings)
        summary = convert_5m_to_daily(
            source_dir=config.source_dir,
            output_dir=config.output_dir,
            symbol=config.symbol,
            include_incomplete=config.include_incomplete,
        )
    except (SettingsError, sqlite3.Error, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print("Конвертация 5m -> 1d_msk завершена.")
    print(f"Символ: {summary.symbol}")
    print(f"Источник 5m DB: {summary.source_dir}")
    print(f"Папка daily DB: {summary.output_dir}")
    print(f"Прочитано 5m-свечей: {summary.source_rows}")
    print(f"Сформировано дневных свечей: {summary.daily_rows}")
    print(f"Записано/обновлено строк в SQLite: {summary.inserted_rows}")
    print(f"Пропущено неполных сессий: {summary.skipped_incomplete_sessions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
