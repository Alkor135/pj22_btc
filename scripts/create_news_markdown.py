r"""Создание markdown-файлов с BTC/crypto-заголовками под дневные свечи.

Скрипт читает настройки из `settings.yaml`, берет интервалы дневных свечей из
`daily.sqlite_path`, фильтрует RSS-заголовки по `news.crypto_title_regex` и
создает файлы `YYYY-MM-DD.md` в `news.markdown_dir`.

Пример запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\create_news_markdown.py
```
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pj22_btc.mexc_downloader import SettingsError
from pj22_btc.news_markdown import create_news_markdown_files, load_news_markdown_config


def build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер аргументов для выбора файла настроек."""
    parser = argparse.ArgumentParser(
        description="Создать markdown-файлы с BTC/crypto RSS-заголовками под 1d_msk свечи."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT / "settings.yaml",
        help="Путь к settings.yaml. По умолчанию используется файл в корне проекта.",
    )
    return parser


def main() -> int:
    """Загружает настройки, запускает генерацию и печатает итоговую сводку."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_news_markdown_config(args.settings)
        summary = create_news_markdown_files(
            daily_db=config.daily_db,
            news_db_dir=config.news_db_dir,
            markdown_dir=config.markdown_dir,
            symbol=config.symbol,
            title_regex=config.title_regex,
        )
    except (SettingsError, sqlite3.Error, re.error, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print("Создание markdown-файлов с BTC/crypto-новостями завершено.")
    print(f"Символ: {summary.symbol}")
    print(f"Daily DB: {summary.daily_db}")
    print(f"Папка RSS DB: {summary.news_db_dir}")
    print(f"Папка markdown: {summary.markdown_dir}")
    print(f"Прочитано дневных сессий: {summary.sessions_read}")
    print(f"Прочитано RSS-строк: {summary.news_rows_read}")
    print(f"Совпало с regex до удаления дублей: {summary.filtered_news_rows}")
    print(f"Уникальных regex-заголовков: {summary.unique_news_rows}")
    print(f"Удалено последних markdown-файлов: {summary.files_deleted}")
    print(f"Создано/обновлено markdown-файлов: {summary.files_written}")
    print(f"Последний созданный markdown-файл: {summary.latest_markdown_file or '-'}")
    print(f"Записано заголовков в markdown: {summary.titles_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
