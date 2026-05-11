r"""Загрузка пятиминутных свечей BTCUSDT с MEXC в годовые SQLite-базы.

Скрипт читает настройки из `settings.yaml`, скачивает исторические свечи
через публичный Spot API MEXC и сохраняет данные в SQLite-файлы по годам:
`data/mexc/klines/BTCUSDT/5m/YYYY.db`.

Дата начала загрузки задается в `settings.yaml`:

```yaml
mexc:
  start_date: 2025-09-01
```

Пример запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\download_mexc_klines.py
```

Пример запуска с явным путем к настройкам:

```powershell
.\.venv\Scripts\python.exe scripts\download_mexc_klines.py --settings settings.yaml
```
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pj22_btc.mexc_downloader import MexcAPIError, SettingsError, load_config, sync_klines


def format_ms(value: int | None) -> str:
    """Форматирует timestamp в миллисекундах в читаемую UTC-дату для вывода в консоль."""
    if value is None:
        return "-"
    return datetime.fromtimestamp(value / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер аргументов для выбора файла настроек."""
    parser = argparse.ArgumentParser(
        description="Скачать пятиминутные свечи BTCUSDT с MEXC в годовые SQLite-базы."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT / "settings.yaml",
        help="Путь к settings.yaml. По умолчанию используется файл в корне проекта.",
    )
    return parser


def main() -> int:
    """Загружает настройки, запускает докачку свечей и печатает итоговую сводку."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.settings)
        summary = sync_klines(config)
    except (MexcAPIError, SettingsError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Загрузка прервана пользователем.", file=sys.stderr)
        return 130

    print("Загрузка MEXC klines завершена.")
    print(f"Символ: {summary.symbol}")
    print(f"Интервал: {summary.interval}")
    print(f"Папка SQLite DB: {summary.db_dir}")
    print(f"Начальная дата из настроек: {format_ms(summary.start_ms)}")
    print(f"Первый запрошенный cursor: {format_ms(summary.first_requested_ms)}")
    print(f"Последняя обработанная свеча (open time): {format_ms(summary.last_open_time_ms)}")
    print(f"Закрытие последней обработанной свечи: {format_ms(summary.last_close_time_ms)}")
    print(f"Получено строк от MEXC: {summary.fetched_rows}")
    print(f"Добавлено новых строк в SQLite: {summary.inserted_rows}")
    print(f"Запросов к MEXC: {summary.batches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
