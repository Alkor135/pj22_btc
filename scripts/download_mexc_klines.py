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
    if value is None:
        return "-"
    return datetime.fromtimestamp(value / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Скачать одноминутные свечи BTCUSDT с MEXC в месячные SQLite-базы."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT / "settings.yaml",
        help="Путь к settings.yaml. По умолчанию используется файл в корне проекта.",
    )
    return parser


def main() -> int:
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
    print(f"Последняя обработанная свеча: {format_ms(summary.last_open_time_ms)}")
    print(f"Получено строк от MEXC: {summary.fetched_rows}")
    print(f"Добавлено новых строк в SQLite: {summary.inserted_rows}")
    print(f"Запросов к MEXC: {summary.batches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

