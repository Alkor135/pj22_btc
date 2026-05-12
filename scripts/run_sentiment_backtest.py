r"""Запустить backtest sentiment-стратегии по rules YAML.

Примеры запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\run_sentiment_backtest.py
.\.venv\Scripts\python.exe scripts\run_sentiment_backtest.py --models gemma3_12b
.\.venv\Scripts\python.exe scripts\run_sentiment_backtest.py --target-column next_open_to_open
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pj22_btc.mexc_downloader import SettingsError
from pj22_btc.sentiment_research import (
    VALID_TARGET_COLUMNS,
    load_sentiment_research_config,
    run_backtest_for_model,
)


def parse_model_keys(value: str | None) -> list[str] | None:
    """Разбирает comma-separated список model_key из CLI."""
    if value is None:
        return None
    keys = [item.strip() for item in value.split(",") if item.strip()]
    return keys or None


def parse_cli_date(value: str | None):
    """Преобразует CLI-дату в date или возвращает None."""
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise argparse.ArgumentTypeError(f"Некорректная дата: {value!r}")
    return parsed.date()


def build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер аргументов."""
    parser = argparse.ArgumentParser(
        description="Запустить sentiment backtest для выбранных моделей."
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
        "--target-column",
        choices=sorted(VALID_TARGET_COLUMNS),
        default=None,
        help="Колонка движения для P/L. По умолчанию reports.target_column или next_body.",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=None,
        help="Количество контрактов/единиц на сделку. По умолчанию reports.quantity.",
    )
    parser.add_argument(
        "--date-from",
        type=parse_cli_date,
        default=None,
        help="Нижняя граница окна YYYY-MM-DD. По умолчанию reports.date_from.",
    )
    parser.add_argument(
        "--date-to",
        type=parse_cli_date,
        default=None,
        help="Верхняя граница окна YYYY-MM-DD. По умолчанию reports.date_to.",
    )
    parser.add_argument(
        "--rules-yaml",
        type=Path,
        default=None,
        help="Явный rules YAML. Разрешено только при выборе одной модели.",
    )
    return parser


def main() -> int:
    """Запускает backtest и печатает краткую сводку."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_sentiment_research_config(args.settings)
        target_column = args.target_column or config.target_column
        model_keys = config.selected_model_keys(parse_model_keys(args.models))
        if args.rules_yaml is not None and len(model_keys) != 1:
            raise ValueError("--rules-yaml можно использовать только с одной моделью")

        print("Запуск sentiment backtest.")
        print(f"Настройки: {args.settings}")
        print(f"Символ: {config.symbol}")
        print(f"Модели: {', '.join(model_keys)}")
        print(f"Target column: {target_column}")

        for model_key in model_keys:
            result = run_backtest_for_model(
                config,
                model_key,
                target_column=target_column,
                quantity=args.quantity,
                date_from=args.date_from,
                date_to=args.date_to,
                rules_yaml=args.rules_yaml,
            )
            pnl = result.result["pnl"]
            print("")
            print(f"Модель: {model_key}")
            print(f"Rules: {result.rules_yaml}")
            print(f"Сделок: {len(result.result)}")
            print(f"Общий P/L: {pnl.sum():.2f}")
            print(f"Доля прибыльных сделок: {(pnl > 0).mean() * 100:.1f}%")
            print(f"XLSX: {result.output_xlsx}")
            print(f"HTML: {result.output_html}")
    except (SettingsError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
