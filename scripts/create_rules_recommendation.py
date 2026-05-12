r"""Сгенерировать rules YAML из sentiment group stats.

Примеры запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\create_rules_recommendation.py
.\.venv\Scripts\python.exe scripts\create_rules_recommendation.py --models gemma3_12b
.\.venv\Scripts\python.exe scripts\create_rules_recommendation.py --target-column next_open_to_open
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

from pj22_btc.mexc_downloader import SettingsError
from pj22_btc.sentiment_research import (
    VALID_TARGET_COLUMNS,
    load_sentiment_research_config,
    run_rules_recommendation_for_model,
)


def parse_model_keys(value: str | None) -> list[str] | None:
    """Разбирает comma-separated список model_key из CLI."""
    if value is None:
        return None
    keys = [item.strip() for item in value.split(",") if item.strip()]
    return keys or None


def build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер аргументов."""
    parser = argparse.ArgumentParser(
        description="Сгенерировать rules YAML из sentiment group stats."
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
        help="Колонка движения, для которой ранее построен group stats XLSX.",
    )
    return parser


def main() -> int:
    """Запускает генерацию rules YAML и печатает краткую сводку."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_sentiment_research_config(args.settings)
        target_column = args.target_column or config.target_column
        model_keys = config.selected_model_keys(parse_model_keys(args.models))

        print("Генерация rules recommendation.")
        print(f"Настройки: {args.settings}")
        print(f"Символ: {config.symbol}")
        print(f"Модели: {', '.join(model_keys)}")
        print(f"Target column: {target_column}")

        for model_key in model_keys:
            result = run_rules_recommendation_for_model(
                config,
                model_key,
                target_column=target_column,
            )
            print("")
            print(f"Модель: {model_key}")
            print(f"XLSX: {result.input_xlsx}")
            print(f"Rules: {result.output_yaml}")
    except (SettingsError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
