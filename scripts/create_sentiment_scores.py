r"""Расчет sentiment PKL для одной или нескольких Ollama-моделей.

Скрипт читает `settings.yaml`, берет markdown-файлы из `news.markdown_dir`,
прогоняет выбранные модели из `sentiment.models` и сохраняет результат в:

```text
data/sentiment/<symbol>/<model_key>/sentiment_scores.pkl
```

Пример запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\create_sentiment_scores.py
.\.venv\Scripts\python.exe scripts\create_sentiment_scores.py --models gemma3_12b,gpt-oss_20b
.\.venv\Scripts\python.exe scripts\create_sentiment_scores.py --no-use-cache
```
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pj22_btc.mexc_downloader import SettingsError
from pj22_btc.sentiment_analysis import (
    get_ollama_processor_status,
    load_sentiment_analysis_config,
    run_sentiment_analysis,
)


def format_duration(seconds: float) -> str:
    """Форматирует длительность модели компактно для консоли."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes, rem_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} мин {rem_seconds:02d} с"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{hours} ч {rem_minutes:02d} мин {rem_seconds:02d} с"


def parse_model_keys(value: str | None) -> list[str] | None:
    """Разбирает comma-separated список model_key из CLI."""
    if value is None:
        return None
    keys = [item.strip() for item in value.split(",") if item.strip()]
    return keys or None


def build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер аргументов."""
    parser = argparse.ArgumentParser(
        description="Создать sentiment_scores.pkl для выбранных моделей из settings.yaml."
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
        help="Модели через запятую. Если не задано, запускаются sentiment.models с enabled=true.",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Использовать PKL-кэш. По умолчанию берется из settings.yaml.",
    )
    return parser


class TqdmProgressReporter:
    """Рисует progress bar с ETA для долгого Ollama-прогона."""

    def __init__(self) -> None:
        self.bar: tqdm | None = None

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def __call__(self, event: dict[str, Any]) -> None:
        event_name = event["event"]
        if event_name == "model_start":
            self.close()
            tqdm.write("")
            tqdm.write(f"=== Модель: {event['model_key']} ({event['ollama_model']}) ===")
            tqdm.write(f"Markdown-файлов: {event['file_count']}")
            tqdm.write(f"PKL: {event['output_pkl']}")
            return

        if event_name == "pass_start":
            self.close()
            self.bar = tqdm(
                total=int(event["file_count"]),
                desc=(
                    f"[{event['model_key']}] проход "
                    f"{int(event['retry_pass']) + 1}/{int(event['retry_limit']) + 1}"
                ),
                unit="file",
                dynamic_ncols=True,
            )
            return

        if event_name == "file_start":
            if self.bar is not None:
                self.bar.set_postfix_str(
                    f"обработка {Path(event['file_path']).name} | processor={event['processor_status']}",
                    refresh=True,
                )
            return

        if event_name == "file_skipped":
            if self.bar is not None:
                self.bar.set_postfix_str(f"кэш {Path(event['file_path']).name}", refresh=False)
                self.bar.update(1)
            return

        if event_name == "file_done":
            if self.bar is not None:
                status = "ошибка" if event.get("failed") else f"sentiment={event.get('sentiment')}"
                self.bar.set_postfix_str(
                    f"{Path(event['file_path']).name} | {status} | tokens={event.get('prompt_tokens')}",
                    refresh=False,
                )
                self.bar.update(1)
            return

        if event_name == "checkpoint_saved":
            tqdm.write(f"Чекпоинт PKL: {event['rows_saved']} строк -> {event['output_pkl']}")
            return

        if event_name == "pass_done":
            self.close()
            if int(event["failed_files"]) > 0 and int(event["retry_pass"]) < int(event["retry_limit"]):
                tqdm.write(
                    "Повторный проход: "
                    f"не распарсилось/ошибок={event['failed_files']}, "
                    "пересчитываю проблемные файлы."
                )
            return

        if event_name == "model_done":
            self.close()
            tqdm.write(
                "Итог модели: "
                f"обработано={event['processed_files']}, "
                f"кэш={event['skipped_files']}, "
                f"ошибок={event['failed_files']}, "
                f"строк={event['rows_saved']}, "
                f"время={format_duration(float(event['elapsed_seconds']))}"
            )


def main() -> int:
    """Загружает настройки, запускает модели и печатает краткую сводку."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_sentiment_analysis_config(args.settings)
        if args.use_cache is not None:
            config = replace(config, use_cache=bool(args.use_cache))
        model_keys = parse_model_keys(args.models)
        selected_models = model_keys or [model.key for model in config.enabled_models()]
        print("Запуск расчета sentiment-оценок.", flush=True)
        print(f"Настройки: {args.settings}", flush=True)
        print(f"Символ: {config.symbol}", flush=True)
        print(f"Markdown: {config.markdown_dir}", flush=True)
        print(f"Daily DB: {config.daily_db}", flush=True)
        print(f"Модели: {', '.join(selected_models)}", flush=True)
        print(f"Кэш: {config.use_cache}", flush=True)
        reporter = TqdmProgressReporter()
        try:
            summaries = run_sentiment_analysis(
                config,
                model_keys=model_keys,
                progress=reporter,
                processor_status=get_ollama_processor_status,
            )
        finally:
            reporter.close()
    except (SettingsError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print("Расчет sentiment-оценок завершен.")
    for summary in summaries:
        print("")
        print(f"Модель: {summary.model_key} ({summary.ollama_model})")
        print(f"Markdown-файлов: {summary.markdown_files}")
        print(f"Обработано: {summary.processed_files}")
        print(f"Пропущено по кэшу: {summary.skipped_files}")
        print(f"Ошибок обработки: {summary.failed_files}")
        print(f"Строк сохранено: {summary.rows_saved}")
        print(f"Время модели: {format_duration(summary.elapsed_seconds)}")
        print(f"PKL: {summary.output_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
