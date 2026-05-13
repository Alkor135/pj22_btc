r"""Последовательный оркестратор исследовательского pipeline BTCUSDT.

Порядок шагов:
1. download_mexc_klines.py
2. convert_5m_to_daily.py
3. create_news_markdown.py
4. create_sentiment_scores.py
5. create_sentiment_group_stats.py
6. create_rules_recommendation.py
7. run_sentiment_backtest.py
8. open_html_reports.py, если указан `--open-reports`

Примеры запуска:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --dry-run
.\.venv\Scripts\python.exe scripts\run_pipeline.py --models gemma3_12b,gpt-oss_20b
.\.venv\Scripts\python.exe scripts\run_pipeline.py --all-targets --open-reports
.\.venv\Scripts\python.exe scripts\run_pipeline.py --models gemma3_12b,qwen2.5_7b,gemma4_e2b,gemma4_e4b
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

from pj22_btc.pipeline_orchestrator import (  # noqa: E402
    PipelineStep,
    build_pipeline_steps,
    command_for_step,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер аргументов."""
    parser = argparse.ArgumentParser(
        description="Последовательно запустить полный BTCUSDT research pipeline."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Модели через запятую. Передается в sentiment/model-аналитические шаги.",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Прогнать аналитику для next_body и next_open_to_open.",
    )
    parser.add_argument(
        "--open-reports",
        action="store_true",
        help="После backtest открыть все HTML-отчеты в новом окне Chrome.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать план команд без запуска.",
    )
    return parser


def format_command(command: list[str]) -> str:
    """Форматирует команду для читаемого вывода."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def print_plan(steps: list[PipelineStep]) -> None:
    """Печатает последовательность команд pipeline."""
    print(f"Шагов pipeline: {len(steps)}")
    for index, step in enumerate(steps, start=1):
        command = command_for_step(Path(sys.executable), step)
        print(f"{index:02d}. {format_command(command)}")


def main() -> int:
    """Строит и запускает pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    steps = build_pipeline_steps(
        ROOT / "scripts",
        models=args.models,
        all_targets=args.all_targets,
        open_reports=args.open_reports,
    )

    print("План запуска:")
    print_plan(steps)
    if args.dry_run:
        return 0

    print("\nЗапуск pipeline.")
    results = run_pipeline(steps, python_executable=Path(sys.executable), cwd=ROOT)

    print("\nИтог:")
    for index, result in enumerate(results, start=1):
        marker = "OK" if result.ok else "FAIL"
        print(
            f"{index:02d}. [{marker}] {result.step.script.name} "
            f"({result.elapsed_seconds:.1f} с)"
        )

    if len(results) < len(steps):
        failed = results[-1]
        print(
            f"\nPipeline остановлен на шаге {len(results)}: "
            f"{failed.step.script.name}, код={failed.returncode}",
            file=sys.stderr,
        )
        return failed.returncode or 1

    if any(not result.ok for result in results):
        failed = next(result for result in results if not result.ok)
        print(
            f"\nPipeline завершился с ошибкой: {failed.step.script.name}, "
            f"код={failed.returncode}",
            file=sys.stderr,
        )
        return failed.returncode or 1

    print("\nPipeline завершен успешно.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
