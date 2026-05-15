r"""Запустить дневной walk-forward backtest sentiment-стратегии.

Примеры запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\run_walk_forward.py
.\.venv\Scripts\python.exe scripts\run_walk_forward.py --models gemma3_12b
.\.venv\Scripts\python.exe scripts\run_walk_forward.py --target-column next_open_to_open
```
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pj22_btc.mexc_downloader import SettingsError  # noqa: E402
from pj22_btc.sentiment_research import VALID_TARGET_COLUMNS  # noqa: E402
from pj22_btc.walk_forward.core import (  # noqa: E402
    error_summary,
    load_walk_forward_config,
    run_walk_forward_for_model,
    save_global_summary,
    save_model_outputs,
)
from pj22_btc.walk_forward.report import build_report  # noqa: E402


def parse_model_keys(value: str | None) -> list[str] | None:
    """Parse comma-separated sentiment model keys."""
    if value is None:
        return None
    keys = [item.strip() for item in value.split(",") if item.strip()]
    return keys or None


def parse_cli_date(value: str | None) -> date | None:
    """Parse YYYY-MM-DD CLI dates."""
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise argparse.ArgumentTypeError(f"Некорректная дата: {value!r}")
    return parsed.date()


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Запустить walk-forward backtest для sentiment-моделей."
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
        help="Колонка движения для P/L. По умолчанию walk_forward.target_column.",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=None,
        help="Количество контрактов/единиц на сделку. По умолчанию walk_forward.quantity.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_cli_date,
        default=None,
        help="Дата начала тестовых дней YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_cli_date,
        default=None,
        help="Дата окончания тестовых дней YYYY-MM-DD.",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=None,
        help="Размер rolling train-окна в месяцах.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Папка результатов. По умолчанию walk_forward.output_dir.",
    )
    parser.add_argument(
        "--save-daily-artifacts",
        action="store_true",
        default=None,
        help="Сохранять daily group_stats.xlsx и rules.yaml.",
    )
    parser.add_argument(
        "--no-save-daily-artifacts",
        action="store_false",
        dest="save_daily_artifacts",
        help="Не сохранять daily group_stats.xlsx и rules.yaml.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=None,
        help="Минимум строк в train-окне.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Остановить запуск на первой ошибке модели.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Не собирать итоговый HTML/XLSX отчет после прогона.",
    )
    return parser


def main() -> int:
    """Run walk-forward and print short progress."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_walk_forward_config(args.settings)
        target_column = args.target_column or config.target_column
        output_dir = args.output_dir or config.output_dir
        model_keys = config.selected_model_keys(parse_model_keys(args.models))
        keep_going = config.keep_going and not args.stop_on_error
        save_daily_artifacts = (
            config.save_daily_artifacts
            if args.save_daily_artifacts is None
            else args.save_daily_artifacts
        )

        print("Запуск walk-forward backtest.")
        print(f"Настройки: {args.settings}")
        print(f"Символ: {config.symbol}")
        print(f"Модели: {', '.join(model_keys)}")
        print(f"Target column: {target_column}")
        print(f"Output: {output_dir}")

        all_daily_summaries: list[dict] = []
        model_summaries: list[dict] = []
        errors = 0

        for model_key in model_keys:
            try:
                result = run_walk_forward_for_model(
                    config,
                    model_key,
                    target_column=target_column,
                    quantity=args.quantity,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    train_months=args.train_months,
                    min_train_rows=args.min_train_rows,
                )
                save_model_outputs(
                    output_dir=output_dir,
                    symbol=config.symbol,
                    model_key=model_key,
                    target_column=target_column,
                    result=result,
                    save_daily_artifacts=save_daily_artifacts,
                )
                all_daily_summaries.extend(result.daily_summaries)
                model_summaries.append(result.model_summary)
                print(
                    f"[OK] {model_key}: days={result.model_summary['days']} "
                    f"trades={result.model_summary['trades']} "
                    f"pnl={result.model_summary['total_pnl']:.2f}"
                )
            except Exception as exc:
                errors += 1
                all_daily_summaries.append(
                    error_summary(
                        symbol=config.symbol,
                        model_key=model_key,
                        target_column=target_column,
                        error=exc,
                    )
                )
                model_summaries.append(_model_error_summary(config.symbol, model_key, target_column, exc))
                print(f"[ERROR] {model_key}: {exc}", file=sys.stderr)
                if not keep_going:
                    break

        save_global_summary(
            output_dir=output_dir,
            target_column=target_column,
            daily_summaries=all_daily_summaries,
            model_summaries=model_summaries,
        )
        print(f"Summary: {output_dir / f'model_summary_{target_column}.csv'}")

        if not args.no_report:
            html_path, xlsx_path = build_report(output_dir, target_column=target_column)
            print(f"HTML: {html_path}")
            print(f"Excel: {xlsx_path}")

        return 1 if errors else 0
    except (SettingsError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


def _model_error_summary(
    symbol: str,
    model_key: str,
    target_column: str,
    error: Exception,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "model_key": model_key,
        "target_column": target_column,
        "status": "error",
        "days": 0,
        "ok_days": 0,
        "skipped_days": 0,
        "error_days": 1,
        "trades": 0,
        "total_pnl": 0.0,
        "winrate": 0.0,
        "max_drawdown": 0.0,
        "error": str(error),
    }


if __name__ == "__main__":
    raise SystemExit(main())
