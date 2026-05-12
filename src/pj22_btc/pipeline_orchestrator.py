"""Планирование и запуск последовательного исследовательского pipeline."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TARGET_COLUMN = "next_body"
ALL_TARGET_COLUMNS = ("next_body", "next_open_to_open")


@dataclass(frozen=True)
class PipelineStep:
    """Один CLI-шаг pipeline."""

    script: Path
    args: list[str]


@dataclass(frozen=True)
class PipelineStepResult:
    """Результат запуска одного шага."""

    step: PipelineStep
    returncode: int
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        """Возвращает True, если шаг завершился успешно."""
        return self.returncode == 0


def build_pipeline_steps(
    scripts_dir: Path,
    *,
    models: str | None = None,
    all_targets: bool = False,
    open_reports: bool = False,
) -> list[PipelineStep]:
    """Возвращает шаги pipeline в подтвержденном порядке."""
    scripts = Path(scripts_dir)
    model_args = ["--models", models] if models else []
    targets = list(ALL_TARGET_COLUMNS if all_targets else (DEFAULT_TARGET_COLUMN,))

    steps = [
        PipelineStep(scripts / "download_mexc_klines.py", []),
        PipelineStep(scripts / "convert_5m_to_daily.py", []),
        PipelineStep(scripts / "create_news_markdown.py", []),
        PipelineStep(scripts / "create_sentiment_scores.py", list(model_args)),
    ]

    for target_column in targets:
        target_args = [*model_args, "--target-column", target_column]
        steps.extend(
            [
                PipelineStep(scripts / "create_sentiment_group_stats.py", target_args),
                PipelineStep(scripts / "create_rules_recommendation.py", target_args),
                PipelineStep(scripts / "run_sentiment_backtest.py", target_args),
            ]
        )

    if open_reports:
        steps.append(PipelineStep(scripts / "open_html_reports.py", []))

    return steps


def command_for_step(python_executable: Path, step: PipelineStep) -> list[str]:
    """Возвращает команду запуска одного шага."""
    return [str(python_executable), str(step.script), *step.args]


def run_pipeline(
    steps: list[PipelineStep],
    *,
    python_executable: Path,
    cwd: Path,
) -> list[PipelineStepResult]:
    """Последовательно запускает шаги pipeline и останавливается на первой ошибке."""
    results: list[PipelineStepResult] = []
    for step in steps:
        started = time.monotonic()
        completed = subprocess.run(command_for_step(python_executable, step), cwd=str(cwd))
        elapsed = time.monotonic() - started
        result = PipelineStepResult(
            step=step,
            returncode=completed.returncode,
            elapsed_seconds=elapsed,
        )
        results.append(result)
        if not result.ok:
            break
    return results
