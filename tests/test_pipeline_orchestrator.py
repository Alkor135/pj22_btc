import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pj22_btc.pipeline_orchestrator import (  # noqa: E402
    build_pipeline_steps,
    command_for_step,
)


class PipelineOrchestratorTests(unittest.TestCase):
    def test_build_pipeline_steps_uses_confirmed_default_order(self) -> None:
        scripts_dir = Path("scripts")

        steps = build_pipeline_steps(scripts_dir)

        self.assertEqual(
            [(step.script.name, step.args) for step in steps],
            [
                ("download_mexc_klines.py", []),
                ("convert_5m_to_daily.py", []),
                ("create_news_markdown.py", []),
                ("create_sentiment_scores.py", []),
                ("create_sentiment_group_stats.py", ["--target-column", "next_body"]),
                ("create_rules_recommendation.py", ["--target-column", "next_body"]),
                ("run_sentiment_backtest.py", ["--target-column", "next_body"]),
            ],
        )

    def test_build_pipeline_steps_passes_models_to_model_steps_only(self) -> None:
        scripts_dir = Path("scripts")

        steps = build_pipeline_steps(scripts_dir, models="gemma3_12b,gpt-oss_20b")

        self.assertEqual(steps[0].args, [])
        self.assertEqual(steps[3].args, ["--models", "gemma3_12b,gpt-oss_20b"])
        self.assertEqual(
            steps[4].args,
            ["--models", "gemma3_12b,gpt-oss_20b", "--target-column", "next_body"],
        )

    def test_build_pipeline_steps_can_run_all_targets_and_open_reports(self) -> None:
        scripts_dir = Path("scripts")

        steps = build_pipeline_steps(scripts_dir, all_targets=True, open_reports=True)

        self.assertEqual(
            [(step.script.name, step.args) for step in steps[4:]],
            [
                ("create_sentiment_group_stats.py", ["--target-column", "next_body"]),
                ("create_rules_recommendation.py", ["--target-column", "next_body"]),
                ("run_sentiment_backtest.py", ["--target-column", "next_body"]),
                (
                    "create_sentiment_group_stats.py",
                    ["--target-column", "next_open_to_open"],
                ),
                (
                    "create_rules_recommendation.py",
                    ["--target-column", "next_open_to_open"],
                ),
                ("run_sentiment_backtest.py", ["--target-column", "next_open_to_open"]),
                ("open_html_reports.py", []),
            ],
        )

    def test_command_for_step_uses_current_python_and_script_path(self) -> None:
        step = build_pipeline_steps(Path("scripts"))[0]

        command = command_for_step(Path("python.exe"), step)

        self.assertEqual(command, ["python.exe", str(Path("scripts") / "download_mexc_klines.py")])


if __name__ == "__main__":
    unittest.main()
