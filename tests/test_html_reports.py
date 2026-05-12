import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pj22_btc.html_reports import (  # noqa: E402
    build_chrome_command,
    collect_html_reports,
)


class HtmlReportsTests(unittest.TestCase):
    def test_collect_html_reports_finds_all_html_recursively_in_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gemma3_12b" / "plots").mkdir(parents=True)
            (root / "gpt-oss_20b" / "plots").mkdir(parents=True)
            (root / "gemma3_12b" / "plots" / "b.html").write_text("b", encoding="utf-8")
            (root / "gemma3_12b" / "plots" / "a.html").write_text("a", encoding="utf-8")
            (root / "gpt-oss_20b" / "plots" / "report.HTML").write_text(
                "report",
                encoding="utf-8",
            )
            (root / "gpt-oss_20b" / "plots" / "notes.txt").write_text(
                "ignore",
                encoding="utf-8",
            )

            reports = collect_html_reports(root)

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in reports],
                [
                    "gemma3_12b/plots/a.html",
                    "gemma3_12b/plots/b.html",
                    "gpt-oss_20b/plots/report.HTML",
                ],
            )

    def test_collect_html_reports_returns_empty_list_for_missing_root(self) -> None:
        self.assertEqual(collect_html_reports(Path("missing-report-root")), [])

    def test_build_chrome_command_opens_reports_in_one_new_window(self) -> None:
        chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        reports = [Path(r"C:\reports\one.html"), Path(r"C:\reports\two.html")]

        command = build_chrome_command(chrome, reports)

        self.assertEqual(
            command,
            [
                str(chrome),
                "--new-window",
                str(reports[0]),
                str(reports[1]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
