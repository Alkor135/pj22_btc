"""Поиск и открытие локальных HTML-отчетов проекта."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable


DEFAULT_CHROME_PATH = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def collect_html_reports(reports_root: Path) -> list[Path]:
    """Возвращает все HTML-файлы из reports_root рекурсивно, в стабильном порядке."""
    root = Path(reports_root)
    if not root.exists():
        return []
    reports = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".html"
    ]
    return sorted(reports, key=lambda path: path.relative_to(root).as_posix().casefold())


def build_chrome_command(chrome_path: Path, reports: list[Path]) -> list[str]:
    """Строит команду Chrome: одно новое окно, каждый HTML в отдельной вкладке."""
    return [str(chrome_path), "--new-window", *[str(path) for path in reports]]


def open_reports_in_chrome(
    chrome_path: Path,
    reports: list[Path],
    *,
    popen: Callable[[list[str]], object] = subprocess.Popen,
) -> None:
    """Открывает HTML-отчеты в новом окне Google Chrome."""
    chrome = Path(chrome_path)
    if not chrome.exists():
        raise FileNotFoundError(f"Google Chrome не найден: {chrome}")
    if not reports:
        return
    popen(build_chrome_command(chrome, reports))
