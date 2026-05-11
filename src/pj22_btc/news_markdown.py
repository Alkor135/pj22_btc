r"""Генерация markdown-файлов с BTC/crypto-заголовками под дневные BTC-сессии.

Модуль читает дневные свечи из `daily_klines` и RSS-новости из файлов
`rss_news_*.db`. Для каждой дневной свечи используется готовый интервал
`[session_start_msk; session_end_msk)`, где граница `21:00 MSK` относится уже
к следующей сессии.

Обычно модуль запускается через CLI-скрипт:

```powershell
.\.venv\Scripts\python.exe scripts\create_news_markdown.py
```
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Pattern

from pj22_btc.daily_converter import DAILY_INTERVAL, moscow_tz
from pj22_btc.mexc_downloader import SettingsError, _parse_simple_yaml


@dataclass(frozen=True)
class NewsMarkdownConfig:
    """Настройки генерации markdown-файлов с новостями."""

    symbol: str
    daily_db: Path
    news_db_dir: Path
    markdown_dir: Path
    title_regex: str


@dataclass(frozen=True)
class DailyNewsSession:
    """Один дневной интервал свечи, к которому привязываются новости."""

    session_date: str
    start_msk: datetime
    end_msk: datetime


@dataclass(frozen=True)
class NewsTitleRow:
    """Один заголовок новости после regex-фильтра."""

    loaded_at: datetime
    provider: str
    title: str
    source_db: str


@dataclass(frozen=True)
class NewsMarkdownSummary:
    """Краткая сводка одного запуска генератора markdown."""

    symbol: str
    daily_db: Path
    news_db_dir: Path
    markdown_dir: Path
    sessions_read: int
    news_rows_read: int
    filtered_news_rows: int
    unique_news_rows: int
    files_deleted: int
    files_written: int
    titles_written: int
    latest_markdown_file: Path | None


def create_news_markdown_files(
    *,
    daily_db: Path,
    news_db_dir: Path,
    markdown_dir: Path,
    symbol: str,
    title_regex: str,
) -> NewsMarkdownSummary:
    """Создает markdown-файлы с crypto-заголовками для дневных BTC-сессий."""
    compiled = re.compile(title_regex)
    sessions = read_daily_sessions(Path(daily_db), symbol)
    news_rows, news_rows_read, filtered_rows = read_filtered_news_rows(
        Path(news_db_dir),
        compiled,
    )

    markdown_path = Path(markdown_dir)
    markdown_path.mkdir(parents=True, exist_ok=True)
    files_deleted = delete_latest_markdown_file(markdown_path)

    files_written = 0
    titles_written = 0
    latest_markdown_file: Path | None = None
    for session in sessions:
        session_rows = [
            row
            for row in news_rows
            if session.start_msk <= row.loaded_at < session.end_msk
        ]
        if not session_rows:
            continue

        output_file = markdown_path / f"{session.session_date}.md"
        if output_file.exists():
            continue

        content = _markdown_content(session_rows)
        output_file.write_text(content, encoding="utf-8")
        files_written += 1
        titles_written += len(session_rows)
        latest_markdown_file = output_file

    return NewsMarkdownSummary(
        symbol=symbol,
        daily_db=Path(daily_db),
        news_db_dir=Path(news_db_dir),
        markdown_dir=markdown_path,
        sessions_read=len(sessions),
        news_rows_read=news_rows_read,
        filtered_news_rows=filtered_rows,
        unique_news_rows=len(news_rows),
        files_deleted=files_deleted,
        files_written=files_written,
        titles_written=titles_written,
        latest_markdown_file=latest_markdown_file,
    )


def delete_latest_markdown_file(markdown_dir: Path) -> int:
    """Удаляет самый поздний markdown-файл с именем `YYYY-MM-DD.md`."""
    dated_files: list[tuple[date, Path]] = []
    for path in Path(markdown_dir).glob("*.md"):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        dated_files.append((file_date, path))

    if not dated_files:
        return 0

    _, latest_path = max(dated_files, key=lambda item: item[0])
    latest_path.unlink()
    return 1


def read_daily_sessions(daily_db: Path, symbol: str) -> list[DailyNewsSession]:
    """Читает дневные интервалы BTCUSDT из `daily_klines`."""
    with closing(sqlite3.connect(daily_db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT session_date, session_start_msk, session_end_msk
            FROM daily_klines
            WHERE symbol = ? AND interval = ? AND is_complete = 1
            ORDER BY session_start_ms
            """,
            (symbol, DAILY_INTERVAL),
        ).fetchall()

    return [
        DailyNewsSession(
            session_date=str(row["session_date"]),
            start_msk=_parse_msk_datetime(str(row["session_start_msk"])),
            end_msk=_parse_msk_datetime(str(row["session_end_msk"])),
        )
        for row in rows
    ]


def read_filtered_news_rows(
    news_db_dir: Path,
    title_pattern: Pattern[str],
) -> tuple[list[NewsTitleRow], int, int]:
    """Читает RSS DB-файлы и возвращает уникальные заголовки, прошедшие regex."""
    rows: list[NewsTitleRow] = []
    total_rows = 0
    matched_rows = 0
    seen: set[tuple[str, str, str]] = set()

    for db_path in sorted(Path(news_db_dir).glob("rss_news_*.db")):
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if not _has_table(conn, "news"):
                continue
            db_rows = conn.execute(
                """
                SELECT loaded_at, date, title, provider
                FROM news
                ORDER BY loaded_at, provider, title
                """
            ).fetchall()

        total_rows += len(db_rows)
        for row in db_rows:
            title = str(row["title"] or "").strip()
            if not title or not title_pattern.search(title):
                continue

            matched_rows += 1
            provider = str(row["provider"] or "").strip()
            loaded_at_raw = str(row["loaded_at"] or "").strip()
            unique_key = (loaded_at_raw, provider, title.casefold())
            if unique_key in seen:
                continue

            seen.add(unique_key)
            rows.append(
                NewsTitleRow(
                    loaded_at=_parse_msk_datetime(loaded_at_raw),
                    provider=provider,
                    title=title,
                    source_db=db_path.name,
                )
            )

    sorted_rows = sorted(rows, key=lambda item: (item.loaded_at, item.provider, item.title))
    return sorted_rows, total_rows, matched_rows


def load_news_markdown_config(path: Path) -> NewsMarkdownConfig:
    """Читает настройки генерации markdown из `settings.yaml`."""
    settings_path = Path(path)
    raw = _parse_simple_yaml(settings_path)
    mexc = _section(raw, "mexc")
    daily = _section(raw, "daily")
    news = _section(raw, "news")

    return NewsMarkdownConfig(
        symbol=str(_required(mexc, "symbol")).upper(),
        daily_db=_resolve_settings_path(settings_path, _required(daily, "sqlite_path")),
        news_db_dir=_resolve_settings_path(settings_path, _required(news, "db_dir")),
        markdown_dir=_resolve_settings_path(settings_path, _required(news, "markdown_dir")),
        title_regex=str(_required(news, "crypto_title_regex")),
    )


def _markdown_content(rows: list[NewsTitleRow]) -> str:
    """Формирует содержимое markdown: заголовки через пустую строку."""
    return "\n\n".join(row.title for row in rows) + "\n"


def _parse_msk_datetime(value: str) -> datetime:
    """Парсит timestamp новости/сессии и возвращает aware datetime в Москве."""
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    tz = moscow_tz()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def _has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    """Проверяет, существует ли таблица в SQLite-соединении."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _section(settings: dict[str, Any], key: str) -> dict[str, Any]:
    """Возвращает вложенную секцию настроек и проверяет, что это mapping."""
    value = settings.get(key, {})
    if not isinstance(value, dict):
        raise SettingsError(f"Секция settings.yaml {key!r} должна быть словарем")
    return value


def _required(settings: dict[str, Any], key: str) -> Any:
    """Возвращает обязательное значение из настроек или выбрасывает `SettingsError`."""
    if key not in settings:
        raise SettingsError(f"В settings.yaml отсутствует обязательный ключ {key!r}")
    return settings[key]


def _resolve_settings_path(settings_path: Path, value: Any) -> Path:
    """Преобразует путь из настроек в абсолютный путь относительно `settings.yaml`."""
    resolved = Path(str(value))
    if not resolved.is_absolute():
        resolved = settings_path.parent / resolved
    return resolved
