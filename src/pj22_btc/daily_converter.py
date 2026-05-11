r"""Конвертация пятиминутных свечей MEXC в дневные свечи по московской сессии.

Модуль читает 5m-свечи из годовых SQLite DB, переводит UTC timestamps в
московское время `MSK` (`UTC+03:00`) и собирает дневную свечу по правилу:
`[21:00 предыдущего дня; 21:00 текущего дня)`.

Свеча с open time ровно `21:00 MSK` не входит в завершающуюся дневную свечу,
а относится к следующей дневной сессии.

Обычно модуль запускается через CLI-скрипт:

```powershell
.\.venv\Scripts\python.exe scripts\convert_5m_to_daily.py
```

Пример использования из Python-кода:

```python
from pathlib import Path

from pj22_btc.daily_converter import convert_5m_to_daily

summary = convert_5m_to_daily(
    source_dir=Path("data/mexc/klines/BTCUSDT/5m"),
    output_dir=Path("data/mexc/klines/BTCUSDT/1d_msk"),
    symbol="BTCUSDT",
)
print(summary.inserted_rows)
```
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from pj22_btc.mexc_downloader import SettingsError, _parse_simple_yaml


MSK = timezone(timedelta(hours=3), "MSK")
FIVE_MINUTES_MS = 300_000
EXPECTED_5M_CANDLES_PER_DAY = 288
DAILY_INTERVAL = "1d_msk"


@dataclass(frozen=True)
class DailyConverterConfig:
    """Настройки конвертера 5m-свечей в дневные свечи."""

    symbol: str
    source_dir: Path
    output_dir: Path
    include_incomplete: bool = False


@dataclass(frozen=True)
class DailyKline:
    """Одна дневная свеча, собранная по московской торговой сессии."""

    symbol: str
    interval: str
    session_date: str
    session_start_ms: int
    session_end_ms: int
    session_start_msk: str
    session_end_msk: str
    open_price: str
    high_price: str
    low_price: str
    close_price: str
    volume: str
    quote_asset_volume: str
    candle_count: int
    is_complete: bool


@dataclass(frozen=True)
class DailyConversionSummary:
    """Краткая сводка результата одного запуска конвертера."""

    symbol: str
    source_dir: Path
    output_dir: Path
    source_rows: int
    daily_rows: int
    inserted_rows: int
    skipped_incomplete_sessions: int


def session_date_for_open_time(open_time_ms: int) -> date:
    """Возвращает дату дневной сессии для 5m-свечи по границе `21:00 MSK`."""
    local_dt = datetime.fromtimestamp(open_time_ms / 1000, UTC).astimezone(MSK)
    boundary = time(21, 0)
    if local_dt.time() >= boundary:
        return local_dt.date() + timedelta(days=1)
    return local_dt.date()


def session_bounds_ms(session_day: date) -> tuple[int, int, str, str]:
    """Возвращает UTC millisecond bounds и MSK-строки для дневной сессии."""
    session_start = datetime.combine(session_day - timedelta(days=1), time(21, 0), MSK)
    session_end = datetime.combine(session_day, time(21, 0), MSK)
    start_ms = int(session_start.astimezone(UTC).timestamp() * 1000)
    end_ms = int(session_end.astimezone(UTC).timestamp() * 1000)
    return start_ms, end_ms, session_start.isoformat(), session_end.isoformat()


def aggregate_daily_klines(
    rows: list[dict[str, Any]],
    *,
    include_incomplete: bool = False,
) -> list[DailyKline]:
    """Группирует 5m-свечи в дневные свечи по московской сессии."""
    grouped: dict[date, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: int(item["open_time_ms"])):
        grouped.setdefault(session_date_for_open_time(int(row["open_time_ms"])), []).append(row)

    daily: list[DailyKline] = []
    for session_day in sorted(grouped):
        session_rows = sorted(grouped[session_day], key=lambda item: int(item["open_time_ms"]))
        start_ms, end_ms, start_msk, end_msk = session_bounds_ms(session_day)
        is_complete = _is_complete_session(session_rows, start_ms, end_ms)
        if not is_complete and not include_incomplete:
            continue

        daily.append(
            DailyKline(
                symbol=str(session_rows[0]["symbol"]),
                interval=DAILY_INTERVAL,
                session_date=session_day.isoformat(),
                session_start_ms=start_ms,
                session_end_ms=end_ms,
                session_start_msk=start_msk,
                session_end_msk=end_msk,
                open_price=str(session_rows[0]["open_price"]),
                high_price=_decimal_max(session_rows, "high_price"),
                low_price=_decimal_min(session_rows, "low_price"),
                close_price=str(session_rows[-1]["close_price"]),
                volume=_decimal_sum(session_rows, "volume"),
                quote_asset_volume=_decimal_sum(session_rows, "quote_asset_volume"),
                candle_count=len(session_rows),
                is_complete=is_complete,
            )
        )
    return daily


def load_5m_klines(source_dir: Path, symbol: str) -> list[dict[str, Any]]:
    """Читает все 5m-свечи указанного символа из годовых SQLite DB-файлов."""
    rows: list[dict[str, Any]] = []
    for db_path in sorted(Path(source_dir).glob("*.db")):
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if not _has_table(conn, "klines"):
                continue
            query_rows = conn.execute(
                """
                SELECT
                    symbol,
                    interval,
                    open_time_ms,
                    open_time_utc,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    close_time_ms,
                    close_time_utc,
                    quote_asset_volume
                FROM klines
                WHERE symbol = ? AND interval = '5m'
                ORDER BY open_time_ms
                """,
                (symbol,),
            ).fetchall()
            rows.extend(dict(row) for row in query_rows)
    return sorted(rows, key=lambda item: int(item["open_time_ms"]))


class DailySQLiteKlineStore:
    """SQLite-хранилище дневных свечей с разбиением DB-файлов по годам."""

    def __init__(self, root_dir: Path) -> None:
        """Создает хранилище дневных свечей и директорию для DB-файлов."""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def db_path_for_session_date(self, session_date: str) -> Path:
        """Возвращает путь к годовой DB по дате дневной сессии."""
        return self.root_dir / f"{session_date[:4]}.db"

    def insert_daily_klines(self, candles: list[DailyKline]) -> int:
        """Записывает дневные свечи и возвращает число новых или обновленных строк."""
        changed = 0
        grouped: dict[Path, list[DailyKline]] = {}
        for candle in candles:
            grouped.setdefault(self.db_path_for_session_date(candle.session_date), []).append(candle)

        for db_path, db_candles in grouped.items():
            with closing(sqlite3.connect(db_path)) as conn:
                self._ensure_schema(conn)
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT INTO daily_klines (
                        symbol,
                        interval,
                        session_date,
                        session_start_ms,
                        session_end_ms,
                        session_start_msk,
                        session_end_msk,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume,
                        quote_asset_volume,
                        candle_count,
                        is_complete
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, interval, session_date) DO UPDATE SET
                        session_start_ms = excluded.session_start_ms,
                        session_end_ms = excluded.session_end_ms,
                        session_start_msk = excluded.session_start_msk,
                        session_end_msk = excluded.session_end_msk,
                        open_price = excluded.open_price,
                        high_price = excluded.high_price,
                        low_price = excluded.low_price,
                        close_price = excluded.close_price,
                        volume = excluded.volume,
                        quote_asset_volume = excluded.quote_asset_volume,
                        candle_count = excluded.candle_count,
                        is_complete = excluded.is_complete
                    """,
                    [
                        (
                            candle.symbol,
                            candle.interval,
                            candle.session_date,
                            candle.session_start_ms,
                            candle.session_end_ms,
                            candle.session_start_msk,
                            candle.session_end_msk,
                            candle.open_price,
                            candle.high_price,
                            candle.low_price,
                            candle.close_price,
                            candle.volume,
                            candle.quote_asset_volume,
                            candle.candle_count,
                            int(candle.is_complete),
                        )
                        for candle in db_candles
                    ],
                )
                changed += conn.total_changes - before
                conn.commit()
        return changed

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        """Создает таблицу дневных свечей и индекс, если их еще нет."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_klines (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                session_date TEXT NOT NULL,
                session_start_ms INTEGER NOT NULL,
                session_end_ms INTEGER NOT NULL,
                session_start_msk TEXT NOT NULL,
                session_end_msk TEXT NOT NULL,
                open_price TEXT NOT NULL,
                high_price TEXT NOT NULL,
                low_price TEXT NOT NULL,
                close_price TEXT NOT NULL,
                volume TEXT NOT NULL,
                quote_asset_volume TEXT NOT NULL,
                candle_count INTEGER NOT NULL,
                is_complete INTEGER NOT NULL,
                PRIMARY KEY (symbol, interval, session_date)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_daily_klines_symbol_interval_date
            ON daily_klines (symbol, interval, session_date)
            """
        )


def convert_5m_to_daily(
    *,
    source_dir: Path,
    output_dir: Path,
    symbol: str,
    include_incomplete: bool = False,
) -> DailyConversionSummary:
    """Читает 5m DB, собирает дневные свечи и записывает их в daily DB."""
    source_rows = load_5m_klines(source_dir, symbol)
    all_daily = aggregate_daily_klines(source_rows, include_incomplete=True)
    daily = [candle for candle in all_daily if candle.is_complete or include_incomplete]
    skipped = len(all_daily) - len(daily)
    store = DailySQLiteKlineStore(output_dir)
    inserted = store.insert_daily_klines(daily)
    return DailyConversionSummary(
        symbol=symbol,
        source_dir=source_dir,
        output_dir=output_dir,
        source_rows=len(source_rows),
        daily_rows=len(daily),
        inserted_rows=inserted,
        skipped_incomplete_sessions=skipped,
    )


def load_daily_converter_config(path: Path) -> DailyConverterConfig:
    """Читает настройки конвертера дневных свечей из `settings.yaml`."""
    settings_path = Path(path)
    raw = _parse_simple_yaml(settings_path)
    mexc = _section(raw, "mexc")
    storage = _section(raw, "storage")
    daily = _section(raw, "daily")
    source_dir = _resolve_settings_path(settings_path, _required(storage, "sqlite_dir"))
    output_dir = _resolve_settings_path(settings_path, _required(daily, "sqlite_dir"))
    return DailyConverterConfig(
        symbol=str(_required(mexc, "symbol")).upper(),
        source_dir=source_dir,
        output_dir=output_dir,
        include_incomplete=bool(daily.get("include_incomplete", False)),
    )


def _is_complete_session(rows: list[dict[str, Any]], start_ms: int, end_ms: int) -> bool:
    """Проверяет, содержит ли группа полный набор 5m-свечей для дневной сессии."""
    if len(rows) != EXPECTED_5M_CANDLES_PER_DAY:
        return False
    first_open = int(rows[0]["open_time_ms"])
    last_open = int(rows[-1]["open_time_ms"])
    return first_open == start_ms and last_open == end_ms - FIVE_MINUTES_MS


def _decimal_sum(rows: list[dict[str, Any]], key: str) -> str:
    """Суммирует decimal-значения из строк и возвращает строковое представление."""
    return str(sum(Decimal(str(row[key])) for row in rows))


def _decimal_max(rows: list[dict[str, Any]], key: str) -> str:
    """Находит максимум decimal-значений из строк и возвращает исходную строку."""
    return str(max(Decimal(str(row[key])) for row in rows))


def _decimal_min(rows: list[dict[str, Any]], key: str) -> str:
    """Находит минимум decimal-значений из строк и возвращает исходную строку."""
    return str(min(Decimal(str(row[key])) for row in rows))


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
