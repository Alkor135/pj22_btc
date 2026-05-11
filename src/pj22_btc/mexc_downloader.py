r"""Библиотека загрузки пятиминутных свечей MEXC и записи в SQLite.

Этот модуль содержит переиспользуемую логику для:

- чтения настроек из `settings.yaml`;
- запроса свечей через публичный Spot API MEXC `GET /api/v3/klines`;
- сохранения свечей в SQLite 3 базы, разбитые по годам;
- докачки новых свечей после последней уже сохраненной записи.

Обычно этот файл не запускается напрямую. Для загрузки данных используется
CLI-скрипт `scripts/download_mexc_klines.py`.

Пример запуска из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\download_mexc_klines.py
```

Пример использования из Python-кода:

```python
from pathlib import Path

from pj22_btc.mexc_downloader import load_config, sync_klines

config = load_config(Path("settings.yaml"))
summary = sync_klines(config)
print(summary.inserted_rows)
```
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MEXC_KLINES_PATH = "/api/v3/klines"
SUPPORTED_INTERVALS_MS = {
    "5m": 300_000,
}


class SettingsError(ValueError):
    """Ошибка чтения или разбора `settings.yaml`."""


class MexcAPIError(RuntimeError):
    """Ошибка ответа или доступности публичного API MEXC."""


class MexcNoDataError(MexcAPIError):
    """Ошибка пустого ответа MEXC для диапазона, где ожидались свечи."""


@dataclass(frozen=True)
class DownloaderConfig:
    """Настройки загрузчика, прочитанные из `settings.yaml`."""

    symbol: str
    interval: str
    start_date: str
    sqlite_dir: Path
    base_url: str = "https://api.mexc.com"
    request_limit: int = 1000
    request_pause_seconds: float = 0.2
    request_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class Kline:
    """Одна свеча MEXC в нормализованном виде для записи в SQLite."""

    symbol: str
    interval: str
    open_time_ms: int
    open_price: str
    high_price: str
    low_price: str
    close_price: str
    volume: str
    close_time_ms: int
    quote_asset_volume: str

    @classmethod
    def from_mexc_row(cls, symbol: str, interval: str, row: list[Any]) -> Kline:
        """Преобразует одну строку ответа MEXC `/api/v3/klines` в объект `Kline`."""
        if len(row) < 8:
            raise ValueError(f"MEXC kline row must contain at least 8 fields, got {len(row)}")
        return cls(
            symbol=symbol,
            interval=interval,
            open_time_ms=int(row[0]),
            open_price=str(row[1]),
            high_price=str(row[2]),
            low_price=str(row[3]),
            close_price=str(row[4]),
            volume=str(row[5]),
            close_time_ms=int(row[6]),
            quote_asset_volume=str(row[7]),
        )

    @property
    def open_time_utc(self) -> str:
        """Возвращает время открытия свечи в ISO-формате UTC."""
        return datetime.fromtimestamp(self.open_time_ms / 1000, UTC).isoformat()

    @property
    def close_time_utc(self) -> str:
        """Возвращает время закрытия свечи в ISO-формате UTC."""
        return datetime.fromtimestamp(self.close_time_ms / 1000, UTC).isoformat()


@dataclass(frozen=True)
class DownloadSummary:
    """Краткая сводка результата одного запуска загрузчика."""

    symbol: str
    interval: str
    db_dir: Path
    start_ms: int
    end_ms: int
    first_requested_ms: int | None
    last_open_time_ms: int | None
    fetched_rows: int
    inserted_rows: int
    batches: int


def utc_ms(value: str) -> int:
    """Преобразует дату или дату-время в UTC timestamp в миллисекундах."""
    if len(value) == 10:
        parsed_date = date.fromisoformat(value)
        parsed = datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=UTC)
    else:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        else:
            parsed = parsed.astimezone(UTC)
    return int(parsed.timestamp() * 1000)


def interval_ms(interval: str) -> int:
    """Возвращает длительность поддерживаемого интервала свечи в миллисекундах."""
    try:
        return SUPPORTED_INTERVALS_MS[interval]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS_MS))
        raise ValueError(f"Unsupported interval {interval!r}. Supported intervals: {supported}") from exc


def year_key(open_time_ms: int) -> str:
    """Возвращает ключ года `YYYY` по времени открытия свечи."""
    return datetime.fromtimestamp(open_time_ms / 1000, UTC).strftime("%Y")


def closed_open_time_ms(now_ms: int, interval: str) -> int:
    """Возвращает open time последней полностью закрытой свечи для текущего времени."""
    step_ms = interval_ms(interval)
    return (now_ms // step_ms) * step_ms - step_ms


class YearlySQLiteKlineStore:
    """SQLite-хранилище, которое раскладывает свечи по годовым DB-файлам."""

    def __init__(self, root_dir: Path) -> None:
        """Создает хранилище и гарантирует существование корневой директории DB-файлов."""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def db_path_for_open_time(self, open_time_ms: int) -> Path:
        """Возвращает путь к годовой SQLite-базе для заданного open time свечи."""
        return self.root_dir / f"{year_key(open_time_ms)}.db"

    def insert_klines(self, klines: list[Kline]) -> int:
        """Записывает свечи в годовые SQLite-базы и возвращает число новых строк."""
        inserted = 0
        grouped: dict[Path, list[Kline]] = {}
        for kline in klines:
            grouped.setdefault(self.db_path_for_open_time(kline.open_time_ms), []).append(kline)

        for db_path, month_klines in grouped.items():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(db_path)) as conn:
                self._ensure_schema(conn)
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO klines (
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
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            kline.symbol,
                            kline.interval,
                            kline.open_time_ms,
                            kline.open_time_utc,
                            kline.open_price,
                            kline.high_price,
                            kline.low_price,
                            kline.close_price,
                            kline.volume,
                            kline.close_time_ms,
                            kline.close_time_utc,
                            kline.quote_asset_volume,
                        )
                        for kline in month_klines
                    ],
                )
                inserted += conn.total_changes - before
                conn.commit()
        return inserted

    def latest_open_time_ms(self, symbol: str, interval: str) -> int | None:
        """Ищет максимальный `open_time_ms` по всем годовым DB-файлам хранилища."""
        latest: int | None = None
        for db_path in sorted(self.root_dir.glob("*.db")):
            with closing(sqlite3.connect(db_path)) as conn:
                if not self._has_klines_table(conn):
                    continue
                row = conn.execute(
                    """
                    SELECT MAX(open_time_ms)
                    FROM klines
                    WHERE symbol = ? AND interval = ?
                    """,
                    (symbol, interval),
                ).fetchone()
            if row and row[0] is not None:
                latest = max(latest, int(row[0])) if latest is not None else int(row[0])
        return latest

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        """Создает таблицу и индекс для свечей, если их еще нет в SQLite-базе."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS klines (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time_ms INTEGER NOT NULL,
                open_time_utc TEXT NOT NULL,
                open_price TEXT NOT NULL,
                high_price TEXT NOT NULL,
                low_price TEXT NOT NULL,
                close_price TEXT NOT NULL,
                volume TEXT NOT NULL,
                close_time_ms INTEGER NOT NULL,
                close_time_utc TEXT NOT NULL,
                quote_asset_volume TEXT NOT NULL,
                PRIMARY KEY (symbol, interval, open_time_ms)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval_open_time
            ON klines (symbol, interval, open_time_ms)
            """
        )

    @staticmethod
    def _has_klines_table(conn: sqlite3.Connection) -> bool:
        """Проверяет, существует ли таблица `klines` в открытом SQLite-соединении."""
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'klines'"
        ).fetchone()
        return row is not None


class MexcKlineClient:
    """Минимальный HTTP-клиент для публичного endpoint-а свечей MEXC."""

    def __init__(self, base_url: str = "https://api.mexc.com", timeout_seconds: float = 30.0) -> None:
        """Создает клиент MEXC с базовым URL и timeout-ом запроса."""
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[list[Any]]:
        """Запрашивает пачку свечей MEXC для указанного символа, интервала и диапазона."""
        params = urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
        )
        request = Request(
            f"{self.base_url}{MEXC_KLINES_PATH}?{params}",
            headers={"User-Agent": "pj22-btc-research/0.1"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise MexcAPIError(f"MEXC HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise MexcAPIError(f"Cannot connect to MEXC: {exc.reason}") from exc

        payload = json.loads(body)
        if not isinstance(payload, list):
            raise MexcAPIError(f"Unexpected MEXC klines payload: {payload!r}")
        return payload


def sync_klines(
    config: DownloaderConfig,
    *,
    client: MexcKlineClient | Any | None = None,
    store: YearlySQLiteKlineStore | None = None,
    now_ms: int | None = None,
) -> DownloadSummary:
    """Докачивает свечи от последней сохраненной записи до последней закрытой свечи."""
    step_ms = interval_ms(config.interval)
    start_ms = utc_ms(config.start_date)
    end_ms = closed_open_time_ms(
        int(time.time() * 1000) if now_ms is None else now_ms,
        config.interval,
    )
    if store is None:
        store = YearlySQLiteKlineStore(config.sqlite_dir)
    if client is None:
        client = MexcKlineClient(config.base_url, config.request_timeout_seconds)

    latest = store.latest_open_time_ms(config.symbol, config.interval)
    cursor = max(start_ms, latest + step_ms) if latest is not None else start_ms
    first_requested = cursor if cursor <= end_ms else None
    fetched_rows = 0
    inserted_rows = 0
    batches = 0
    last_open_time = latest

    while cursor <= end_ms:
        request_end = min(end_ms, cursor + step_ms * config.request_limit - step_ms)
        rows = client.fetch_klines(
            symbol=config.symbol,
            interval=config.interval,
            start_ms=cursor,
            end_ms=request_end,
            limit=config.request_limit,
        )
        batches += 1
        if not rows:
            if fetched_rows == 0:
                requested_from = datetime.fromtimestamp(cursor / 1000, UTC).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
                requested_to = datetime.fromtimestamp(request_end / 1000, UTC).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
                raise MexcNoDataError(
                    "MEXC не вернул свечи для запрошенного диапазона "
                    f"{requested_from} - {requested_to}. "
                    f"Для старых {config.interval}-диапазонов REST endpoint может не отдавать историю."
                )
            break

        klines = [
            Kline.from_mexc_row(config.symbol, config.interval, row)
            for row in rows
            if cursor <= int(row[0]) <= end_ms
        ]
        if not klines:
            break

        fetched_rows += len(klines)
        inserted_rows += store.insert_klines(klines)
        last_open_time = max(kline.open_time_ms for kline in klines)
        cursor = last_open_time + step_ms

        if config.request_pause_seconds > 0 and cursor <= end_ms:
            time.sleep(config.request_pause_seconds)

    return DownloadSummary(
        symbol=config.symbol,
        interval=config.interval,
        db_dir=config.sqlite_dir,
        start_ms=start_ms,
        end_ms=end_ms,
        first_requested_ms=first_requested,
        last_open_time_ms=last_open_time,
        fetched_rows=fetched_rows,
        inserted_rows=inserted_rows,
        batches=batches,
    )


def load_config(path: Path) -> DownloaderConfig:
    """Читает `settings.yaml` и возвращает нормализованный `DownloaderConfig`."""
    settings_path = Path(path)
    raw = _parse_simple_yaml(settings_path)
    mexc = _section(raw, "mexc")
    storage = _section(raw, "storage")

    sqlite_dir = _required(storage, "sqlite_dir")
    sqlite_path = Path(str(sqlite_dir))
    if not sqlite_path.is_absolute():
        sqlite_path = settings_path.parent / sqlite_path

    return DownloaderConfig(
        symbol=str(_required(mexc, "symbol")).upper(),
        interval=str(_required(mexc, "interval")),
        start_date=str(_required(mexc, "start_date")),
        sqlite_dir=sqlite_path,
        base_url=str(mexc.get("base_url", "https://api.mexc.com")),
        request_limit=int(mexc.get("request_limit", 1000)),
        request_pause_seconds=float(mexc.get("request_pause_seconds", 0.2)),
        request_timeout_seconds=float(mexc.get("request_timeout_seconds", 30.0)),
    )


def _section(settings: dict[str, Any], key: str) -> dict[str, Any]:
    """Возвращает вложенную секцию настроек и проверяет, что это mapping."""
    value = settings.get(key, {})
    if not isinstance(value, dict):
        raise SettingsError(f"settings.yaml section {key!r} must be a mapping")
    return value


def _required(settings: dict[str, Any], key: str) -> Any:
    """Возвращает обязательное значение из настроек или выбрасывает `SettingsError`."""
    if key not in settings:
        raise SettingsError(f"settings.yaml is missing required key {key!r}")
    return settings[key]


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Разбирает простой YAML-файл с вложенными словарями и скалярными значениями."""
    if not path.exists():
        raise SettingsError(f"Settings file not found: {path}")

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise SettingsError(f"Invalid settings line {line_number}: {raw_line!r}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> str | int | float | bool:
    """Преобразует строковое YAML-значение в простой Python-скаляр."""
    unquoted = value.strip()
    if (unquoted.startswith('"') and unquoted.endswith('"')) or (
        unquoted.startswith("'") and unquoted.endswith("'")
    ):
        return unquoted[1:-1]
    lowered = unquoted.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(unquoted)
    except ValueError:
        pass
    try:
        return float(unquoted)
    except ValueError:
        return unquoted
