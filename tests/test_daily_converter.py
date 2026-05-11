import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pj22_btc.daily_converter import (  # noqa: E402
    DailySQLiteKlineStore,
    aggregate_daily_klines,
    convert_5m_to_daily,
    moscow_tz,
    session_date_for_open_time,
)


def utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    return int(parsed.timestamp() * 1000)


def make_5m_row(open_time_ms: int, price: int) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "interval": "5m",
        "open_time_ms": open_time_ms,
        "open_time_utc": datetime.fromtimestamp(open_time_ms / 1000, UTC).isoformat(),
        "open_price": str(price),
        "high_price": str(price + 10),
        "low_price": str(price - 10),
        "close_price": str(price + 1),
        "volume": "2.5",
        "close_time_ms": open_time_ms + 300_000 - 1,
        "close_time_utc": datetime.fromtimestamp((open_time_ms + 300_000 - 1) / 1000, UTC).isoformat(),
        "quote_asset_volume": "250.0",
    }


def full_session_rows(start_utc: str, start_price: int = 100) -> list[dict[str, object]]:
    start = datetime.fromisoformat(start_utc.replace("Z", "+00:00")).astimezone(UTC)
    return [
        make_5m_row(int((start + timedelta(minutes=5 * index)).timestamp() * 1000), start_price + index)
        for index in range(288)
    ]


def create_source_db(path: Path, rows: list[dict[str, object]]) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE klines (
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
        conn.executemany(
            """
            INSERT INTO klines (
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
            VALUES (
                :symbol,
                :interval,
                :open_time_ms,
                :open_time_utc,
                :open_price,
                :high_price,
                :low_price,
                :close_price,
                :volume,
                :close_time_ms,
                :close_time_utc,
                :quote_asset_volume
            )
            """,
            rows,
        )
        conn.commit()


class DailyConverterTests(unittest.TestCase):
    def test_session_date_uses_moscow_21_00_boundary_and_excludes_21_00_candle(self) -> None:
        before_boundary = session_date_for_open_time(utc_ms("2025-09-02T17:55:00Z"))
        at_boundary = session_date_for_open_time(utc_ms("2025-09-02T18:00:00Z"))

        self.assertEqual(before_boundary.isoformat(), "2025-09-02")
        self.assertEqual(at_boundary.isoformat(), "2025-09-03")

    def test_aggregate_daily_klines_builds_complete_moscow_session(self) -> None:
        rows = full_session_rows("2025-09-01T18:00:00Z", start_price=100)

        candles = aggregate_daily_klines(rows, include_incomplete=False)

        self.assertEqual(len(candles), 1)
        candle = candles[0]
        self.assertEqual(candle.session_date, "2025-09-02")
        self.assertEqual(candle.session_start_msk, "2025-09-01T21:00:00+03:00")
        self.assertEqual(candle.session_end_msk, "2025-09-02T21:00:00+03:00")
        self.assertEqual(candle.open_price, "100")
        self.assertEqual(candle.high_price, "397")
        self.assertEqual(candle.low_price, "90")
        self.assertEqual(candle.close_price, "388")
        self.assertEqual(candle.volume, "720.0")
        self.assertEqual(candle.candle_count, 288)
        self.assertTrue(candle.is_complete)

    def test_aggregate_daily_klines_skips_incomplete_sessions_by_default(self) -> None:
        rows = full_session_rows("2025-09-01T18:00:00Z", start_price=100)[:100]

        candles = aggregate_daily_klines(rows, include_incomplete=False)

        self.assertEqual(candles, [])

    def test_moscow_tz_uses_named_zoneinfo_timezone(self) -> None:
        self.assertEqual(moscow_tz().key, "Europe/Moscow")

    def test_daily_store_writes_single_database_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "daily.db"
            store = DailySQLiteKlineStore(db_path)
            candles = aggregate_daily_klines(
                full_session_rows("2025-12-31T18:00:00Z", start_price=100),
                include_incomplete=False,
            )

            inserted = store.insert_daily_klines(candles)

            self.assertEqual(inserted, 1)
            self.assertTrue(db_path.exists())

    def test_convert_5m_to_daily_reads_yearly_sources_and_writes_single_daily_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "5m"
            output_db = root / "daily.db"
            source_dir.mkdir()
            create_source_db(
                source_dir / "2025.db",
                full_session_rows("2025-09-01T18:00:00Z", start_price=100),
            )

            summary = convert_5m_to_daily(
                source_dir=source_dir,
                output_db=output_db,
                symbol="BTCUSDT",
                include_incomplete=False,
            )

            self.assertEqual(summary.source_rows, 288)
            self.assertEqual(summary.daily_rows, 1)
            self.assertEqual(summary.inserted_rows, 1)
            with closing(sqlite3.connect(output_db)) as conn:
                row = conn.execute(
                    "SELECT session_date, open_price, close_price, candle_count FROM daily_klines"
                ).fetchone()
            self.assertEqual(row, ("2025-09-02", "100", "388", 288))


if __name__ == "__main__":
    unittest.main()
