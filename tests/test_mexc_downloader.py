import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pj22_btc.mexc_downloader import (  # noqa: E402
    DownloaderConfig,
    Kline,
    MexcNoDataError,
    YearlySQLiteKlineStore,
    load_config,
    sync_klines,
    utc_ms,
)


def api_row(open_time_ms: int, close: str = "100.0", interval_ms: int = 300_000) -> list[object]:
    return [
        open_time_ms,
        "99.0",
        "101.0",
        "98.0",
        close,
        "12.5",
        open_time_ms + interval_ms - 1,
        "1240.0",
    ]


class FakeMexcClient:
    def __init__(self, batches: list[list[list[object]]]) -> None:
        self.batches = list(batches)
        self.calls: list[tuple[str, str, int, int, int]] = []

    def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[list[object]]:
        self.calls.append((symbol, interval, start_ms, end_ms, limit))
        if not self.batches:
            return []
        return self.batches.pop(0)


class MexcDownloaderTests(unittest.TestCase):
    def test_yearly_store_uses_open_time_year_for_database_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = YearlySQLiteKlineStore(Path(tmp))

            year_2025 = store.db_path_for_open_time(utc_ms("2025-12-31T23:55:00Z"))
            year_2026 = store.db_path_for_open_time(utc_ms("2026-01-01T00:00:00Z"))

            self.assertEqual(year_2025.name, "2025.db")
            self.assertEqual(year_2026.name, "2026.db")

    def test_insert_klines_is_idempotent_and_tracks_latest_open_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = YearlySQLiteKlineStore(Path(tmp))
            rows = [
                Kline.from_mexc_row("BTCUSDT", "5m", api_row(utc_ms("2025-09-01T00:00:00Z"))),
                Kline.from_mexc_row("BTCUSDT", "5m", api_row(utc_ms("2025-09-01T00:05:00Z"))),
            ]

            first_inserted = store.insert_klines(rows)
            second_inserted = store.insert_klines(rows)

            self.assertEqual(first_inserted, 2)
            self.assertEqual(second_inserted, 0)
            self.assertEqual(store.latest_open_time_ms("BTCUSDT", "5m"), rows[-1].open_time_ms)

            with closing(sqlite3.connect(Path(tmp) / "2025.db")) as conn:
                count = conn.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
            self.assertEqual(count, 2)

    def test_sync_resumes_after_latest_stored_kline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp)
            store = YearlySQLiteKlineStore(db_dir)
            existing = Kline.from_mexc_row(
                "BTCUSDT",
                "5m",
                api_row(utc_ms("2025-09-01T00:00:00Z")),
            )
            store.insert_klines([existing])
            client = FakeMexcClient(
                [
                    [
                        api_row(utc_ms("2025-09-01T00:05:00Z")),
                        api_row(utc_ms("2025-09-01T00:10:00Z")),
                    ],
                    [],
                ]
            )
            config = DownloaderConfig(
                symbol="BTCUSDT",
                interval="5m",
                start_date="2025-09-01",
                sqlite_dir=db_dir,
                request_limit=1000,
                request_pause_seconds=0.0,
            )

            summary = sync_klines(
                config,
                client=client,
                store=store,
                now_ms=utc_ms("2025-09-01T00:20:30Z"),
            )

            self.assertEqual(client.calls[0][2], utc_ms("2025-09-01T00:05:00Z"))
            self.assertEqual(summary.inserted_rows, 2)
            self.assertEqual(store.latest_open_time_ms("BTCUSDT", "5m"), utc_ms("2025-09-01T00:10:00Z"))

    def test_sync_writes_records_to_their_year_database_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp)
            client = FakeMexcClient(
                [
                    [
                        api_row(utc_ms("2025-12-31T23:55:00Z")),
                        api_row(utc_ms("2026-01-01T00:00:00Z")),
                    ],
                    [],
                ]
            )
            config = DownloaderConfig(
                symbol="BTCUSDT",
                interval="5m",
                start_date="2025-12-31",
                sqlite_dir=db_dir,
                request_limit=1000,
                request_pause_seconds=0.0,
            )

            summary = sync_klines(
                config,
                client=client,
                now_ms=utc_ms("2026-01-01T00:10:00Z"),
            )

            self.assertEqual(summary.inserted_rows, 2)
            self.assertTrue((db_dir / "2025.db").exists())
            self.assertTrue((db_dir / "2026.db").exists())

            with closing(sqlite3.connect(db_dir / "2025.db")) as db_2025:
                count_2025 = db_2025.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
            with closing(sqlite3.connect(db_dir / "2026.db")) as db_2026:
                count_2026 = db_2026.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
            self.assertEqual(count_2025, 1)
            self.assertEqual(count_2026, 1)

    def test_sync_raises_clear_error_when_first_requested_range_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp)
            client = FakeMexcClient([[]])
            config = DownloaderConfig(
                symbol="BTCUSDT",
                interval="5m",
                start_date="2025-09-01",
                sqlite_dir=db_dir,
                request_limit=1000,
                request_pause_seconds=0.0,
            )

            with self.assertRaisesRegex(MexcNoDataError, "MEXC не вернул свечи"):
                sync_klines(
                    config,
                    client=client,
                    now_ms=utc_ms("2025-09-01T02:00:00Z"),
                )

    def test_load_config_reads_settings_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.yaml"
            db_dir = Path(tmp) / "data" / "mexc"
            settings_path.write_text(
                "\n".join(
                    [
                        "mexc:",
                        "  symbol: BTCUSDT",
                        "  interval: 5m",
                        "  start_date: 2025-09-01",
                        "  base_url: https://api.mexc.com",
                        "  request_limit: 1000",
                        "  request_pause_seconds: 0",
                        "storage:",
                        f"  sqlite_dir: {db_dir.as_posix()}",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(settings_path)

            self.assertEqual(config.symbol, "BTCUSDT")
            self.assertEqual(config.interval, "5m")
            self.assertEqual(config.start_date, "2025-09-01")
            self.assertEqual(config.sqlite_dir, db_dir)
            self.assertEqual(config.request_limit, 1000)
            self.assertEqual(config.request_pause_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
