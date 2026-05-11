import sqlite3
import sys
import tempfile
import unittest
import re
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pj22_btc.news_markdown import (  # noqa: E402
    create_news_markdown_files,
    load_news_markdown_config,
)


def create_daily_db(path: Path) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE daily_klines (
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
            VALUES (
                'BTCUSDT',
                '1d_msk',
                '2025-09-02',
                1756749600000,
                1756836000000,
                '2025-09-01T21:00:00+03:00',
                '2025-09-02T21:00:00+03:00',
                '108934.63',
                '111760',
                '107451.23',
                '110906.55',
                '10383.56849894',
                '1142074870.74',
                288,
                1
            )
            """
        )
        conn.commit()


def create_two_session_daily_db(path: Path) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE daily_klines (
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
            """,
            [
                (
                    "BTCUSDT",
                    "1d_msk",
                    "2025-09-02",
                    1756749600000,
                    1756836000000,
                    "2025-09-01T21:00:00+03:00",
                    "2025-09-02T21:00:00+03:00",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    288,
                    1,
                ),
                (
                    "BTCUSDT",
                    "1d_msk",
                    "2025-09-03",
                    1756836000000,
                    1756922400000,
                    "2025-09-02T21:00:00+03:00",
                    "2025-09-03T21:00:00+03:00",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    288,
                    1,
                ),
            ],
        )
        conn.commit()


def create_two_session_news_db(path: Path) -> None:
    rows = [
        ("2025-09-01 21:05:00", "2025-09-01", "Bitcoin first session", "investing"),
        ("2025-09-02 21:05:00", "2025-09-02", "Bitcoin latest session", "investing"),
    ]
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE news (
                loaded_at TEXT,
                date TEXT,
                title TEXT,
                provider TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO news (loaded_at, date, title, provider) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def create_news_db(path: Path) -> None:
    rows = [
        ("2025-09-01 20:59:59", "2025-09-01", "Bitcoin before session is excluded", "investing"),
        ("2025-09-01 21:00:00", "2025-09-01", "Bitcoin starts the session", "investing"),
        ("2025-09-01 21:05:00", "2025-09-01", "Soleno Therapeutics reports earnings", "investing"),
        ("2025-09-02 09:15:00", "2025-09-02", "Биткоин удерживает поддержку", "prime"),
        ("2025-09-02 20:59:59", "2025-09-02", "Coinbase rallies with crypto market", "investing"),
        ("2025-09-02 20:59:59", "2025-09-02", "Coinbase rallies with crypto market", "investing"),
        ("2025-09-02 21:00:00", "2025-09-02", "Bitcoin after session is excluded", "investing"),
    ]
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE news (
                loaded_at TEXT,
                date TEXT,
                title TEXT,
                provider TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO news (loaded_at, date, title, provider) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()


class NewsMarkdownTests(unittest.TestCase):
    def test_create_news_markdown_filters_crypto_titles_and_uses_left_closed_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_db = root / "daily_msk.db"
            news_dir = root / "rss"
            output_dir = root / "md"
            news_dir.mkdir()
            create_daily_db(daily_db)
            create_news_db(news_dir / "rss_news_2025_09.db")

            summary = create_news_markdown_files(
                daily_db=daily_db,
                news_db_dir=news_dir,
                markdown_dir=output_dir,
                symbol="BTCUSDT",
                title_regex=r"(?iu)биткоин\w*|\bbitcoin\b|\bcrypto\b|\bcoinbase\b",
            )

            self.assertEqual(summary.sessions_read, 1)
            self.assertEqual(summary.news_rows_read, 7)
            self.assertEqual(summary.filtered_news_rows, 6)
            self.assertEqual(summary.unique_news_rows, 5)
            self.assertEqual(summary.files_written, 1)
            self.assertEqual(summary.titles_written, 3)
            self.assertEqual((output_dir / "2025-09-02.md").read_text(encoding="utf-8"), (
                "Bitcoin starts the session\n\n"
                "Биткоин удерживает поддержку\n\n"
                "Coinbase rallies with crypto market\n"
            ))

    def test_create_news_markdown_deletes_latest_file_and_keeps_existing_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_db = root / "daily_msk.db"
            news_dir = root / "rss"
            output_dir = root / "md"
            news_dir.mkdir()
            output_dir.mkdir()
            create_two_session_daily_db(daily_db)
            create_two_session_news_db(news_dir / "rss_news_2025_09.db")
            first_file = output_dir / "2025-09-02.md"
            latest_file = output_dir / "2025-09-03.md"
            first_file.write_text("manual historical content\n", encoding="utf-8")
            latest_file.write_text("stale latest content\n", encoding="utf-8")

            summary = create_news_markdown_files(
                daily_db=daily_db,
                news_db_dir=news_dir,
                markdown_dir=output_dir,
                symbol="BTCUSDT",
                title_regex=r"(?iu)\bbitcoin\b",
            )

            self.assertEqual(summary.files_deleted, 1)
            self.assertEqual(summary.files_written, 1)
            self.assertEqual(summary.titles_written, 1)
            self.assertEqual(summary.latest_markdown_file, latest_file)
            self.assertEqual(first_file.read_text(encoding="utf-8"), "manual historical content\n")
            self.assertEqual(latest_file.read_text(encoding="utf-8"), "Bitcoin latest session\n")

    def test_load_news_markdown_config_resolves_paths_and_reads_regex_from_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_path = root / "settings.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        "mexc:",
                        "  symbol: BTCUSDT",
                        "daily:",
                        "  sqlite_path: data/mexc/klines/BTCUSDT/daily_msk.db",
                        "news:",
                        "  db_dir: C:\\Users\\Alkor\\gd\\db_rss",
                        "  markdown_dir: data/news_markdown/BTCUSDT/daily_msk",
                        "  crypto_title_regex: '(?iu)биткоин\\w*|\\bbitcoin\\b'",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_news_markdown_config(settings_path)

            self.assertEqual(config.symbol, "BTCUSDT")
            self.assertEqual(config.daily_db, root / "data/mexc/klines/BTCUSDT/daily_msk.db")
            self.assertEqual(config.news_db_dir, Path("C:/Users/Alkor/gd/db_rss"))
            self.assertEqual(config.markdown_dir, root / "data/news_markdown/BTCUSDT/daily_msk")
            self.assertEqual(config.title_regex, r"(?iu)биткоин\w*|\bbitcoin\b")

    def test_project_crypto_regex_rejects_broad_tron_company_name(self) -> None:
        config = load_news_markdown_config(ROOT / "settings.yaml")
        pattern = re.compile(config.title_regex)

        self.assertIsNone(
            pattern.search("M Tron Industries: доходы, прибыль побили прогнозы в Q1")
        )
        self.assertIsNotNone(pattern.search('"Мосбиржа" запустит фьючерсы на Solana, Ripple и Tron'))


if __name__ == "__main__":
    unittest.main()
