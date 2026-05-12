import pickle
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pj22_btc.ollama_client as ollama_client_module  # noqa: E402
from pj22_btc.ollama_client import OllamaClient  # noqa: E402
from pj22_btc.sentiment_analysis import (  # noqa: E402
    attach_market_features,
    is_cached_row,
    load_sentiment_analysis_config,
    parse_ollama_processor_status,
    parse_sentiment_strict,
    run_sentiment_analysis,
)
from scripts.create_sentiment_scores import parse_model_keys  # noqa: E402


class FakeOllamaClient:
    def __init__(self, response: str = "4") -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        keep_alive: str | None,
        timeout_seconds: int,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "keep_alive": keep_alive,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


class SequenceOllamaClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        keep_alive: str | None,
        timeout_seconds: int,
    ) -> str:
        self.calls += 1
        if not self.responses:
            raise AssertionError("No fake response left")
        return self.responses.pop(0)


class CheckpointInspectingClient(FakeOllamaClient):
    def __init__(self, pkl_path: Path, response: str = "4") -> None:
        super().__init__(response)
        self.pkl_path = pkl_path
        self.checkpoint_seen_before_second_call = False

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        keep_alive: str | None,
        timeout_seconds: int,
    ) -> str:
        if len(self.calls) == 1 and self.pkl_path.exists():
            with self.pkl_path.open("rb") as file_obj:
                self.checkpoint_seen_before_second_call = len(pickle.load(file_obj)) == 1
        return super().generate(
            model=model,
            prompt=prompt,
            keep_alive=keep_alive,
            timeout_seconds=timeout_seconds,
        )


class FakeHTTPResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"response": "4"}


def create_daily_db(path: Path) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE daily_klines (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                session_date TEXT NOT NULL,
                open_price TEXT NOT NULL,
                close_price TEXT NOT NULL,
                is_complete INTEGER NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO daily_klines (
                symbol, interval, session_date, open_price, close_price, is_complete
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("BTCUSDT", "1d_msk", "2025-09-02", "100", "110", 1),
                ("BTCUSDT", "1d_msk", "2025-09-03", "110", "105", 1),
                ("BTCUSDT", "1d_msk", "2025-09-04", "105", "120", 1),
            ],
        )
        conn.commit()


class SentimentAnalysisTests(unittest.TestCase):
    def test_ollama_client_uses_deterministic_generation_options(self) -> None:
        captured: dict[str, object] = {}
        original_post = ollama_client_module.requests.post

        def fake_post(url, *, json, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeHTTPResponse()

        try:
            ollama_client_module.requests.post = fake_post

            response = OllamaClient().generate(
                model="gemma3:12b",
                prompt="hello",
                keep_alive="5m",
                timeout_seconds=180,
            )
        finally:
            ollama_client_module.requests.post = original_post

        self.assertEqual(response, "4")
        payload = captured["json"]
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["keep_alive"], "5m")
        self.assertEqual(
            payload["options"],
            {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "seed": 42,
            },
        )

    def test_parse_sentiment_strict_accepts_single_number_and_rejects_explanations(self) -> None:
        self.assertEqual(parse_sentiment_strict(" 4 "), 4)
        self.assertEqual(parse_sentiment_strict("-3,5"), -4)
        self.assertEqual(parse_sentiment_strict("+11"), 10)
        self.assertEqual(parse_sentiment_strict("-10.4"), -10)
        self.assertIsNone(parse_sentiment_strict("sentiment: 4"))
        self.assertIsNone(parse_sentiment_strict(""))

    def test_parse_ollama_processor_status_reads_gpu_cpu_placement(self) -> None:
        ps_output = (
            "NAME             ID              SIZE      PROCESSOR    UNTIL\n"
            "gemma3:12b       abc123          8.1 GB    100% GPU     4 minutes from now\n"
            "qwen3:14b        def456          9.0 GB    47%/53% CPU/GPU  4 minutes from now\n"
        )

        self.assertEqual(parse_ollama_processor_status(ps_output, "gemma3:12b"), "100% GPU")
        self.assertEqual(parse_ollama_processor_status(ps_output, "qwen3:14b"), "47%/53% CPU/GPU")
        self.assertEqual(parse_ollama_processor_status(ps_output, "gpt-oss:20b"), "not loaded")

    def test_is_cached_row_uses_single_row_without_dataframe_scan(self) -> None:
        row = {
            "content_hash": "content-1",
            "prompt_hash": "prompt-1",
            "ollama_model": "gemma3:12b",
            "sentiment": 4,
        }

        self.assertTrue(
            is_cached_row(
                row,
                content_hash="content-1",
                prompt_hash="prompt-1",
                ollama_model="gemma3:12b",
            )
        )
        self.assertTrue(
            is_cached_row(
                {
                    "content_hash": "content-1",
                    "prompt_template": "prompt template",
                    "symbol": "BTCUSDT",
                    "ollama_model": "gemma3:12b",
                    "sentiment": 4,
                },
                content_hash="content-1",
                prompt_template="prompt template",
                symbol="BTCUSDT",
                ollama_model="gemma3:12b",
            )
        )
        self.assertFalse(
            is_cached_row(
                {
                    "content_hash": "content-1",
                    "prompt_template": "prompt template",
                    "symbol": "BTCUSDT",
                    "ollama_model": "gemma3:12b",
                    "sentiment": 4,
                },
                content_hash="content-1",
                prompt_template="changed template",
                symbol="BTCUSDT",
                ollama_model="gemma3:12b",
            )
        )
        self.assertFalse(
            is_cached_row(
                row,
                content_hash="content-2",
                prompt_hash="prompt-1",
                ollama_model="gemma3:12b",
            )
        )
        self.assertFalse(
            is_cached_row(
                {**row, "sentiment": None},
                content_hash="content-1",
                prompt_hash="prompt-1",
                ollama_model="gemma3:12b",
            )
        )

    def test_load_sentiment_config_resolves_paths_prompt_and_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = root / "settings.yaml"
            settings.write_text(
                "\n".join(
                    [
                        "mexc:",
                        "  symbol: BTCUSDT",
                        "daily:",
                        "  sqlite_path: data/daily_msk.db",
                        "news:",
                        "  markdown_dir: data/news_md",
                        "sentiment:",
                        "  output_dir: data/sentiment/BTCUSDT",
                        "  use_cache: true",
                        "  keep_alive: 5m",
                        "  ollama_timeout_seconds: 180",
                        "  token_limit: 16000",
                        "  prompt_template: |",
                        "    Оцени влияние на {symbol}.",
                        "",
                        "    {news_text}",
                        "  models:",
                        "    gemma3_12b:",
                        "      enabled: true",
                        "      ollama_model: gemma3:12b",
                        "    qwen3_14b:",
                        "      enabled: false",
                        "      ollama_model: qwen3:14b",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_sentiment_analysis_config(settings)

            self.assertEqual(config.symbol, "BTCUSDT")
            self.assertEqual(config.daily_db, root / "data/daily_msk.db")
            self.assertEqual(config.markdown_dir, root / "data/news_md")
            self.assertEqual(config.output_dir, root / "data/sentiment/BTCUSDT")
            self.assertIn("{news_text}", config.prompt_template)
            self.assertEqual([model.key for model in config.enabled_models()], ["gemma3_12b"])
            self.assertEqual(config.models["gemma3_12b"].ollama_model, "gemma3:12b")

    def test_run_sentiment_analysis_writes_pkl_with_market_features_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_db = root / "data" / "daily_msk.db"
            markdown_dir = root / "data" / "news_md"
            daily_db.parent.mkdir(parents=True)
            markdown_dir.mkdir(parents=True)
            create_daily_db(daily_db)
            (markdown_dir / "2025-09-02.md").write_text("Bitcoin rallies\n", encoding="utf-8")
            (markdown_dir / "2025-09-03.md").write_text("ETF inflows slow\n", encoding="utf-8")
            settings = root / "settings.yaml"
            settings.write_text(
                "\n".join(
                    [
                        "mexc:",
                        "  symbol: BTCUSDT",
                        "daily:",
                        "  sqlite_path: data/daily_msk.db",
                        "news:",
                        "  markdown_dir: data/news_md",
                        "sentiment:",
                        "  output_dir: data/sentiment/BTCUSDT",
                        "  use_cache: true",
                        "  keep_alive: 5m",
                        "  ollama_timeout_seconds: 180",
                        "  token_limit: 16000",
                        "  prompt_template: |",
                        "    Оцени влияние новостей на {symbol} от -10 до +10.",
                        "",
                        "    {news_text}",
                        "",
                        "    Верни только одно число.",
                        "  models:",
                        "    fake_model:",
                        "      enabled: true",
                        "      ollama_model: fake:1",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_sentiment_analysis_config(settings)
            client = FakeOllamaClient("4")

            summaries = run_sentiment_analysis(config, client=client)

            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].processed_files, 2)
            self.assertEqual(len(client.calls), 2)
            pkl_path = root / "data" / "sentiment" / "BTCUSDT" / "fake_model" / "sentiment_scores.pkl"
            with pkl_path.open("rb") as file_obj:
                df = pd.DataFrame(pickle.load(file_obj))

            self.assertEqual(len(df), 2)
            row = df[df["source_date"] == "2025-09-02"].iloc[0]
            self.assertEqual(row["symbol"], "BTCUSDT")
            self.assertEqual(row["model_key"], "fake_model")
            self.assertEqual(row["ollama_model"], "fake:1")
            self.assertEqual(row["sentiment"], 4)
            self.assertEqual(
                row["generation_options"],
                {
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "seed": 42,
                },
            )
            self.assertEqual(row["body"], 10.0)
            self.assertEqual(row["next_body"], -5.0)
            self.assertEqual(row["next_open_to_open"], -5.0)
            self.assertIn("Bitcoin rallies", row["prompt"])
            self.assertTrue(row["prompt_hash"])

            cached_client = FakeOllamaClient("9")
            cached_summaries = run_sentiment_analysis(config, client=cached_client)

            self.assertEqual(cached_summaries[0].processed_files, 0)
            self.assertEqual(cached_summaries[0].skipped_files, 2)
            self.assertEqual(cached_client.calls, [])

    def test_attach_market_features_uses_date_maps_for_expected_next_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            daily_db = Path(tmp) / "daily_msk.db"
            create_daily_db(daily_db)
            df = pd.DataFrame(
                [
                    {"source_date": "2025-09-02", "sentiment": 4},
                    {"source_date": "2025-09-03", "sentiment": -2},
                ]
            )

            result = attach_market_features(df, daily_db, "BTCUSDT")

            self.assertEqual(result["body"].tolist(), [10.0, -5.0])
            self.assertEqual(result["next_body"].tolist(), [-5.0, 15.0])
            self.assertEqual(float(result["next_open_to_open"].iloc[0]), -5.0)
            self.assertTrue(pd.isna(result["next_open_to_open"].iloc[1]))

    def test_run_sentiment_analysis_emits_progress_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_db = root / "data" / "daily_msk.db"
            markdown_dir = root / "data" / "news_md"
            daily_db.parent.mkdir(parents=True)
            markdown_dir.mkdir(parents=True)
            create_daily_db(daily_db)
            (markdown_dir / "2025-09-02.md").write_text("Bitcoin rallies\n", encoding="utf-8")
            settings = root / "settings.yaml"
            settings.write_text(
                "\n".join(
                    [
                        "mexc:",
                        "  symbol: BTCUSDT",
                        "daily:",
                        "  sqlite_path: data/daily_msk.db",
                        "news:",
                        "  markdown_dir: data/news_md",
                        "sentiment:",
                        "  output_dir: data/sentiment/BTCUSDT",
                        "  use_cache: true",
                        "  prompt_template: \"{news_text}\"",
                        "  models:",
                        "    fake_model:",
                        "      enabled: true",
                        "      ollama_model: fake:1",
                    ]
                ),
                encoding="utf-8",
            )
            events: list[dict[str, object]] = []

            run_sentiment_analysis(
                load_sentiment_analysis_config(settings),
                client=FakeOllamaClient("5"),
                progress=events.append,
                processor_status=lambda model: "100% GPU",
            )

            event_names = [event["event"] for event in events]
            self.assertEqual(
                event_names,
                ["model_start", "pass_start", "file_start", "file_done", "pass_done", "model_done"],
            )
            self.assertEqual(events[0]["model_key"], "fake_model")
            self.assertEqual(events[0]["file_count"], 1)
            self.assertEqual(events[2]["file_index"], 1)
            self.assertEqual(events[2]["processor_status"], "100% GPU")
            self.assertEqual(events[3]["sentiment"], 5)
            self.assertGreaterEqual(events[-1]["elapsed_seconds"], 0.0)

    def test_run_sentiment_analysis_retries_failed_sentiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_db = root / "data" / "daily_msk.db"
            markdown_dir = root / "data" / "news_md"
            daily_db.parent.mkdir(parents=True)
            markdown_dir.mkdir(parents=True)
            create_daily_db(daily_db)
            (markdown_dir / "2025-09-02.md").write_text("Bitcoin rallies\n", encoding="utf-8")
            settings = root / "settings.yaml"
            settings.write_text(
                "\n".join(
                    [
                        "mexc:",
                        "  symbol: BTCUSDT",
                        "daily:",
                        "  sqlite_path: data/daily_msk.db",
                        "news:",
                        "  markdown_dir: data/news_md",
                        "sentiment:",
                        "  output_dir: data/sentiment/BTCUSDT",
                        "  use_cache: true",
                        "  save_every: 0",
                        "  max_retry_passes: 1",
                        "  prompt_template: \"{news_text}\"",
                        "  models:",
                        "    fake_model:",
                        "      enabled: true",
                        "      ollama_model: fake:1",
                    ]
                ),
                encoding="utf-8",
            )

            summaries = run_sentiment_analysis(
                load_sentiment_analysis_config(settings),
                client=SequenceOllamaClient(["not a number", "7"]),
            )

            self.assertEqual(summaries[0].processed_files, 2)
            self.assertEqual(summaries[0].retry_passes_used, 1)
            self.assertGreaterEqual(summaries[0].elapsed_seconds, 0.0)
            pkl_path = root / "data" / "sentiment" / "BTCUSDT" / "fake_model" / "sentiment_scores.pkl"
            with pkl_path.open("rb") as file_obj:
                df = pd.DataFrame(pickle.load(file_obj))
            self.assertEqual(df["sentiment"].tolist(), [7])

    def test_run_sentiment_analysis_saves_checkpoint_after_configured_processed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_db = root / "data" / "daily_msk.db"
            markdown_dir = root / "data" / "news_md"
            daily_db.parent.mkdir(parents=True)
            markdown_dir.mkdir(parents=True)
            create_daily_db(daily_db)
            (markdown_dir / "2025-09-02.md").write_text("Bitcoin rallies\n", encoding="utf-8")
            (markdown_dir / "2025-09-03.md").write_text("ETF inflows slow\n", encoding="utf-8")
            settings = root / "settings.yaml"
            settings.write_text(
                "\n".join(
                    [
                        "mexc:",
                        "  symbol: BTCUSDT",
                        "daily:",
                        "  sqlite_path: data/daily_msk.db",
                        "news:",
                        "  markdown_dir: data/news_md",
                        "sentiment:",
                        "  output_dir: data/sentiment/BTCUSDT",
                        "  use_cache: true",
                        "  save_every: 1",
                        "  prompt_template: \"{news_text}\"",
                        "  models:",
                        "    fake_model:",
                        "      enabled: true",
                        "      ollama_model: fake:1",
                    ]
                ),
                encoding="utf-8",
            )
            pkl_path = root / "data" / "sentiment" / "BTCUSDT" / "fake_model" / "sentiment_scores.pkl"
            client = CheckpointInspectingClient(pkl_path)

            run_sentiment_analysis(load_sentiment_analysis_config(settings), client=client)

            self.assertTrue(client.checkpoint_seen_before_second_call)


class SentimentCliTests(unittest.TestCase):
    def test_parse_model_keys_accepts_comma_separated_values(self) -> None:
        self.assertIsNone(parse_model_keys(None))
        self.assertIsNone(parse_model_keys(" "))
        self.assertEqual(parse_model_keys("gemma3_12b, qwen3_14b"), ["gemma3_12b", "qwen3_14b"])


if __name__ == "__main__":
    unittest.main()
