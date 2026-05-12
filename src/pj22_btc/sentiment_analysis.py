"""Расчет sentiment-оценок markdown-новостей через несколько Ollama-моделей."""

from __future__ import annotations

import hashlib
import math
import pickle
import re
import sqlite3
import subprocess
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd
import tiktoken
import yaml

from pj22_btc.daily_converter import DAILY_INTERVAL
from pj22_btc.mexc_downloader import SettingsError
from pj22_btc.ollama_client import DETERMINISTIC_OLLAMA_OPTIONS, OllamaClient


DEFAULT_PROMPT_TEMPLATE = (
    "Оцени влияние новостей на {symbol} от -10 до +10.\n\n"
    "Текст новости:\n\n{news_text}\n\n"
    "Верни только одно число от -10 до +10 без пояснений."
)
DEFAULT_TOKEN_LIMIT = 16_000
STRICT_NUMBER_REGEX = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*$")
ProgressCallback = Callable[[dict[str, Any]], None]
ProcessorStatusProvider = Callable[[str], str]


class TextGenerator(Protocol):
    """Минимальный интерфейс клиента генерации текста."""

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        keep_alive: str | None,
        timeout_seconds: int,
    ) -> str:
        """Возвращает ответ модели."""


@dataclass(frozen=True)
class SentimentModelConfig:
    """Настройки одной модели sentiment."""

    key: str
    ollama_model: str
    enabled: bool = True
    output_pkl: Path | None = None


@dataclass(frozen=True)
class SentimentAnalysisConfig:
    """Настройки расчета sentiment для всех моделей."""

    symbol: str
    daily_db: Path
    markdown_dir: Path
    output_dir: Path
    prompt_template: str
    models: dict[str, SentimentModelConfig]
    use_cache: bool = True
    keep_alive: str | None = "5m"
    ollama_timeout_seconds: int = 60
    token_limit: int = DEFAULT_TOKEN_LIMIT
    save_every: int = 10
    max_retry_passes: int = 3

    def enabled_models(self) -> list[SentimentModelConfig]:
        """Возвращает модели, отмеченные как enabled."""
        return [model for model in self.models.values() if model.enabled]


@dataclass(frozen=True)
class SentimentRunSummary:
    """Краткая сводка одного прогона модели."""

    model_key: str
    ollama_model: str
    output_pkl: Path
    markdown_files: int
    processed_files: int
    skipped_files: int
    failed_files: int
    rows_saved: int
    retry_passes_used: int
    elapsed_seconds: float


@dataclass(frozen=True)
class DailyMarketRow:
    """Одна дневная свеча с заранее рассчитанным телом свечи."""

    session_date: str
    open_price: Decimal
    close_price: Decimal

    @property
    def body(self) -> float:
        """Возвращает close-open как float для аналитических таблиц."""
        return float(self.close_price - self.open_price)


def load_sentiment_analysis_config(path: Path) -> SentimentAnalysisConfig:
    """Читает sentiment-настройки из `settings.yaml`."""
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    mexc = _section(raw, "mexc")
    daily = _section(raw, "daily")
    news = _section(raw, "news")
    sentiment = _section(raw, "sentiment")

    symbol = str(_required(mexc, "symbol")).upper()
    output_dir = _resolve_settings_path(
        settings_path,
        sentiment.get("output_dir", f"data/sentiment/{symbol}"),
    )
    models = _load_model_configs(settings_path, sentiment)
    if not models:
        raise SettingsError("settings.yaml sentiment.models must contain at least one model")

    return SentimentAnalysisConfig(
        symbol=symbol,
        daily_db=_resolve_settings_path(settings_path, _required(daily, "sqlite_path")),
        markdown_dir=_resolve_settings_path(settings_path, _required(news, "markdown_dir")),
        output_dir=output_dir,
        prompt_template=str(sentiment.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE),
        models=models,
        use_cache=bool(sentiment.get("use_cache", True)),
        keep_alive=_optional_str(sentiment.get("keep_alive", "5m")),
        ollama_timeout_seconds=int(sentiment.get("ollama_timeout_seconds", 60)),
        token_limit=int(sentiment.get("token_limit", DEFAULT_TOKEN_LIMIT)),
        save_every=int(sentiment.get("save_every", 10)),
        max_retry_passes=max(0, int(sentiment.get("max_retry_passes", 3))),
    )


def parse_sentiment_strict(response: str) -> int | None:
    """Строго парсит ответ модели как одно число `-10..10`."""
    if not response:
        return None
    match = STRICT_NUMBER_REGEX.fullmatch(response)
    if not match:
        return None
    value = match.group(1).replace(",", ".")
    try:
        score = float(value)
    except ValueError:
        return None
    rounded = _round_half_away_from_zero(score)
    return max(min(rounded, 10), -10)


def parse_ollama_processor_status(ps_output: str, model: str) -> str:
    """Извлекает CPU/GPU-размещение модели из вывода `ollama ps`."""
    processor_pattern = re.compile(r"((?:\d+%/\d+%\s+CPU/GPU)|(?:\d+%\s+(?:CPU|GPU)))\s+\d+")
    for line in ps_output.splitlines():
        if not line.strip().startswith(model):
            continue
        match = processor_pattern.search(line)
        if match:
            return match.group(1)
    return "not loaded"


def get_ollama_processor_status(model: str) -> str:
    """Возвращает CPU/GPU-размещение модели из `ollama ps` без остановки пайплайна."""
    try:
        completed = subprocess.run(
            ["ollama", "ps"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception as exc:
        return f"unavailable ({exc.__class__.__name__})"
    return parse_ollama_processor_status(completed.stdout, model)


def run_sentiment_analysis(
    config: SentimentAnalysisConfig,
    *,
    model_keys: list[str] | None = None,
    client: TextGenerator | None = None,
    progress: ProgressCallback | None = None,
    processor_status: ProcessorStatusProvider | None = None,
) -> list[SentimentRunSummary]:
    """Считает sentiment PKL для выбранных или enabled-моделей."""
    selected_models = _select_models(config, model_keys)
    if client is None:
        client = OllamaClient()
    return [
        _run_model_sentiment(
            config=config,
            model_config=model_config,
            client=client,
            progress=progress,
            processor_status=processor_status,
        )
        for model_config in selected_models
    ]


def find_markdown_files(markdown_dir: Path) -> list[Path]:
    """Возвращает отсортированные markdown-файлы новостей."""
    return sorted(path for path in Path(markdown_dir).rglob("*.md") if path.is_file())


def build_prompt(symbol: str, prompt_template: str, news_text: str) -> str:
    """Подставляет `{symbol}`, `{ticker}` и `{news_text}` в prompt template."""
    return prompt_template.format(symbol=symbol, ticker=symbol, news_text=news_text)


def output_pkl_for_model(config: SentimentAnalysisConfig, model_config: SentimentModelConfig) -> Path:
    """Возвращает путь к PKL для модели."""
    if model_config.output_pkl is not None:
        return model_config.output_pkl
    return config.output_dir / model_config.key / "sentiment_scores.pkl"


def _run_model_sentiment(
    *,
    config: SentimentAnalysisConfig,
    model_config: SentimentModelConfig,
    client: TextGenerator,
    progress: ProgressCallback | None,
    processor_status: ProcessorStatusProvider | None,
) -> SentimentRunSummary:
    started_at = time.monotonic()
    output_pkl = output_pkl_for_model(config, model_config)
    files = find_markdown_files(config.markdown_dir)
    _emit_progress(
        progress,
        event="model_start",
        model_key=model_config.key,
        ollama_model=model_config.ollama_model,
        output_pkl=output_pkl,
        file_count=len(files),
    )
    existing_df = _load_existing_results(output_pkl) if config.use_cache else pd.DataFrame()
    rows_by_path = _rows_by_path(existing_df)
    processed = 0
    skipped = 0
    failed = 0
    retry_passes_used = 0

    for retry_pass in range(config.max_retry_passes + 1):
        _emit_progress(
            progress,
            event="pass_start",
            model_key=model_config.key,
            ollama_model=model_config.ollama_model,
            retry_pass=retry_pass,
            retry_limit=config.max_retry_passes,
            file_count=len(files),
        )
        processed_since_checkpoint = 0
        pass_processed = 0
        pass_skipped = 0
        pass_failed = 0

        for file_index, md_file in enumerate(files, start=1):
            md_file_path = str(md_file.resolve())
            content_hash = _sha256_file(md_file)

            if config.use_cache and is_cached_row(
                rows_by_path.get(md_file_path),
                content_hash=content_hash,
                prompt_template=config.prompt_template,
                symbol=config.symbol,
                ollama_model=model_config.ollama_model,
            ):
                skipped += 1
                pass_skipped += 1
                _emit_progress(
                    progress,
                    event="file_skipped",
                    model_key=model_config.key,
                    ollama_model=model_config.ollama_model,
                    retry_pass=retry_pass,
                    file_index=file_index,
                    file_count=len(files),
                    file_path=md_file,
                    source_date=_extract_date_from_path(md_file),
                )
                continue

            news_text = md_file.read_text(encoding="utf-8", errors="replace").strip()
            prompt = build_prompt(config.symbol, config.prompt_template, news_text)
            prompt_hash = _sha256_text(prompt)
            current_processor_status = (
                processor_status(model_config.ollama_model)
                if processor_status is not None
                else "not checked"
            )
            _emit_progress(
                progress,
                event="file_start",
                model_key=model_config.key,
                ollama_model=model_config.ollama_model,
                retry_pass=retry_pass,
                file_index=file_index,
                file_count=len(files),
                file_path=md_file,
                source_date=_extract_date_from_path(md_file),
                processor_status=current_processor_status,
            )
            try:
                raw_response = client.generate(
                    model=model_config.ollama_model,
                    prompt=prompt,
                    keep_alive=config.keep_alive,
                    timeout_seconds=config.ollama_timeout_seconds,
                )
                sentiment = parse_sentiment_strict(raw_response)
            except Exception as exc:
                raw_response = f"{type(exc).__name__}: {exc}"
                sentiment = None

            if sentiment is None:
                failed += 1
                pass_failed += 1

            prompt_tokens = _token_count(prompt)
            rows_by_path[md_file_path] = {
                "file_path": md_file_path,
                "content_hash": content_hash,
                "source_date": _extract_date_from_path(md_file),
                "symbol": config.symbol,
                "model_key": model_config.key,
                "ollama_model": model_config.ollama_model,
                "prompt_template": config.prompt_template,
                "prompt_hash": prompt_hash,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "generation_options": DETERMINISTIC_OLLAMA_OPTIONS.copy(),
                "processor_status": current_processor_status,
                "raw_response": raw_response,
                "sentiment": sentiment,
                "processed_at": datetime.now(UTC),
            }
            _emit_progress(
                progress,
                event="file_done",
                model_key=model_config.key,
                ollama_model=model_config.ollama_model,
                retry_pass=retry_pass,
                file_index=file_index,
                file_count=len(files),
                file_path=md_file,
                source_date=_extract_date_from_path(md_file),
                sentiment=sentiment,
                prompt_tokens=prompt_tokens,
                failed=sentiment is None,
            )
            processed += 1
            pass_processed += 1
            processed_since_checkpoint += 1

            if config.save_every > 0 and processed_since_checkpoint >= config.save_every:
                checkpoint_df = _prepare_output_df(
                    pd.DataFrame(rows_by_path.values()),
                    config.daily_db,
                    config.symbol,
                    drop_failed=False,
                )
                _save_results(output_pkl, checkpoint_df)
                _emit_progress(
                    progress,
                    event="checkpoint_saved",
                    model_key=model_config.key,
                    ollama_model=model_config.ollama_model,
                    retry_pass=retry_pass,
                    output_pkl=output_pkl,
                    rows_saved=len(checkpoint_df),
                )
                processed_since_checkpoint = 0

        df = _prepare_output_df(
            pd.DataFrame(rows_by_path.values()),
            config.daily_db,
            config.symbol,
            drop_failed=False,
        )
        _emit_progress(
            progress,
            event="pass_done",
            model_key=model_config.key,
            ollama_model=model_config.ollama_model,
            retry_pass=retry_pass,
            retry_limit=config.max_retry_passes,
            processed_files=pass_processed,
            skipped_files=pass_skipped,
            failed_files=pass_failed,
            rows_saved=len(df),
        )
        if not _has_failed_sentiments(df):
            break
        retry_passes_used = retry_pass + 1
        if retry_pass >= config.max_retry_passes:
            df = _drop_failed_sentiments(df)
            break

    df = _prepare_output_df(df, config.daily_db, config.symbol, drop_failed=True)
    df = _deduplicate_by_source_date(df)
    _save_results(output_pkl, df)

    elapsed_seconds = time.monotonic() - started_at
    _emit_progress(
        progress,
        event="model_done",
        model_key=model_config.key,
        ollama_model=model_config.ollama_model,
        output_pkl=output_pkl,
        file_count=len(files),
        processed_files=processed,
        skipped_files=skipped,
        failed_files=failed,
        rows_saved=len(df),
        retry_passes_used=retry_passes_used,
        elapsed_seconds=elapsed_seconds,
    )

    return SentimentRunSummary(
        model_key=model_config.key,
        ollama_model=model_config.ollama_model,
        output_pkl=output_pkl,
        markdown_files=len(files),
        processed_files=processed,
        skipped_files=skipped,
        failed_files=failed,
        rows_saved=len(df),
        retry_passes_used=retry_passes_used,
        elapsed_seconds=elapsed_seconds,
    )


def attach_market_features(df: pd.DataFrame, daily_db: Path, symbol: str) -> pd.DataFrame:
    """Добавляет `body`, `next_body`, `next_open_to_open` из daily_klines."""
    if df.empty or not Path(daily_db).exists():
        return df

    market_rows = _load_daily_market_rows(Path(daily_db), symbol)
    if not market_rows:
        return df

    body_by_date = {row.session_date: row.body for row in market_rows}
    next_body_by_date: dict[str, float | None] = {}
    next_open_to_open_by_date: dict[str, float | None] = {}
    for idx, row in enumerate(market_rows):
        next_row = market_rows[idx + 1] if idx + 1 < len(market_rows) else None
        next_next_row = market_rows[idx + 2] if idx + 2 < len(market_rows) else None
        next_body_by_date[row.session_date] = next_row.body if next_row is not None else None
        if next_row is None or next_next_row is None:
            next_open_to_open_by_date[row.session_date] = None
        else:
            next_open_to_open_by_date[row.session_date] = float(
                next_next_row.open_price - next_row.open_price
            )

    def body_for(source_date: Any) -> float | None:
        date_str = _date_str(source_date)
        if date_str is None:
            return None
        return body_by_date.get(date_str)

    def next_body_for(source_date: Any) -> float | None:
        date_str = _date_str(source_date)
        if date_str is None:
            return None
        return next_body_by_date.get(date_str)

    def next_open_to_open_for(source_date: Any) -> float | None:
        date_str = _date_str(source_date)
        if date_str is None:
            return None
        return next_open_to_open_by_date.get(date_str)

    result = df.copy()
    result["body"] = result["source_date"].apply(body_for)
    result["next_body"] = result["source_date"].apply(next_body_for)
    result["next_open_to_open"] = result["source_date"].apply(next_open_to_open_for)
    return result


def _prepare_output_df(
    df: pd.DataFrame,
    daily_db: Path,
    symbol: str,
    *,
    drop_failed: bool,
) -> pd.DataFrame:
    """Готовит DataFrame к checkpoint/final-сохранению."""
    result = _deduplicate_by_source_date(df)
    if drop_failed:
        result = _drop_failed_sentiments(result)
    return attach_market_features(result, daily_db, symbol)


def _load_model_configs(
    settings_path: Path,
    sentiment: dict[str, Any],
) -> dict[str, SentimentModelConfig]:
    raw_models = sentiment.get("models") or {}
    if not isinstance(raw_models, dict):
        raise SettingsError("settings.yaml sentiment.models must be a mapping")

    models: dict[str, SentimentModelConfig] = {}
    for key, raw_model in raw_models.items():
        if not isinstance(raw_model, dict):
            raise SettingsError(f"settings.yaml sentiment.models.{key} must be a mapping")
        output_pkl = raw_model.get("output_pkl")
        models[str(key)] = SentimentModelConfig(
            key=str(key),
            ollama_model=str(_required(raw_model, "ollama_model")),
            enabled=bool(raw_model.get("enabled", True)),
            output_pkl=(
                _resolve_settings_path(settings_path, output_pkl)
                if output_pkl is not None
                else None
            ),
        )
    return models


def _emit_progress(progress: ProgressCallback | None, **event: Any) -> None:
    if progress is not None:
        progress(event)


def is_cached_row(
    row: dict[str, Any] | None,
    *,
    content_hash: str,
    ollama_model: str,
    prompt_hash: str | None = None,
    prompt_template: str | None = None,
    symbol: str | None = None,
) -> bool:
    """Проверяет одну кэш-строку без DataFrame scan."""
    if row is None:
        return False
    if row.get("content_hash") != content_hash or row.get("ollama_model") != ollama_model:
        return False
    if prompt_hash is not None:
        prompt_matches = row.get("prompt_hash") == prompt_hash
    elif prompt_template is not None:
        prompt_matches = row.get("prompt_template") == prompt_template
    else:
        prompt_matches = True
    if symbol is not None and row.get("symbol") != symbol:
        return False
    return (
        prompt_matches
        and pd.notna(row.get("sentiment"))
    )


def _select_models(
    config: SentimentAnalysisConfig,
    model_keys: list[str] | None,
) -> list[SentimentModelConfig]:
    if model_keys is None:
        selected = config.enabled_models()
    else:
        unknown = [key for key in model_keys if key not in config.models]
        if unknown:
            available = ", ".join(sorted(config.models))
            raise SettingsError(f"Unknown sentiment model(s): {unknown}. Available: {available}")
        selected = [config.models[key] for key in model_keys]
    if not selected:
        raise SettingsError("No sentiment models selected")
    return selected


def _rows_by_path(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty or "file_path" not in df.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        rows[str(row["file_path"])] = row
    return rows


def _load_existing_results(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    with Path(path).open("rb") as file_obj:
        return pd.DataFrame(pickle.load(file_obj))


def _save_results(path: Path, df: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as file_obj:
        pickle.dump(df, file_obj)


def _load_daily_market_rows(daily_db: Path, symbol: str) -> list[DailyMarketRow]:
    with closing(sqlite3.connect(daily_db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT session_date, open_price, close_price
            FROM daily_klines
            WHERE symbol = ? AND interval = ? AND is_complete = 1
            ORDER BY session_date
            """,
            (symbol, DAILY_INTERVAL),
        ).fetchall()

    return [
        DailyMarketRow(
            session_date=str(row["session_date"]),
            open_price=Decimal(str(row["open_price"])),
            close_price=Decimal(str(row["close_price"])),
        )
        for row in rows
    ]


def _deduplicate_by_source_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "source_date" not in df.columns:
        return df
    return (
        df.sort_values(["source_date", "processed_at"], kind="stable")
        .drop_duplicates(subset="source_date", keep="last")
        .reset_index(drop=True)
    )


def _has_failed_sentiments(df: pd.DataFrame) -> bool:
    return not df.empty and "sentiment" in df.columns and df["sentiment"].isna().any()


def _drop_failed_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sentiment" not in df.columns:
        return df
    return df[df["sentiment"].notna()].reset_index(drop=True)


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _token_count(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _extract_date_from_path(path: Path) -> str | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(path))
    return match.group(1) if match else None


def _date_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _next_index(dates: list[str], date_str: str) -> int | None:
    for idx, candidate in enumerate(dates):
        if candidate > date_str:
            return idx
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _section(settings: dict[str, Any], key: str) -> dict[str, Any]:
    value = settings.get(key, {})
    if not isinstance(value, dict):
        raise SettingsError(f"settings.yaml section {key!r} must be a mapping")
    return value


def _required(settings: dict[str, Any], key: str) -> Any:
    if key not in settings:
        raise SettingsError(f"settings.yaml is missing required key {key!r}")
    return settings[key]


def _resolve_settings_path(settings_path: Path, value: Any) -> Path:
    resolved = Path(str(value))
    if not resolved.is_absolute():
        resolved = settings_path.parent / resolved
    return resolved
