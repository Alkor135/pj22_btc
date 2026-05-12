"""Исследовательская аналитика sentiment-сигналов BTCUSDT.

Модуль переносит логику обработки из старого sentiment-проекта в архитектуру
`pj22_btc`: одна конфигурация, один символ BTCUSDT и несколько моделей из
`settings.yaml`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pj22_btc.mexc_downloader import SettingsError
from pj22_btc.sentiment_analysis import load_sentiment_analysis_config, output_pkl_for_model


SENTIMENT_RANGE = range(-10, 11)
VALID_ACTIONS = {"follow", "invert", "skip"}
VALID_TARGET_COLUMNS = {"next_body", "next_open_to_open"}
DEFAULT_TARGET_COLUMN = "next_body"


@dataclass(frozen=True)
class SentimentResearchConfig:
    """Настройки аналитических отчетов sentiment-моделей."""

    symbol: str
    output_dir: Path
    quantity: int
    notional_capital: float
    date_from: date | None
    date_to: date | None
    target_column: str
    model_pkls: dict[str, Path]
    enabled_model_keys: list[str]

    def selected_model_keys(self, model_keys: list[str] | None) -> list[str]:
        """Возвращает выбранные модели или все enabled-модели по умолчанию."""
        if model_keys is None:
            selected = list(self.enabled_model_keys)
        else:
            unknown = [key for key in model_keys if key not in self.model_pkls]
            if unknown:
                available = ", ".join(sorted(self.model_pkls))
                raise SettingsError(f"Unknown sentiment model(s): {unknown}. Available: {available}")
            selected = list(model_keys)
        if not selected:
            raise SettingsError("No sentiment models selected")
        return selected

    def model_pkl(self, model_key: str) -> Path:
        """Возвращает путь к sentiment PKL выбранной модели."""
        try:
            return self.model_pkls[model_key]
        except KeyError as exc:
            available = ", ".join(sorted(self.model_pkls))
            raise SettingsError(f"Unknown sentiment model {model_key!r}. Available: {available}") from exc

    def model_output_dir(self, model_key: str) -> Path:
        """Возвращает корневую папку отчетов выбранной модели."""
        return self.output_dir / model_key

    def group_stats_xlsx(self, model_key: str, target_column: str | None = None) -> Path:
        """Возвращает путь к XLSX с групповой статистикой."""
        target = target_column or self.target_column
        return (
            self.model_output_dir(model_key)
            / "group_stats"
            / f"sentiment_group_stats_{target}.xlsx"
        )

    def rules_yaml(self, model_key: str, target_column: str | None = None) -> Path:
        """Возвращает путь к YAML с рекомендованными правилами."""
        target = target_column or self.target_column
        return self.model_output_dir(model_key) / "rules" / f"rules_{target}.yaml"

    def backtest_xlsx(self, model_key: str, target_column: str | None = None) -> Path:
        """Возвращает путь к XLSX с результатами backtest."""
        target = target_column or self.target_column
        return (
            self.model_output_dir(model_key)
            / "backtest"
            / f"sentiment_backtest_{target}_results.xlsx"
        )

    def backtest_html(self, model_key: str, target_column: str | None = None) -> Path:
        """Возвращает путь к HTML-отчету backtest."""
        target = target_column or self.target_column
        return self.model_output_dir(model_key) / "plots" / f"sentiment_backtest_{target}.html"


@dataclass(frozen=True)
class GroupStatsRunResult:
    """Результат построения group stats для одной модели."""

    model_key: str
    target_column: str
    trades: pd.DataFrame
    grouped: pd.DataFrame
    output_xlsx: Path


@dataclass(frozen=True)
class RulesRunResult:
    """Результат генерации rules YAML для одной модели."""

    model_key: str
    target_column: str
    rules: list[dict[str, int | str]]
    input_xlsx: Path
    output_yaml: Path


@dataclass(frozen=True)
class BacktestRunResult:
    """Результат backtest для одной модели."""

    model_key: str
    target_column: str
    result: pd.DataFrame
    rules_yaml: Path
    output_xlsx: Path
    output_html: Path


def load_sentiment_research_config(path: Path) -> SentimentResearchConfig:
    """Читает настройки reports и sentiment-моделей из `settings.yaml`."""
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    reports = _section(raw, "reports")
    sentiment_config = load_sentiment_analysis_config(settings_path)
    output_dir = _resolve_settings_path(
        settings_path,
        reports.get("output_dir", f"reports/sentiment/{sentiment_config.symbol}"),
    )
    target_column = str(reports.get("target_column", DEFAULT_TARGET_COLUMN))
    _validate_target_column_for_settings(target_column)

    model_pkls = {
        key: output_pkl_for_model(sentiment_config, model)
        for key, model in sentiment_config.models.items()
    }

    return SentimentResearchConfig(
        symbol=sentiment_config.symbol,
        output_dir=output_dir,
        quantity=int(reports.get("quantity", 1)),
        notional_capital=float(reports.get("notional_capital", 10_000)),
        date_from=_parse_date(reports.get("date_from")),
        date_to=_parse_date(reports.get("date_to")),
        target_column=target_column,
        model_pkls=model_pkls,
        enabled_model_keys=[model.key for model in sentiment_config.enabled_models()],
    )


def load_sentiment_scores(path: Path, target_column: str) -> pd.DataFrame:
    """Загружает sentiment PKL и возвращает очищенный DataFrame."""
    _validate_target_column(target_column)
    if not Path(path).exists():
        raise ValueError(f"Файл sentiment PKL не найден: {path}")
    with Path(path).open("rb") as file_obj:
        data = pickle.load(file_obj)
    return prepare_sentiment_frame(pd.DataFrame(data), target_column)


def prepare_sentiment_frame(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Проверяет sentiment-таблицу и приводит типы нужных колонок."""
    _validate_target_column(target_column)
    required = {"source_date", "sentiment", target_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PKL не содержит обязательные колонки: {sorted(missing)}")

    result = df.copy()
    result["source_date"] = pd.to_datetime(result["source_date"], errors="coerce").dt.date
    result["sentiment"] = pd.to_numeric(result["sentiment"], errors="coerce")
    result[target_column] = pd.to_numeric(result[target_column], errors="coerce")
    result = result.dropna(subset=["source_date", "sentiment", target_column])

    if result["source_date"].duplicated().any():
        duplicates = result.loc[result["source_date"].duplicated(keep=False), "source_date"]
        raise ValueError(f"В sentiment PKL несколько строк за одну дату: {sorted(duplicates.unique())[:5]}")

    return result.sort_values("source_date").reset_index(drop=True)


def filter_by_date(
    df: pd.DataFrame,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
) -> pd.DataFrame:
    """Фильтрует DataFrame по `source_date`."""
    result = df.copy()
    if date_from is not None:
        result = result[result["source_date"] >= date_from]
    if date_to is not None:
        result = result[result["source_date"] <= date_to]
    return result.reset_index(drop=True)


def build_follow_trades(
    df: pd.DataFrame,
    *,
    quantity: int,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """Строит follow-сделки: sentiment >= 0 -> LONG, sentiment < 0 -> SHORT."""
    prepared = prepare_sentiment_frame(df, target_column)
    rows: list[dict[str, Any]] = []
    for row in prepared.to_dict("records"):
        sentiment = float(row["sentiment"])
        target_move = float(row[target_column])
        direction = "LONG" if sentiment >= 0 else "SHORT"
        pnl = target_move * quantity if direction == "LONG" else -target_move * quantity
        rows.append(
            {
                "source_date": row["source_date"],
                "sentiment": sentiment,
                "direction": direction,
                "target_column": target_column,
                "target_move": target_move,
                "quantity": quantity,
                "pnl": pnl,
            }
        )
    return pd.DataFrame(rows)


def group_by_sentiment(trades: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует сделки по значениям sentiment от -10 до +10."""
    if trades.empty:
        grouped = pd.DataFrame(columns=["sentiment", "count_pos", "count_neg", "total_pnl", "trades"])
    else:
        grouped = (
            trades.groupby("sentiment")
            .agg(
                count_pos=("pnl", lambda series: int((series > 0).sum())),
                count_neg=("pnl", lambda series: int((series < 0).sum())),
                total_pnl=("pnl", "sum"),
                trades=("pnl", "size"),
            )
            .reset_index()
        )

    full = pd.DataFrame({"sentiment": [float(sentiment) for sentiment in SENTIMENT_RANGE]})
    result = full.merge(grouped, on="sentiment", how="left").fillna(
        {"count_pos": 0, "count_neg": 0, "total_pnl": 0.0, "trades": 0}
    )
    for column in ("count_pos", "count_neg", "trades"):
        result[column] = result[column].astype(int)
    return result.sort_values("sentiment").reset_index(drop=True)


def run_group_stats_for_model(
    config: SentimentResearchConfig,
    model_key: str,
    *,
    target_column: str | None = None,
    quantity: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> GroupStatsRunResult:
    """Строит и сохраняет group stats для одной модели."""
    target = target_column or config.target_column
    selected_quantity = quantity if quantity is not None else config.quantity
    selected_from = date_from if date_from is not None else config.date_from
    selected_to = date_to if date_to is not None else config.date_to

    source = load_sentiment_scores(config.model_pkl(model_key), target)
    source = filter_by_date(source, date_from=selected_from, date_to=selected_to)
    if source.empty:
        raise ValueError("После фильтра по дате не осталось sentiment-записей")

    trades = build_follow_trades(source, quantity=selected_quantity, target_column=target)
    if trades.empty:
        raise ValueError("Нет сделок для групповой статистики")

    grouped = group_by_sentiment(trades)
    output_xlsx = config.group_stats_xlsx(model_key, target)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_excel(output_xlsx, index=False)
    return GroupStatsRunResult(
        model_key=model_key,
        target_column=target,
        trades=trades,
        grouped=grouped,
        output_xlsx=output_xlsx,
    )


def load_group_stats(path: Path) -> pd.DataFrame:
    """Загружает XLSX group stats и проверяет полный диапазон sentiment."""
    if not Path(path).exists():
        raise ValueError(f"Файл групповой статистики не найден: {path}")
    df = pd.read_excel(path)
    required = {"sentiment", "total_pnl"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"XLSX не содержит обязательные колонки: {sorted(missing)}")

    result = df.copy()
    result["sentiment"] = pd.to_numeric(result["sentiment"], errors="coerce")
    result["total_pnl"] = pd.to_numeric(result["total_pnl"], errors="coerce")
    result = result.dropna(subset=["sentiment", "total_pnl"])
    result["sentiment"] = result["sentiment"].astype(int)

    duplicates = result["sentiment"].duplicated(keep=False)
    if duplicates.any():
        values = sorted(result.loc[duplicates, "sentiment"].unique().tolist())
        raise ValueError(f"В XLSX повторяются значения sentiment: {values}")

    by_sentiment = result.set_index("sentiment")["total_pnl"]
    missing_sentiments = [sentiment for sentiment in SENTIMENT_RANGE if sentiment not in by_sentiment.index]
    if missing_sentiments:
        raise ValueError(f"В XLSX отсутствуют значения sentiment: {missing_sentiments}")
    return result.sort_values("sentiment").reset_index(drop=True)


def recommend_action(total_pnl_by_sentiment: pd.Series, sentiment: int) -> str:
    """Возвращает action по знаку total_pnl, для нуля ищет ближайшего соседа."""
    total_pnl = float(total_pnl_by_sentiment.loc[sentiment])
    if total_pnl > 0:
        return "follow"
    if total_pnl < 0:
        return "invert"

    for distance in range(1, len(SENTIMENT_RANGE)):
        left_value = _neighbor_total_pnl(total_pnl_by_sentiment, sentiment - distance)
        right_value = _neighbor_total_pnl(total_pnl_by_sentiment, sentiment + distance)
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            return _action_from_total_pnl(right_value)
        if right_value is None:
            return _action_from_total_pnl(left_value)
        if abs(left_value) > abs(right_value):
            return _action_from_total_pnl(left_value)
        if abs(right_value) > abs(left_value):
            return _action_from_total_pnl(right_value)

    raise ValueError("Невозможно определить рекомендацию: все значения total_pnl равны 0")


def build_rules_recommendation(grouped: pd.DataFrame) -> list[dict[str, int | str]]:
    """Строит по одному правилу на каждое значение sentiment."""
    prepared = grouped.copy()
    prepared["sentiment"] = pd.to_numeric(prepared["sentiment"], errors="coerce").astype(int)
    prepared["total_pnl"] = pd.to_numeric(prepared["total_pnl"], errors="coerce")
    total_pnl_by_sentiment = prepared.set_index("sentiment")["total_pnl"]
    missing_sentiments = [sentiment for sentiment in SENTIMENT_RANGE if sentiment not in total_pnl_by_sentiment.index]
    if missing_sentiments:
        raise ValueError(f"В group stats отсутствуют значения sentiment: {missing_sentiments}")
    return [
        {
            "min": sentiment,
            "max": sentiment,
            "action": recommend_action(total_pnl_by_sentiment, sentiment),
        }
        for sentiment in SENTIMENT_RANGE
    ]


def render_rules_yaml(
    rules: list[dict[str, int | str]],
    *,
    symbol: str,
    model_key: str,
    target_column: str,
) -> str:
    """Рендерит rules YAML в компактном человекочитаемом формате."""
    lines = [f"rules:  # {symbol} {model_key} target={target_column}"]
    for rule in rules:
        lines.append(
            f"  - {{min: {rule['min']}, max: {rule['max']}, action: {rule['action']}}}"
        )
    return "\n".join(lines) + "\n"


def run_rules_recommendation_for_model(
    config: SentimentResearchConfig,
    model_key: str,
    *,
    target_column: str | None = None,
) -> RulesRunResult:
    """Строит и сохраняет rules YAML для одной модели."""
    target = target_column or config.target_column
    input_xlsx = config.group_stats_xlsx(model_key, target)
    grouped = load_group_stats(input_xlsx)
    rules = build_rules_recommendation(grouped)
    output_yaml = config.rules_yaml(model_key, target)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(
        render_rules_yaml(
            rules,
            symbol=config.symbol,
            model_key=model_key,
            target_column=target,
        ),
        encoding="utf-8",
    )
    return RulesRunResult(
        model_key=model_key,
        target_column=target,
        rules=rules,
        input_xlsx=input_xlsx,
        output_yaml=output_yaml,
    )


def load_rules(path: Path) -> list[dict[str, Any]]:
    """Загружает и валидирует YAML-правила."""
    if not Path(path).exists():
        raise ValueError(f"Rules YAML не найден: {path}")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rules = data.get("rules") or []
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"В {path} нет списка 'rules' или он пустой")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"Правило #{index} должно быть объектом: {rule}")
        for key in ("min", "max", "action"):
            if key not in rule:
                raise ValueError(f"Правило #{index} без поля '{key}': {rule}")
        if rule["action"] not in VALID_ACTIONS:
            raise ValueError(
                f"Правило #{index}: action должен быть одним из {sorted(VALID_ACTIONS)}, "
                f"получено {rule['action']!r}"
            )
        if float(rule["min"]) > float(rule["max"]):
            raise ValueError(f"Правило #{index}: min > max ({rule})")
    return rules


def match_action(sentiment: float, rules: list[dict[str, Any]]) -> str:
    """Возвращает action из первого подходящего правила или skip."""
    for rule in rules:
        if float(rule["min"]) <= sentiment <= float(rule["max"]):
            return str(rule["action"])
    return "skip"


def direction_for_action(sentiment: float, action: str) -> str:
    """Возвращает направление сделки для follow/invert."""
    if action == "follow":
        return "LONG" if sentiment >= 0 else "SHORT"
    if action == "invert":
        return "SHORT" if sentiment >= 0 else "LONG"
    raise ValueError(f"action должен быть follow или invert, получено {action!r}")


def build_backtest(
    df: pd.DataFrame,
    *,
    quantity: int,
    rules: list[dict[str, Any]],
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """Строит backtest-сделки по sentiment, rules и выбранной target-колонке."""
    prepared = prepare_sentiment_frame(df, target_column)
    rows: list[dict[str, Any]] = []
    for row in prepared.to_dict("records"):
        sentiment = float(row["sentiment"])
        target_move = float(row[target_column])
        action = match_action(sentiment, rules)
        if action == "skip":
            continue
        direction = direction_for_action(sentiment, action)
        pnl = target_move * quantity if direction == "LONG" else -target_move * quantity
        rows.append(
            {
                "source_date": row["source_date"],
                "sentiment": sentiment,
                "action": action,
                "direction": direction,
                "target_column": target_column,
                "target_move": target_move,
                "quantity": quantity,
                "pnl": pnl,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("source_date").reset_index(drop=True)
    result["cum_pnl"] = result["pnl"].cumsum()
    return result


def run_backtest_for_model(
    config: SentimentResearchConfig,
    model_key: str,
    *,
    target_column: str | None = None,
    quantity: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    rules_yaml: Path | None = None,
) -> BacktestRunResult:
    """Запускает и сохраняет backtest для одной модели."""
    target = target_column or config.target_column
    selected_quantity = quantity if quantity is not None else config.quantity
    selected_from = date_from if date_from is not None else config.date_from
    selected_to = date_to if date_to is not None else config.date_to
    rules_path = Path(rules_yaml) if rules_yaml is not None else config.rules_yaml(model_key, target)

    source = load_sentiment_scores(config.model_pkl(model_key), target)
    source = filter_by_date(source, date_from=selected_from, date_to=selected_to)
    if source.empty:
        raise ValueError("После фильтра по дате не осталось sentiment-записей")

    rules = load_rules(rules_path)
    result = build_backtest(
        source,
        quantity=selected_quantity,
        rules=rules,
        target_column=target,
    )
    if result.empty:
        raise ValueError("Нет доступных сделок для backtest. Проверьте PKL и rules YAML")

    output_xlsx = config.backtest_xlsx(model_key, target)
    output_html = config.backtest_html(model_key, target)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_xlsx, index=False)
    build_backtest_report_html(
        result,
        symbol=config.symbol,
        model_key=model_key,
        target_column=target,
        rules_yaml=rules_path,
        output_html=output_html,
    )
    return BacktestRunResult(
        model_key=model_key,
        target_column=target,
        result=result,
        rules_yaml=rules_path,
        output_xlsx=output_xlsx,
        output_html=output_html,
    )


def build_backtest_report_html(
    result: pd.DataFrame,
    *,
    symbol: str,
    model_key: str,
    target_column: str,
    rules_yaml: Path,
    output_html: Path,
) -> None:
    """Сохраняет подробный Plotly HTML-отчет по результатам backtest."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = result.copy()
    df["source_date"] = pd.to_datetime(df["source_date"])
    df = df.sort_values("source_date").reset_index(drop=True)

    pnl = df["pnl"].astype(float)
    cum = pnl.cumsum()
    day_colors = ["#d32f2f" if value < 0 else "#2e7d32" for value in pnl]

    df["Неделя"] = df["source_date"].dt.to_period("W")
    weekly = df.groupby("Неделя", as_index=False)["pnl"].sum()
    weekly["dt"] = weekly["Неделя"].apply(lambda period: period.start_time)
    week_colors = ["#d32f2f" if value < 0 else "#00838f" for value in weekly["pnl"]]

    df["Месяц"] = df["source_date"].dt.to_period("M")
    monthly = df.groupby("Месяц", as_index=False)["pnl"].sum()
    monthly["dt"] = monthly["Месяц"].dt.to_timestamp()
    month_colors = ["#d32f2f" if value < 0 else "#1565c0" for value in monthly["pnl"]]

    drawdown = cum - cum.cummax()
    for window in (5, 10, 20):
        df[f"MA{window}"] = pnl.rolling(window, min_periods=1).mean()

    by_sentiment = (
        df.groupby("sentiment")
        .agg(trades=("pnl", "size"), pnl=("pnl", "sum"))
        .reset_index()
        .sort_values("sentiment")
    )
    action_stats = (
        df.groupby("action")
        .agg(
            trades=("pnl", "size"),
            pnl=("pnl", "sum"),
            winrate=("pnl", lambda series: (series > 0).mean() * 100),
        )
        .reset_index()
    )

    total_profit = float(cum.iloc[-1])
    total_trades = len(df)
    win_trades = int((pnl > 0).sum())
    loss_trades = int((pnl < 0).sum())
    win_rate = win_trades / max(total_trades, 1) * 100
    max_dd = float(drawdown.min())
    best_trade = float(pnl.max())
    worst_trade = float(pnl.min())
    avg_trade = float(pnl.mean())
    median_trade = float(pnl.median())
    std_trade = float(pnl.std()) if total_trades > 1 else 0.0

    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(abs(pnl[pnl < 0].sum()))
    avg_win = float(pnl[pnl > 0].mean()) if win_trades else 0.0
    avg_loss = float(abs(pnl[pnl < 0].mean())) if loss_trades else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")
    recovery_factor = total_profit / abs(max_dd) if max_dd != 0 else float("inf")
    expectancy = (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss
    sharpe = (avg_trade / std_trade) * np.sqrt(252) if std_trade > 0 else 0.0

    downside = pnl[pnl < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = (avg_trade / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

    date_range_days = (df["source_date"].max() - df["source_date"].min()).days or 1
    annual_profit = total_profit * 365 / date_range_days
    calmar = annual_profit / abs(max_dd) if max_dd != 0 else float("inf")

    signs = pnl.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    max_consec_wins = _max_consecutive(signs, 1)
    max_consec_losses = _max_consecutive(signs, -1)
    max_dd_duration = _drawdown_duration(drawdown)
    volatility = std_trade * np.sqrt(252)

    stats_text = (
        f"Итого: {total_profit:,.0f} | Сделок: {total_trades} | "
        f"Win: {win_trades} ({win_rate:.0f}%) | Loss: {loss_trades} | "
        f"PF: {profit_factor:.2f} | RF: {recovery_factor:.2f} | "
        f"Sharpe: {sharpe:.2f} | MaxDD: {max_dd:,.0f}"
    )
    test_period_text = (
        "Период тестирования: "
        f"{df['source_date'].min():%Y-%m-%d} - {df['source_date'].max():%Y-%m-%d}"
    )

    fig = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=(
            "P/L по сделкам",
            "Накопленная прибыль (equity)",
            "P/L по неделям",
            "P/L по месяцам",
            "Drawdown от максимума",
            "Распределение P/L сделок",
            "Скользящие средние P/L (5/10/20)",
            "P/L по action (follow/invert)",
            "P/L по значениям sentiment",
            "Кол-во сделок по sentiment",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Bar(
            x=df["source_date"],
            y=pnl,
            marker_color=day_colors,
            name="P/L сделки",
            hovertemplate="%{x|%Y-%m-%d}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["source_date"],
            y=cum,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#2e7d32", width=2),
            fillcolor="rgba(46,125,50,0.15)",
            name="Equity",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=weekly["dt"],
            y=weekly["pnl"],
            marker_color=week_colors,
            name="P/L неделя",
            hovertemplate="Нед. %{x|%Y-%m-%d}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=monthly["dt"],
            y=monthly["pnl"],
            marker_color=month_colors,
            name="P/L месяц",
            text=[f"{value:,.0f}" for value in monthly["pnl"]],
            textposition="outside",
            hovertemplate="%{x|%Y-%m}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df["source_date"],
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#d32f2f", width=1.5),
            fillcolor="rgba(211,47,47,0.2)",
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:,.0f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=pnl[pnl > 0],
            marker_color="#2e7d32",
            opacity=0.7,
            name="Прибыль",
            nbinsx=20,
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=pnl[pnl < 0],
            marker_color="#d32f2f",
            opacity=0.7,
            name="Убыток",
            nbinsx=20,
        ),
        row=3,
        col=2,
    )
    for window, color in [(5, "#1565c0"), (10, "#ff6f00"), (20, "#7b1fa2")]:
        fig.add_trace(
            go.Scatter(
                x=df["source_date"],
                y=df[f"MA{window}"],
                mode="lines",
                line=dict(color=color, width=1.5),
                name=f"MA{window}",
                hovertemplate=f"MA{window}: " + "%{y:,.0f}<extra></extra>",
            ),
            row=4,
            col=1,
        )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)

    action_colors = ["#d32f2f" if value < 0 else "#2e7d32" for value in action_stats["pnl"]]
    fig.add_trace(
        go.Bar(
            x=action_stats["action"],
            y=action_stats["pnl"],
            marker_color=action_colors,
            text=[
                f"{pnl_value:,.0f}<br>{trades} сд.<br>приб. {winrate:.0f}%"
                for pnl_value, trades, winrate in zip(
                    action_stats["pnl"],
                    action_stats["trades"],
                    action_stats["winrate"],
                    strict=False,
                )
            ],
            textposition="outside",
            name="P/L по action",
            hovertemplate="%{x}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=4,
        col=2,
    )

    sentiment_colors = ["#d32f2f" if value < 0 else "#2e7d32" for value in by_sentiment["pnl"]]
    fig.add_trace(
        go.Bar(
            x=by_sentiment["sentiment"],
            y=by_sentiment["pnl"],
            marker_color=sentiment_colors,
            text=[f"{value:,.0f}" for value in by_sentiment["pnl"]],
            textposition="outside",
            name="P/L по sentiment",
            hovertemplate="sentiment: %{x}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=by_sentiment["sentiment"],
            y=by_sentiment["trades"],
            marker_color="#1565c0",
            name="Кол-во сделок",
            hovertemplate="sentiment: %{x}<br>сделок: %{y}<extra></extra>",
        ),
        row=5,
        col=2,
    )

    title = (
        f"{symbol} | {model_key} | target={target_column} | "
        f"бэктест sentiment - правила: {rules_yaml.name}"
    )
    fig.update_layout(
        title_text=f"{title}<br><sub>{test_period_text}</sub><br><sub>{stats_text}</sub>",
        title_x=0.5,
        height=2240,
        width=1500,
        margin=dict(t=140),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    for row, col in [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 1),
        (4, 1),
        (4, 2),
        (5, 1),
        (5, 2),
    ]:
        fig.update_yaxes(tickformat=",", row=row, col=col)

    sec1 = [
        ["<b>ДОХОДНОСТЬ</b>", ""],
        ["Чистая прибыль", f"{total_profit:,.0f}"],
        ["Годовая прибыль (экстрапол.)", f"{annual_profit:,.0f}"],
        ["Средний P/L на сделку", f"{avg_trade:,.0f}"],
        ["Медианный P/L на сделку", f"{median_trade:,.0f}"],
        ["Лучшая сделка", f"{best_trade:,.0f}"],
        ["Худшая сделка", f"{worst_trade:,.0f}"],
    ]
    sec2 = [
        ["<b>РИСК</b>", ""],
        ["Max Drawdown", f"{max_dd:,.0f}"],
        ["Длит. макс. просадки", f"{max_dd_duration} сделок"],
        ["Волатильность (год.)", f"{volatility:,.0f}"],
        ["Std сделки", f"{std_trade:,.0f}"],
        ["VaR 95%", f"{np.percentile(pnl, 5):,.0f}"],
        ["CVaR 95%", f"{pnl[pnl <= np.percentile(pnl, 5)].mean():,.0f}"],
    ]
    sec3 = [
        ["<b>СТАТИСТИКА СДЕЛОК</b>", ""],
        ["Всего сделок", f"{total_trades}"],
        ["Win / Loss", f"{win_trades} / {loss_trades}"],
        ["Win rate", f"{win_rate:.1f}%"],
        ["Ср. выигрыш / проигрыш", f"{avg_win:,.0f} / {avg_loss:,.0f}"],
        ["Макс. серия побед", f"{max_consec_wins}"],
        ["Макс. серия убытков", f"{max_consec_losses}"],
    ]
    num_rows = max(len(sec1), len(sec2), len(sec3))
    for section in (sec1, sec2, sec3):
        while len(section) < num_rows:
            section.append(["", ""])

    cols_values = [[], [], [], [], [], []]
    table_colors = [[], [], []]
    for row_index in range(num_rows):
        for section_index, section in enumerate((sec1, sec2, sec3)):
            name, value = section[row_index]
            is_header = value == "" and name.startswith("<b>")
            cols_values[section_index * 2].append(name)
            cols_values[section_index * 2 + 1].append(
                f"<b>{value}</b>" if value and not is_header else value
            )
            table_colors[section_index].append(
                "#e3f2fd" if is_header else ("#f5f5f5" if row_index % 2 == 0 else "white")
            )

    fig_stats = go.Figure(
        go.Table(
            columnwidth=[200, 130, 180, 120, 220, 120],
            header=dict(
                values=["<b>Показатель</b>", "<b>Значение</b>"] * 3,
                fill_color="#1565c0",
                font=dict(color="white", size=14),
                align="left",
                height=32,
            ),
            cells=dict(
                values=cols_values,
                fill_color=[
                    table_colors[0],
                    table_colors[0],
                    table_colors[1],
                    table_colors[1],
                    table_colors[2],
                    table_colors[2],
                ],
                font=dict(size=13, color="#212121"),
                align=["left", "right", "left", "right", "left", "right"],
                height=26,
            ),
        )
    )
    fig_stats.update_layout(
        title_text=f"<b>{symbol} | {model_key} - Backtest: статистика стратегии</b>",
        title_x=0.5,
        title_font_size=18,
        height=32 + num_rows * 26 + 80,
        width=1500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    coefficients = [
        {
            "name": "Recovery Factor",
            "formula": "Чистая прибыль / |Max Drawdown|",
            "value": f"{recovery_factor:.2f}",
            "description": (
                "Коэффициент восстановления - во сколько раз прибыль превышает "
                "максимальную просадку. RF > 1 - стратегия заработала больше, "
                "чем потеряла в худший период."
            ),
        },
        {
            "name": "Profit Factor",
            "formula": "Валовая прибыль / Валовый убыток",
            "value": f"{profit_factor:.2f}",
            "description": "Фактор прибыли. PF > 1 - прибыльность, 1.5-2.0 хорошо.",
        },
        {
            "name": "Payoff Ratio",
            "formula": "Средний выигрыш / Средний проигрыш",
            "value": f"{payoff_ratio:.2f}",
            "description": "При высоком payoff стратегия прибыльна даже при win rate < 50%.",
        },
        {
            "name": "Sharpe Ratio",
            "formula": "(Ср. P/L / Std) x sqrt(252)",
            "value": f"{sharpe:.2f}",
            "description": "Отношение доходности к риску, приведенное к году.",
        },
        {
            "name": "Sortino Ratio",
            "formula": "(Ср. P/L / Downside Std) x sqrt(252)",
            "value": f"{sortino:.2f}",
            "description": "Модификация Шарпа, учитывающая только нисходящую волатильность.",
        },
        {
            "name": "Calmar Ratio",
            "formula": "Годовая доходность / |Max Drawdown|",
            "value": f"{calmar:.2f}",
            "description": "Отношение годовой прибыли к максимальной просадке.",
        },
        {
            "name": "Expectancy",
            "formula": "Win% x Ср.выигрыш - Loss% x Ср.проигрыш",
            "value": f"{expectancy:,.0f}",
            "description": "Матожидание на одну сделку. Положительное значение - edge.",
        },
    ]
    fig_table = go.Figure(
        go.Table(
            columnwidth=[150, 250, 80, 450],
            header=dict(
                values=[
                    "<b>Коэффициент</b>",
                    "<b>Формула</b>",
                    "<b>Значение</b>",
                    "<b>Расшифровка</b>",
                ],
                fill_color="#1565c0",
                font=dict(color="white", size=14),
                align="left",
                height=36,
            ),
            cells=dict(
                values=[
                    [f"<b>{coefficient['name']}</b>" for coefficient in coefficients],
                    [coefficient["formula"] for coefficient in coefficients],
                    [f"<b>{coefficient['value']}</b>" for coefficient in coefficients],
                    [coefficient["description"] for coefficient in coefficients],
                ],
                fill_color=[
                    ["#f5f5f5" if index % 2 == 0 else "white" for index in range(len(coefficients))]
                ]
                * 4,
                font=dict(size=13, color="#212121"),
                align=["left", "left", "center", "left"],
                height=60,
            ),
        )
    )
    fig_table.update_layout(
        title_text=f"<b>{symbol} | {model_key} - Backtest: ключевые коэффициенты</b>",
        title_x=0.5,
        title_font_size=18,
        height=560,
        width=1500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    forecast_html = build_next_month_forecast_html(result)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    with output_html.open("w", encoding="utf-8") as file_obj:
        file_obj.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
        file_obj.write(
            f"<title>{escape(symbol)} | {escape(model_key)} | sentiment backtest</title>\n"
        )
        file_obj.write("</head><body>\n")
        file_obj.write(fig.to_html(include_plotlyjs="cdn", full_html=False))
        file_obj.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        file_obj.write(fig_stats.to_html(include_plotlyjs=False, full_html=False))
        file_obj.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        file_obj.write(fig_table.to_html(include_plotlyjs=False, full_html=False))
        if forecast_html:
            file_obj.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
            file_obj.write(forecast_html)
        file_obj.write("\n</body></html>\n")


def build_next_month_forecast_html(
    result: pd.DataFrame,
    forecast_days: int = 21,
    bootstrap_samples: int = 200_000,
) -> str:
    """Строит HTML-блок с прогнозным распределением PnL на следующий месяц."""
    pnl = pd.to_numeric(result["pnl"], errors="coerce").dropna().astype(float)
    if len(pnl) < 2:
        return ""

    avg_daily = float(pnl.mean())
    std_daily = float(pnl.std(ddof=1))
    mean_month = avg_daily * forecast_days
    sigma_month = std_daily * np.sqrt(forecast_days)
    normal_rows = _forecast_interval_rows(mean_month, sigma_month)

    rng = np.random.default_rng(42)
    bootstrap = rng.choice(
        pnl.to_numpy(),
        size=(bootstrap_samples, forecast_days),
        replace=True,
    ).sum(axis=1)
    bootstrap_specs = [
        ("50%", 25, 75),
        ("68%", 16, 84),
        ("80%", 10, 90),
        ("90%", 5, 95),
        ("95%", 2.5, 97.5),
        ("99%", 0.5, 99.5),
    ]
    bootstrap_rows = [
        {
            "probability": probability,
            "low": float(np.percentile(bootstrap, low_pct)),
            "high": float(np.percentile(bootstrap, high_pct)),
        }
        for probability, low_pct, high_pct in bootstrap_specs
    ]

    threshold_rows = [
        ("P(PnL <= -10 000)", float((bootstrap <= -10_000).mean() * 100)),
        ("P(PnL <= -5 000)", float((bootstrap <= -5_000).mean() * 100)),
        ("P(PnL <= 0)", float((bootstrap <= 0).mean() * 100)),
        ("Вероятность прибыли, P(PnL > 0)", float((bootstrap > 0).mean() * 100)),
        ("P(PnL >= 5 000)", float((bootstrap >= 5_000).mean() * 100)),
        ("P(PnL >= 10 000)", float((bootstrap >= 10_000).mean() * 100)),
        ("P(PnL >= 20 000)", float((bootstrap >= 20_000).mean() * 100)),
    ]

    normal_rows_html = "\n".join(
        f'<tr style="{_row_style(index)}">'
        f"<td>{row['probability']}</td>"
        f"<td>{_fmt_num(row['low'])} ... {_fmt_num(row['high'])}</td>"
        "</tr>"
        for index, row in enumerate(normal_rows)
    )
    bootstrap_rows_html = "\n".join(
        f'<tr style="{_row_style(index)}">'
        f"<td>{row['probability']}</td>"
        f"<td>{_fmt_num(row['low'])} ... {_fmt_num(row['high'])}</td>"
        "</tr>"
        for index, row in enumerate(bootstrap_rows)
    )
    threshold_rows_html = "\n".join(
        f'<tr style="{_row_style(index)}">'
        f"<td>{escape(label, quote=False)}</td><td>{value:.1f}%</td></tr>"
        for index, (label, value) in enumerate(threshold_rows)
    )

    return f"""
<section id="next-month-forecast" style="width:1450px; margin:32px auto 44px auto; font-family:Arial, sans-serif; color:#212121;">
  <h2 style="text-align:center; margin:0 0 8px 0;">Прогноз на следующий месяц</h2>
  <p style="text-align:center; margin:0 0 22px 0; color:#555;">
    Оценка распределения PnL на {forecast_days} будущих сигналов/дней по историческому ряду сделок.
  </p>
  <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:18px; align-items:start;">
    <div style="border:1px solid #ddd; border-radius:6px; padding:16px;">
      <h3 style="margin:0 0 12px 0; font-size:18px;">Базовые параметры</h3>
      <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <tr style="{_row_style(0)}"><td>Наблюдений</td><td style="text-align:right;"><b>{len(pnl)}</b></td></tr>
        <tr style="{_row_style(1)}"><td>Средний дневной PnL</td><td style="text-align:right;"><b>{_fmt_num(avg_daily)}</b></td></tr>
        <tr style="{_row_style(2)}"><td>Дневная sigma</td><td style="text-align:right;"><b>{_fmt_num(std_daily)}</b></td></tr>
        <tr style="{_row_style(3)}"><td>Ожидаемый PnL месяца</td><td style="text-align:right;"><b>{_fmt_num(mean_month)}</b></td></tr>
        <tr style="{_row_style(4)}"><td>Месячная sigma</td><td style="text-align:right;"><b>{_fmt_num(sigma_month)}</b></td></tr>
      </table>
      <p style="font-size:13px; color:#555; line-height:1.4;">
        Нормальная модель: mean +/- z * sigma, где месячная sigma = дневная sigma * sqrt(N).
      </p>
    </div>
    <div style="border:1px solid #ddd; border-radius:6px; padding:16px;">
      <h3 style="margin:0 0 12px 0; font-size:18px;">Нормальные интервалы</h3>
      <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <tr style="background:#1565c0; color:white;"><th style="text-align:left; padding:7px;">Вероятность</th><th style="text-align:right; padding:7px;">Диапазон PnL</th></tr>
        {normal_rows_html}
      </table>
    </div>
    <div style="border:1px solid #ddd; border-radius:6px; padding:16px;">
      <h3 style="margin:0 0 12px 0; font-size:18px;">Бутстрэп</h3>
      <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <tr style="background:#1565c0; color:white;"><th style="text-align:left; padding:7px;">Вероятность</th><th style="text-align:right; padding:7px;">Диапазон PnL</th></tr>
        {bootstrap_rows_html}
      </table>
    </div>
  </div>
  <div style="border:1px solid #ddd; border-radius:6px; padding:16px; margin-top:18px;">
    <h3 style="margin:0 0 12px 0; font-size:18px;">Вероятности порогов по бутстрэпу</h3>
    <table style="width:100%; border-collapse:collapse; font-size:14px;">
      <tr style="background:#1565c0; color:white;"><th style="text-align:left; padding:7px;">Событие</th><th style="text-align:right; padding:7px;">Вероятность</th></tr>
      {threshold_rows_html}
    </table>
    <p style="font-size:13px; color:#555; line-height:1.4;">
      Это не прогноз рынка, а статистическая оценка следующего месяца при условии, что будущие сделки похожи на исторический бэктест.
    </p>
  </div>
</section>
"""


def _fmt_num(value: float) -> str:
    """Форматирует число с пробелами между тысячами для HTML-отчета."""
    return f"{value:,.0f}".replace(",", " ")


def _forecast_interval_rows(mean_month: float, sigma_month: float) -> list[dict[str, str | float]]:
    """Возвращает нормальные прогнозные интервалы для месячного PnL."""
    z_by_probability = [
        ("50%", 0.67448975),
        ("68%", 1.0),
        ("80%", 1.28155156),
        ("90%", 1.64485363),
        ("95%", 1.95996398),
        ("99%", 2.5758293),
    ]
    return [
        {
            "probability": probability,
            "low": mean_month - z_value * sigma_month,
            "high": mean_month + z_value * sigma_month,
        }
        for probability, z_value in z_by_probability
    ]


def _row_style(index: int) -> str:
    """Возвращает фон строки для чередования в HTML-таблицах."""
    color = "#f7f7f7" if index % 2 == 0 else "#ffffff"
    return f"background:{color};"


def _max_consecutive(series: pd.Series, condition: int) -> int:
    """Возвращает максимальную длину серии значений, равных condition."""
    streaks = (series != condition).cumsum()
    filtered = series[series == condition]
    if filtered.empty:
        return 0
    return int(filtered.groupby(streaks[series == condition]).size().max())


def _drawdown_duration(drawdown: pd.Series) -> int:
    """Вычисляет максимальную длительность просадки в количестве сделок."""
    max_duration = 0
    current_start = None
    for index in range(len(drawdown)):
        if drawdown.iloc[index] < 0:
            if current_start is None:
                current_start = index
        elif current_start is not None:
            max_duration = max(max_duration, index - current_start)
            current_start = None
    if current_start is not None:
        max_duration = max(max_duration, len(drawdown) - current_start)
    return max_duration


def _neighbor_total_pnl(total_pnl_by_sentiment: pd.Series, sentiment: int) -> float | None:
    if sentiment not in total_pnl_by_sentiment.index:
        return None
    value = float(total_pnl_by_sentiment.loc[sentiment])
    return value if value != 0 else None


def _action_from_total_pnl(total_pnl: float) -> str:
    return "follow" if total_pnl > 0 else "invert"


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise SettingsError(f"Некорректная дата в reports: {value!r}")
    return parsed.date()


def _validate_target_column(target_column: str) -> None:
    if target_column not in VALID_TARGET_COLUMNS:
        raise ValueError(
            f"target_column должен быть одним из {sorted(VALID_TARGET_COLUMNS)}, "
            f"получено {target_column!r}"
        )


def _validate_target_column_for_settings(target_column: str) -> None:
    try:
        _validate_target_column(target_column)
    except ValueError as exc:
        raise SettingsError(str(exc)) from exc


def _section(settings: dict[str, Any], key: str) -> dict[str, Any]:
    value = settings.get(key, {})
    if not isinstance(value, dict):
        raise SettingsError(f"settings.yaml section {key!r} must be a mapping")
    return value


def _resolve_settings_path(settings_path: Path, value: Any) -> Path:
    resolved = Path(str(value))
    if not resolved.is_absolute():
        resolved = settings_path.parent / resolved
    return resolved
