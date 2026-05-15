"""Daily walk-forward backtest for sentiment strategies.

The module uses the existing `pj22_btc.sentiment_research` primitives and adds
only the rolling training/test split plus artifact layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pj22_btc.mexc_downloader import SettingsError
from pj22_btc.sentiment_research import (
    VALID_TARGET_COLUMNS,
    build_backtest,
    build_follow_trades,
    build_rules_recommendation,
    group_by_sentiment,
    load_sentiment_research_config,
    load_sentiment_scores,
)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Settings for walk-forward runs loaded from project `settings.yaml`."""

    symbol: str
    output_dir: Path
    quantity: int
    target_column: str
    model_pkls: dict[str, Path]
    enabled_model_keys: list[str]
    backtest_start_date: date | None
    backtest_end_date: date | None
    train_months: int
    save_daily_artifacts: bool
    min_train_rows: int
    keep_going: bool

    def selected_model_keys(self, model_keys: list[str] | None) -> list[str]:
        """Return explicit model keys or enabled models by default."""
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
        """Return sentiment PKL path for a configured model."""
        try:
            return self.model_pkls[model_key]
        except KeyError as exc:
            available = ", ".join(sorted(self.model_pkls))
            raise SettingsError(f"Unknown sentiment model {model_key!r}. Available: {available}") from exc

    def model_output_dir(self, model_key: str, target_column: str | None = None) -> Path:
        """Return output directory for a model/target pair."""
        target = target_column or self.target_column
        return self.output_dir / self.symbol / model_key / target


@dataclass(frozen=True)
class WalkForwardDayResult:
    """Result for one rolling test date."""

    summary: dict[str, Any]
    trade: dict[str, Any] | None
    grouped: pd.DataFrame | None
    rules: list[dict[str, Any]] | None


@dataclass(frozen=True)
class WalkForwardModelResult:
    """Result for one model over all selected walk-forward dates."""

    daily_summaries: list[dict[str, Any]]
    trades: pd.DataFrame
    model_summary: dict[str, Any]
    daily_artifacts: dict[date, WalkForwardDayResult]


def load_walk_forward_config(path: Path) -> WalkForwardConfig:
    """Read walk-forward settings from the project settings file."""
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    walk_forward = _section(raw, "walk_forward")
    research = load_sentiment_research_config(settings_path)

    output_dir = _resolve_settings_path(
        settings_path,
        walk_forward.get("output_dir", "reports/walk_forward"),
    )
    target_column = str(walk_forward.get("target_column", research.target_column))
    _validate_target_column_for_settings(target_column)

    backtest_start_date = _parse_date(
        walk_forward.get(
            "backtest_start_date",
            walk_forward.get("start_date", research.date_from),
        )
    )
    backtest_end_date = _parse_date(
        walk_forward.get(
            "backtest_end_date",
            walk_forward.get("end_date", research.date_to),
        )
    )

    train_months = int(walk_forward.get("train_months", 6))
    if train_months < 1:
        raise SettingsError("walk_forward.train_months must be >= 1")
    min_train_rows = int(walk_forward.get("min_train_rows", 20))
    if min_train_rows < 1:
        raise SettingsError("walk_forward.min_train_rows must be >= 1")

    return WalkForwardConfig(
        symbol=research.symbol,
        output_dir=output_dir,
        quantity=int(walk_forward.get("quantity", research.quantity)),
        target_column=target_column,
        model_pkls=research.model_pkls,
        enabled_model_keys=research.enabled_model_keys,
        backtest_start_date=backtest_start_date,
        backtest_end_date=backtest_end_date,
        train_months=train_months,
        save_daily_artifacts=bool(walk_forward.get("save_daily_artifacts", False)),
        min_train_rows=min_train_rows,
        keep_going=bool(walk_forward.get("keep_going", True)),
    )


def training_window_for(test_date: date, train_months: int) -> tuple[date, date]:
    """Return the inclusive training window ending one day before `test_date`."""
    if train_months < 1:
        raise ValueError("train_months должен быть >= 1")
    start = (pd.Timestamp(test_date) - pd.DateOffset(months=train_months)).date()
    end = test_date - timedelta(days=1)
    return start, end


def split_walk_forward_day(
    source: pd.DataFrame,
    *,
    target_column: str,
    test_date: date,
    train_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split prepared sentiment rows into train rows and one test date."""
    prepared = _prepare_source(source, target_column)
    train_start, train_end = training_window_for(test_date, train_months)
    train_mask = (prepared["source_date"] >= train_start) & (prepared["source_date"] <= train_end)
    test_mask = prepared["source_date"] == test_date
    return prepared.loc[train_mask].copy(), prepared.loc[test_mask].copy()


def iter_test_dates(
    source: pd.DataFrame,
    *,
    target_column: str,
    start_date: date | None,
    end_date: date | None,
) -> list[date]:
    """Return sorted source dates inside the requested test period."""
    prepared = _prepare_source(source, target_column)
    if prepared.empty:
        return []
    first_date = min(prepared["source_date"])
    last_date = max(prepared["source_date"])
    effective_start = start_date or first_date
    effective_end = end_date or last_date
    return [
        source_date
        for source_date in prepared["source_date"].tolist()
        if effective_start <= source_date <= effective_end
    ]


def run_walk_forward_day(
    source: pd.DataFrame,
    *,
    symbol: str,
    model_key: str,
    quantity: int,
    target_column: str,
    test_date: date,
    train_months: int,
    min_train_rows: int,
) -> WalkForwardDayResult:
    """Train rules on the lookback window and test them on one date."""
    train_start, train_end = training_window_for(test_date, train_months)
    train, test = split_walk_forward_day(
        source,
        target_column=target_column,
        test_date=test_date,
        train_months=train_months,
    )
    summary = _base_summary(
        symbol=symbol,
        model_key=model_key,
        target_column=target_column,
        test_date=test_date,
        train_start=train_start,
        train_end=train_end,
        train_rows=len(train),
        test_rows=len(test),
    )

    if test.empty:
        summary["status"] = "skipped"
        summary["skip_reason"] = "no_test_row"
        return WalkForwardDayResult(summary, None, None, None)
    if len(train) < min_train_rows:
        summary["status"] = "skipped"
        summary["skip_reason"] = "insufficient_train_rows"
        return WalkForwardDayResult(summary, None, None, None)

    try:
        grouped = group_by_sentiment(
            build_follow_trades(
                train,
                quantity=quantity,
                target_column=target_column,
            )
        )
        rules = build_rules_recommendation(grouped)
        test_result = build_backtest(
            test,
            quantity=quantity,
            rules=rules,
            target_column=target_column,
        )
    except Exception as exc:
        summary["status"] = "skipped"
        summary["skip_reason"] = "rules_unavailable"
        summary["error"] = str(exc)
        return WalkForwardDayResult(summary, None, None, None)

    if test_result.empty:
        summary["status"] = "skipped"
        summary["skip_reason"] = "no_trade"
        return WalkForwardDayResult(summary, None, grouped, rules)

    trade = test_result.iloc[0].to_dict()
    summary["status"] = "ok"
    summary["trades"] = 1
    summary["pnl"] = float(trade["pnl"])
    return WalkForwardDayResult(summary, trade, grouped, rules)


def run_walk_forward_model(
    source: pd.DataFrame,
    *,
    symbol: str,
    model_key: str,
    quantity: int,
    target_column: str,
    start_date: date | None,
    end_date: date | None,
    train_months: int,
    min_train_rows: int,
) -> WalkForwardModelResult:
    """Run daily walk-forward for one sentiment model."""
    prepared = _prepare_source(source, target_column)
    daily_summaries: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    daily_artifacts: dict[date, WalkForwardDayResult] = {}

    for test_date in iter_test_dates(
        prepared,
        target_column=target_column,
        start_date=start_date,
        end_date=end_date,
    ):
        day = run_walk_forward_day(
            prepared,
            symbol=symbol,
            model_key=model_key,
            quantity=quantity,
            target_column=target_column,
            test_date=test_date,
            train_months=train_months,
            min_train_rows=min_train_rows,
        )
        daily_summaries.append(day.summary)
        daily_artifacts[test_date] = day
        if day.trade is not None:
            row = dict(day.trade)
            row["symbol"] = symbol
            row["model_key"] = model_key
            row["target_column"] = target_column
            row["train_start"] = day.summary["train_start"]
            row["train_end"] = day.summary["train_end"]
            row["train_rows"] = day.summary["train_rows"]
            trade_rows.append(row)

    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        trades = trades.sort_values("source_date").reset_index(drop=True)
        trades["cum_pnl"] = trades["pnl"].cumsum()

    model_summary = summarize_model(
        symbol=symbol,
        model_key=model_key,
        target_column=target_column,
        daily_summaries=daily_summaries,
        trades=trades,
    )
    return WalkForwardModelResult(daily_summaries, trades, model_summary, daily_artifacts)


def run_walk_forward_for_model(
    config: WalkForwardConfig,
    model_key: str,
    *,
    target_column: str | None = None,
    quantity: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    train_months: int | None = None,
    min_train_rows: int | None = None,
) -> WalkForwardModelResult:
    """Load one model PKL and run walk-forward using config defaults."""
    target = target_column or config.target_column
    source = load_sentiment_scores(config.model_pkl(model_key), target)
    return run_walk_forward_model(
        source,
        symbol=config.symbol,
        model_key=model_key,
        quantity=quantity if quantity is not None else config.quantity,
        target_column=target,
        start_date=start_date if start_date is not None else config.backtest_start_date,
        end_date=end_date if end_date is not None else config.backtest_end_date,
        train_months=train_months if train_months is not None else config.train_months,
        min_train_rows=min_train_rows if min_train_rows is not None else config.min_train_rows,
    )


def summarize_model(
    *,
    symbol: str,
    model_key: str,
    target_column: str,
    daily_summaries: list[dict[str, Any]],
    trades: pd.DataFrame,
) -> dict[str, Any]:
    """Build one-row model summary for saved reports."""
    ok_days = sum(1 for row in daily_summaries if row["status"] == "ok")
    skipped_days = sum(1 for row in daily_summaries if row["status"] == "skipped")
    error_days = sum(1 for row in daily_summaries if row["status"] == "error")
    total_pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
    winrate = float((trades["pnl"] > 0).mean() * 100) if not trades.empty else 0.0
    max_drawdown = 0.0
    if not trades.empty:
        equity = trades["pnl"].cumsum()
        max_drawdown = float((equity - equity.cummax()).min())
    return {
        "symbol": symbol,
        "model_key": model_key,
        "target_column": target_column,
        "status": "ok" if error_days == 0 else "error",
        "days": len(daily_summaries),
        "ok_days": ok_days,
        "skipped_days": skipped_days,
        "error_days": error_days,
        "trades": int(len(trades)),
        "total_pnl": total_pnl,
        "winrate": winrate,
        "max_drawdown": max_drawdown,
    }


def save_model_outputs(
    *,
    output_dir: Path,
    symbol: str,
    model_key: str,
    target_column: str,
    result: WalkForwardModelResult,
    save_daily_artifacts: bool,
) -> Path:
    """Save per-model walk-forward artifacts and return the target directory."""
    target = Path(output_dir) / symbol / model_key / target_column
    target.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(target / "trades.csv", index=False, encoding="utf-8-sig")
    result.trades.to_excel(target / "trades.xlsx", index=False)
    pd.DataFrame(result.daily_summaries).to_csv(
        target / "daily_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(result.daily_summaries).to_excel(target / "daily_summary.xlsx", index=False)
    (target / "summary.json").write_text(
        json.dumps(result.model_summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    if save_daily_artifacts:
        for test_date, day in result.daily_artifacts.items():
            if day.grouped is None or day.rules is None:
                continue
            daily_dir = target / "daily" / test_date.isoformat()
            daily_dir.mkdir(parents=True, exist_ok=True)
            day.grouped.to_excel(daily_dir / "group_stats.xlsx", index=False)
            (daily_dir / "rules.yaml").write_text(
                render_day_rules_yaml(
                    day.rules,
                    symbol=symbol,
                    model_key=model_key,
                    target_column=target_column,
                    test_date=test_date,
                    train_start=day.summary["train_start"],
                    train_end=day.summary["train_end"],
                ),
                encoding="utf-8",
            )
    return target


def save_global_summary(
    *,
    output_dir: Path,
    target_column: str,
    daily_summaries: list[dict[str, Any]],
    model_summaries: list[dict[str, Any]],
) -> None:
    """Save global daily and per-model summaries for a target column."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    suffix = _safe_name(target_column)
    daily = pd.DataFrame(daily_summaries)
    models = pd.DataFrame(model_summaries)
    daily.to_csv(target / f"summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    daily.to_excel(target / f"summary_{suffix}.xlsx", index=False)
    models.to_csv(target / f"model_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    models.to_excel(target / f"model_summary_{suffix}.xlsx", index=False)


def render_day_rules_yaml(
    rules: list[dict[str, Any]],
    *,
    symbol: str,
    model_key: str,
    target_column: str,
    test_date: date,
    train_start: date,
    train_end: date,
) -> str:
    """Render rules used for one walk-forward day."""
    lines = [
        (
            f"rules:  # WF {symbol} {model_key} target={target_column} "
            f"test_date={test_date} train={train_start}..{train_end}"
        )
    ]
    for rule in rules:
        lines.append(
            f"  - {{min: {rule['min']}, max: {rule['max']}, action: {rule['action']}}}"
        )
    return "\n".join(lines) + "\n"


def error_summary(
    *,
    symbol: str,
    model_key: str,
    target_column: str,
    error: Exception,
) -> dict[str, Any]:
    """Build a global summary row for a model-level error."""
    return {
        "symbol": symbol,
        "model_key": model_key,
        "target_column": target_column,
        "source_date": "",
        "train_start": "",
        "train_end": "",
        "train_rows": 0,
        "test_rows": 0,
        "status": "error",
        "skip_reason": "",
        "error": str(error),
        "trades": 0,
        "pnl": 0.0,
    }


def _prepare_source(source: pd.DataFrame, target_column: str) -> pd.DataFrame:
    prepared = source.copy()
    if "source_date" not in prepared.columns and prepared.index.name == "source_date":
        prepared = prepared.reset_index()
    return load_or_prepare_frame(prepared, target_column)


def load_or_prepare_frame(source: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Validate an in-memory sentiment frame for walk-forward calculations."""
    _validate_target_column_for_value_error(target_column)
    from pj22_btc.sentiment_research import prepare_sentiment_frame

    return prepare_sentiment_frame(source, target_column)


def _base_summary(
    *,
    symbol: str,
    model_key: str,
    target_column: str,
    test_date: date,
    train_start: date,
    train_end: date,
    train_rows: int,
    test_rows: int,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "model_key": model_key,
        "target_column": target_column,
        "source_date": test_date,
        "train_start": train_start,
        "train_end": train_end,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "status": "",
        "skip_reason": "",
        "error": "",
        "trades": 0,
        "pnl": 0.0,
    }


def _json_default(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise SettingsError(f"Некорректная дата в walk_forward: {value!r}")
    return parsed.date()


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


def _validate_target_column_for_settings(target_column: str) -> None:
    if target_column not in VALID_TARGET_COLUMNS:
        raise SettingsError(
            f"target_column должен быть одним из {sorted(VALID_TARGET_COLUMNS)}, "
            f"получено {target_column!r}"
        )


def _validate_target_column_for_value_error(target_column: str) -> None:
    if target_column not in VALID_TARGET_COLUMNS:
        raise ValueError(
            f"target_column должен быть одним из {sorted(VALID_TARGET_COLUMNS)}, "
            f"получено {target_column!r}"
        )


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
