"""Compare saved ordinary backtest and walk-forward backtest artifacts."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from html import escape
from numbers import Number
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from pj22_btc.html_reports import DEFAULT_CHROME_PATH, build_chrome_command


TARGET_COLUMNS = ("next_body", "next_open_to_open")
DEFAULT_OUTPUT_HTML = Path("reports/backtest_comparison/backtest_vs_walk_forward.html")
REQUIRED_COLUMNS = {"source_date", "sentiment", "action", "direction", "quantity", "pnl"}


@dataclass(frozen=True)
class ComparisonPair:
    """Paths for one model and target-column comparison."""

    symbol: str
    model_key: str
    target_column: str
    ordinary_path: Path
    walk_path: Path


@dataclass
class PairComparison:
    """Prepared aligned ordinary and walk-forward comparison."""

    pair: ComparisonPair
    ordinary: pd.DataFrame
    walk: pd.DataFrame
    metrics: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class ReportResult:
    """Result metadata after writing a comparison report."""

    output_html: Path
    comparisons: list[PairComparison]
    errors: list[dict[str, str]]


def discover_pairs(
    *,
    symbol: str,
    reports_dir: Path,
    walk_forward_dir: Path,
    model_keys: list[str],
    target_columns: list[str] | tuple[str, ...] = TARGET_COLUMNS,
) -> list[ComparisonPair]:
    """Build expected ordinary/WF artifact paths for selected models and targets."""
    pairs: list[ComparisonPair] = []
    for model_key in model_keys:
        for target_column in target_columns:
            pairs.append(
                ComparisonPair(
                    symbol=symbol,
                    model_key=model_key,
                    target_column=target_column,
                    ordinary_path=(
                        Path(reports_dir)
                        / model_key
                        / "backtest"
                        / f"sentiment_backtest_{target_column}_results.xlsx"
                    ),
                    walk_path=(
                        Path(walk_forward_dir)
                        / symbol
                        / model_key
                        / target_column
                        / "trades.xlsx"
                    ),
                )
            )
    return pairs


def normalize_trades(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a saved trades table."""
    result = frame.copy()
    missing = REQUIRED_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"Нет обязательных колонок: {sorted(missing)}")

    result["source_date"] = pd.to_datetime(result["source_date"], errors="coerce").dt.date
    for column in ("sentiment", "quantity", "pnl"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["action"] = result["action"].fillna("").astype(str)
    result["direction"] = result["direction"].fillna("").astype(str)
    result = result.dropna(subset=["source_date", "pnl"]).sort_values("source_date")
    return result.reset_index(drop=True)


def prepare_comparison(
    *,
    pair: ComparisonPair,
    ordinary: pd.DataFrame,
    walk: pd.DataFrame,
) -> PairComparison:
    """Align ordinary and walk-forward trades on shared dates and calculate metrics."""
    ordinary_norm = normalize_trades(ordinary)
    walk_norm = normalize_trades(walk)
    merged = ordinary_norm.merge(
        walk_norm,
        on="source_date",
        how="inner",
        suffixes=("_ordinary", "_walk"),
    ).sort_values("source_date")

    if merged.empty:
        return PairComparison(pair, pd.DataFrame(), pd.DataFrame(), {}, "Нет пересекающихся дат")

    ordinary_overlap = _frame_from_merged(merged, "ordinary")
    walk_overlap = _frame_from_merged(merged, "walk")
    ordinary_overlap["cum_pnl"] = ordinary_overlap["pnl"].cumsum()
    walk_overlap["cum_pnl"] = walk_overlap["pnl"].cumsum()

    ordinary_total_pnl = float(ordinary_overlap["pnl"].sum())
    walk_total_pnl = float(walk_overlap["pnl"].sum())
    metrics = {
        "symbol": pair.symbol,
        "model_key": pair.model_key,
        "target_column": pair.target_column,
        "start_date": ordinary_overlap["source_date"].iloc[0],
        "end_date": ordinary_overlap["source_date"].iloc[-1],
        "overlap_rows": int(len(ordinary_overlap)),
        "ordinary_total_pnl": ordinary_total_pnl,
        "walk_total_pnl": walk_total_pnl,
        "delta_pnl": float(walk_total_pnl - ordinary_total_pnl),
        "ordinary_max_drawdown": _max_drawdown(ordinary_overlap["cum_pnl"]),
        "walk_max_drawdown": _max_drawdown(walk_overlap["cum_pnl"]),
        "ordinary_win_rate": _win_rate(ordinary_overlap["pnl"]),
        "walk_win_rate": _win_rate(walk_overlap["pnl"]),
        "signal_match_rate": _signal_match_rate(ordinary_overlap, walk_overlap),
    }
    return PairComparison(pair, ordinary_overlap, walk_overlap, metrics)


def build_report(
    *,
    symbol: str,
    reports_dir: Path,
    walk_forward_dir: Path,
    output_html: Path,
    model_keys: list[str],
    target_columns: list[str] | tuple[str, ...] = TARGET_COLUMNS,
) -> ReportResult:
    """Write the HTML comparison report and return its metadata."""
    comparisons: list[PairComparison] = []
    errors: list[dict[str, str]] = []
    pairs = discover_pairs(
        symbol=symbol,
        reports_dir=reports_dir,
        walk_forward_dir=walk_forward_dir,
        model_keys=model_keys,
        target_columns=target_columns,
    )

    for pair in pairs:
        if not pair.ordinary_path.exists():
            errors.append(_error_row(pair, f"Не найден ordinary backtest: {pair.ordinary_path}"))
            continue
        if not pair.walk_path.exists():
            errors.append(_error_row(pair, f"Не найден walk-forward trades: {pair.walk_path}"))
            continue
        try:
            comparison = prepare_comparison(
                pair=pair,
                ordinary=pd.read_excel(pair.ordinary_path),
                walk=pd.read_excel(pair.walk_path),
            )
        except Exception as exc:
            errors.append(_error_row(pair, str(exc)))
            continue
        if comparison.error is not None:
            errors.append(_error_row(pair, comparison.error))
            continue
        comparisons.append(comparison)

    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_html(comparisons=comparisons, errors=errors, target_columns=target_columns),
        encoding="utf-8",
    )
    return ReportResult(output_html=output, comparisons=comparisons, errors=errors)


def build_html(
    *,
    comparisons: list[PairComparison],
    errors: list[dict[str, str]],
    target_columns: list[str] | tuple[str, ...] = TARGET_COLUMNS,
) -> str:
    """Render a self-contained HTML report body."""
    metrics_rows = [item.metrics for item in comparisons]
    sections: list[str] = []
    include_plotlyjs = True

    for target_column in target_columns:
        target_comparisons = [
            item for item in comparisons if item.pair.target_column == target_column
        ]
        pair_blocks: list[str] = []
        for comparison in sorted(target_comparisons, key=lambda item: item.pair.model_key):
            label = f"{comparison.pair.symbol} / {comparison.pair.model_key} / {target_column}"
            pair_blocks.append(
                f"""
                <div class="pair">
                  <h3>{escape(label)}</h3>
                  <div class="grid">
                    <div>{_plot_div(_equity_figure(comparison), include_plotlyjs=include_plotlyjs)}</div>
                    <div>{_plot_div(_drawdown_figure(comparison), include_plotlyjs=False)}</div>
                  </div>
                  {_metrics_table(comparison.metrics)}
                </div>
                """
            )
            include_plotlyjs = False
        if not pair_blocks:
            pair_blocks.append("<p class='muted'>Нет сопоставимых пар для этого target.</p>")
        sections.append(
            f"""
            <section>
              <h2>{escape(target_column)}</h2>
              {''.join(pair_blocks)}
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Сравнение ordinary backtest и walk-forward</title>
  {_style()}
</head>
<body>
  <main>
    <h1>Сравнение ordinary backtest и walk-forward</h1>
    <p class="muted">Сравнение строится только на датах, которые есть в обоих источниках для пары model/target.</p>
    <section>
      <h2>Сводка</h2>
      {_summary_table(metrics_rows)}
    </section>
    {''.join(sections)}
    <section>
      <h2>Ошибки и пропуски</h2>
      {_errors_table(errors)}
    </section>
  </main>
</body>
</html>"""


def open_report_in_chrome(
    html_path: Path,
    *,
    chrome_path: Path = DEFAULT_CHROME_PATH,
    popen: Callable[[list[str]], object] = subprocess.Popen,
) -> None:
    """Open one HTML report in a new Chrome window."""
    chrome = Path(chrome_path)
    report = Path(html_path)
    if not chrome.exists():
        raise FileNotFoundError(f"Google Chrome не найден: {chrome}")
    if not report.exists():
        raise FileNotFoundError(f"HTML-отчет не найден: {report}")
    popen(build_chrome_command(chrome, [report]))


def _frame_from_merged(merged: pd.DataFrame, suffix: str) -> pd.DataFrame:
    columns = {
        "source_date": merged["source_date"],
        "sentiment": merged[f"sentiment_{suffix}"].astype(float),
        "action": merged[f"action_{suffix}"].astype(str),
        "direction": merged[f"direction_{suffix}"].astype(str),
        "quantity": merged[f"quantity_{suffix}"],
        "pnl": merged[f"pnl_{suffix}"].astype(float),
    }
    target_move_column = f"target_move_{suffix}"
    if target_move_column in merged.columns:
        columns["target_move"] = merged[target_move_column]
    target_column_column = f"target_column_{suffix}"
    if target_column_column in merged.columns:
        columns["target_column"] = merged[target_column_column]
    return pd.DataFrame(columns).reset_index(drop=True)


def _max_drawdown(cum_pnl: pd.Series) -> float:
    if cum_pnl.empty:
        return 0.0
    drawdown = cum_pnl - cum_pnl.cummax()
    return float(drawdown.min())


def _win_rate(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    return float((pnl > 0).mean() * 100)


def _signal_match_rate(ordinary: pd.DataFrame, walk: pd.DataFrame) -> float:
    if ordinary.empty:
        return 0.0
    matches = (
        (ordinary["action"].reset_index(drop=True) == walk["action"].reset_index(drop=True))
        & (ordinary["direction"].reset_index(drop=True) == walk["direction"].reset_index(drop=True))
    )
    return float(matches.mean() * 100)


def _error_row(pair: ComparisonPair, error: str) -> dict[str, str]:
    return {
        "symbol": pair.symbol,
        "model_key": pair.model_key,
        "target_column": pair.target_column,
        "error": error,
    }


def _equity_figure(comparison: PairComparison) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[str(item) for item in comparison.ordinary["source_date"]],
            y=comparison.ordinary["cum_pnl"],
            mode="lines",
            name="Ordinary backtest",
            line={"color": "#2563eb", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[str(item) for item in comparison.walk["source_date"]],
            y=comparison.walk["cum_pnl"],
            mode="lines",
            name="Walk-forward",
            line={"color": "#dc2626", "width": 2},
        )
    )
    fig.update_layout(
        title="Equity на общих датах",
        template="plotly_white",
        height=360,
        margin={"l": 45, "r": 20, "t": 55, "b": 40},
        hovermode="x unified",
    )
    return fig


def _drawdown_figure(comparison: PairComparison) -> go.Figure:
    ordinary_dd = comparison.ordinary["cum_pnl"] - comparison.ordinary["cum_pnl"].cummax()
    walk_dd = comparison.walk["cum_pnl"] - comparison.walk["cum_pnl"].cummax()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[str(item) for item in comparison.ordinary["source_date"]],
            y=ordinary_dd,
            mode="lines",
            name="Ordinary backtest",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[str(item) for item in comparison.walk["source_date"]],
            y=walk_dd,
            mode="lines",
            name="Walk-forward",
        )
    )
    fig.update_layout(
        title="Drawdown на общих датах",
        template="plotly_white",
        height=280,
        margin={"l": 45, "r": 20, "t": 55, "b": 40},
        hovermode="x unified",
    )
    return fig


def _plot_div(fig: go.Figure, *, include_plotlyjs: bool) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )


def _metrics_table(metrics: dict[str, Any]) -> str:
    rows = [
        ("Период", f"{metrics['start_date']} .. {metrics['end_date']}"),
        ("Общих строк", metrics["overlap_rows"]),
        ("Ordinary P/L", metrics["ordinary_total_pnl"]),
        ("Walk-forward P/L", metrics["walk_total_pnl"]),
        ("Delta P/L", metrics["delta_pnl"]),
        ("Ordinary MaxDD", metrics["ordinary_max_drawdown"]),
        ("Walk-forward MaxDD", metrics["walk_max_drawdown"]),
        ("Ordinary win rate", f"{metrics['ordinary_win_rate']:.1f}%"),
        ("Walk-forward win rate", f"{metrics['walk_win_rate']:.1f}%"),
        ("Совпадение сигналов", f"{metrics['signal_match_rate']:.1f}%"),
    ]
    body = "".join(
        f"<tr><th>{escape(str(name))}</th><td>{escape(_format_number(value))}</td></tr>"
        for name, value in rows
    )
    return f"<table class='metrics'><tbody>{body}</tbody></table>"


def _summary_table(metrics_rows: list[dict[str, Any]]) -> str:
    if not metrics_rows:
        return "<p class='muted'>Нет сопоставимых пар.</p>"
    columns = [
        ("target_column", "Target"),
        ("model_key", "Модель"),
        ("overlap_rows", "Строк"),
        ("ordinary_total_pnl", "Ordinary P/L"),
        ("walk_total_pnl", "WF P/L"),
        ("delta_pnl", "Delta P/L"),
        ("signal_match_rate", "Сигналы %"),
    ]
    header = "".join(f"<th>{escape(title)}</th>" for _, title in columns)
    rows = []
    for row in sorted(metrics_rows, key=lambda item: (item["target_column"], -item["delta_pnl"])):
        cells = "".join(f"<td>{escape(_format_number(row[key]))}</td>" for key, _ in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _errors_table(errors: list[dict[str, str]]) -> str:
    if not errors:
        return "<p class='muted'>Ошибок и пропусков нет.</p>"
    header = "<th>Target</th><th>Модель</th><th>Ошибка</th>"
    rows = "".join(
        "<tr>"
        f"<td>{escape(row['target_column'])}</td>"
        f"<td>{escape(row['model_key'])}</td>"
        f"<td>{escape(row['error'])}</td>"
        "</tr>"
        for row in errors
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Number) and not isinstance(value, bool):
        if isinstance(value, int):
            return f"{value:,}".replace(",", " ")
        return f"{float(value):,.2f}".replace(",", " ")
    return str(value)


def _style() -> str:
    return """
<style>
body { margin: 0; font-family: Arial, sans-serif; color: #20242a; background: #f7f8fa; }
main { max-width: 1380px; margin: 0 auto; padding: 28px 24px 44px; }
h1 { margin: 0; font-size: 30px; }
h2 { margin: 30px 0 12px; font-size: 20px; }
h3 { margin: 0 0 14px; font-size: 17px; }
section { margin: 22px 0; }
.pair { margin: 18px 0 28px; padding-top: 14px; border-top: 1px solid #dfe4ea; }
.muted { color: #5f6875; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
table { width: 100%; border-collapse: collapse; background: white; font-size: 13px; margin-top: 12px; }
th { background: #22324a; color: white; text-align: left; padding: 8px; white-space: nowrap; }
td { border-bottom: 1px solid #e6e9ef; padding: 7px 8px; white-space: nowrap; text-align: right; }
td:first-child, td:nth-child(2) { text-align: left; }
.metrics th { width: 250px; background: #eef2f7; color: #20242a; }
.metrics td { text-align: right; }
@media (max-width: 760px) { main { padding: 20px 12px 32px; } .grid { grid-template-columns: 1fr; } }
</style>
"""
