"""Report builders for saved walk-forward results."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def build_report(
    output_dir: Path,
    *,
    target_column: str,
    output_html: Path | None = None,
    output_xlsx: Path | None = None,
) -> tuple[Path, Path]:
    """Build HTML and XLSX reports from saved walk-forward CSV artifacts."""
    output_dir = Path(output_dir)
    suffix = _safe_name(target_column)
    summary_path = output_dir / f"summary_{suffix}.csv"
    model_summary_path = output_dir / f"model_summary_{suffix}.csv"
    if not summary_path.exists():
        raise ValueError(f"Walk-forward summary not found: {summary_path}")
    if not model_summary_path.exists():
        raise ValueError(f"Walk-forward model summary not found: {model_summary_path}")

    summary = pd.read_csv(summary_path)
    model_summary = pd.read_csv(model_summary_path)
    trades = load_all_trades(output_dir, model_summary, target_column=target_column)

    output_html = output_html or output_dir / f"walk_forward_report_{suffix}.html"
    output_xlsx = output_xlsx or output_dir / f"walk_forward_report_{suffix}.xlsx"
    write_excel_report(
        summary=summary,
        model_summary=model_summary,
        trades=trades,
        output_xlsx=output_xlsx,
    )
    write_html_report(
        summary=summary,
        model_summary=model_summary,
        trades=trades,
        target_column=target_column,
        output_html=output_html,
    )
    return output_html, output_xlsx


def load_all_trades(
    output_dir: Path,
    model_summary: pd.DataFrame,
    *,
    target_column: str,
) -> pd.DataFrame:
    """Load per-model `trades.csv` files listed in model summary."""
    frames: list[pd.DataFrame] = []
    for row in model_summary.to_dict("records"):
        symbol = str(row.get("symbol", ""))
        model_key = str(row.get("model_key", ""))
        trades_path = Path(output_dir) / symbol / model_key / target_column / "trades.csv"
        if not trades_path.exists():
            continue
        frame = pd.read_csv(trades_path)
        if frame.empty:
            continue
        frame["symbol"] = frame.get("symbol", symbol)
        frame["model_key"] = frame.get("model_key", model_key)
        frame["target_column"] = frame.get("target_column", target_column)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "source_date" in result.columns:
        result["source_date"] = pd.to_datetime(result["source_date"], errors="coerce")
        result = result.sort_values(["model_key", "source_date"]).reset_index(drop=True)
    return result


def write_excel_report(
    *,
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    trades: pd.DataFrame,
    output_xlsx: Path,
) -> None:
    """Write the workbook with model summary, daily summary, and trades."""
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        _excel_safe(model_summary).to_excel(writer, sheet_name="models", index=False)
        _excel_safe(summary).to_excel(writer, sheet_name="daily", index=False)
        _excel_safe(trades).to_excel(writer, sheet_name="trades", index=False)


def write_html_report(
    *,
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    trades: pd.DataFrame,
    target_column: str,
    output_html: Path,
) -> None:
    """Write a compact HTML dashboard."""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    total_pnl = float(pd.to_numeric(model_summary.get("total_pnl", pd.Series(dtype=float)), errors="coerce").sum())
    total_trades = int(pd.to_numeric(model_summary.get("trades", pd.Series(dtype=float)), errors="coerce").sum())
    ok_days = int((summary.get("status", pd.Series(dtype=str)) == "ok").sum())
    skipped_days = int((summary.get("status", pd.Series(dtype=str)) == "skipped").sum())

    blocks = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>Walk-forward report {escape(target_column)}</title>",
        _style(),
        "</head><body>",
        "<main>",
        "<h1>Walk-forward report</h1>",
        f"<p class='subtle'>target={escape(target_column)}</p>",
        "<section class='kpis'>",
        _kpi("Total P/L", _format_number(total_pnl)),
        _kpi("Trades", str(total_trades)),
        _kpi("OK days", str(ok_days)),
        _kpi("Skipped days", str(skipped_days)),
        "</section>",
    ]
    if not trades.empty:
        blocks.append(_plot_div(_equity_figure(trades), include_plotlyjs=True))
        blocks.append(_plot_div(_model_pnl_figure(model_summary), include_plotlyjs=False))
    blocks.extend(
        [
            "<h2>Models</h2>",
            _table_html(model_summary),
            "<h2>Daily Summary</h2>",
            _table_html(summary, max_rows=200),
            "</main></body></html>",
        ]
    )
    output_html.write_text("\n".join(blocks), encoding="utf-8")


def _equity_figure(trades: pd.DataFrame) -> go.Figure:
    prepared = trades.copy()
    prepared["source_date"] = pd.to_datetime(prepared["source_date"], errors="coerce")
    prepared["pnl"] = pd.to_numeric(prepared["pnl"], errors="coerce").fillna(0.0)
    daily = prepared.groupby("source_date", dropna=True)["pnl"].sum().sort_index()
    equity = daily.cumsum()
    fig = go.Figure(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines+markers",
            name="Equity",
            hovertemplate="%{x|%Y-%m-%d}<br>P/L: %{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Aggregate Equity",
        template="plotly_white",
        height=360,
        margin={"l": 45, "r": 20, "t": 60, "b": 40},
    )
    return fig


def _model_pnl_figure(model_summary: pd.DataFrame) -> go.Figure:
    prepared = model_summary.copy()
    prepared["total_pnl"] = pd.to_numeric(prepared["total_pnl"], errors="coerce").fillna(0.0)
    fig = go.Figure(
        go.Bar(
            x=prepared["model_key"],
            y=prepared["total_pnl"],
            name="Total P/L",
            hovertemplate="%{x}<br>P/L: %{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Model Total P/L",
        template="plotly_white",
        height=340,
        margin={"l": 45, "r": 20, "t": 60, "b": 80},
    )
    return fig


def _plot_div(fig: go.Figure, *, include_plotlyjs: bool) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
    )


def _table_html(frame: pd.DataFrame, max_rows: int = 100) -> str:
    if frame.empty:
        return "<p class='subtle'>No rows.</p>"
    visible = frame.head(max_rows)
    return visible.to_html(index=False, classes="data-table", border=0, escape=True)


def _kpi(label: str, value: Any) -> str:
    return (
        "<div class='kpi'>"
        f"<span>{escape(str(label))}</span>"
        f"<strong>{escape(str(value))}</strong>"
        "</div>"
    )


def _format_number(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime("%Y-%m-%d")
    return result


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _style() -> str:
    return """
<style>
body { margin: 0; font-family: Arial, sans-serif; color: #20242a; background: #f7f8fa; }
main { max-width: 1320px; margin: 0 auto; padding: 28px 24px 44px; }
h1 { margin: 0; font-size: 30px; }
h2 { margin: 30px 0 12px; font-size: 20px; }
.subtle { color: #5f6875; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin: 20px 0 24px; }
.kpi { background: white; border: 1px solid #dfe4ea; border-radius: 6px; padding: 14px 16px; }
.kpi span { display: block; color: #5f6875; font-size: 13px; margin-bottom: 8px; }
.kpi strong { font-size: 22px; }
.data-table { width: 100%; border-collapse: collapse; background: white; font-size: 13px; }
.data-table th { background: #22324a; color: white; text-align: left; padding: 8px; white-space: nowrap; }
.data-table td { border-bottom: 1px solid #e6e9ef; padding: 7px 8px; white-space: nowrap; }
@media (max-width: 760px) { .kpis { grid-template-columns: 1fr 1fr; } main { padding: 20px 12px 32px; } }
</style>
"""
