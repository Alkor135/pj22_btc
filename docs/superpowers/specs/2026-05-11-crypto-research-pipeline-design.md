# Crypto Research Pipeline Design

Date: 2026-05-11
Project: pj22_btc

## Goal

Build a Python research project for cryptocurrency trading experiments, starting with BTC-focused workflows. The project should support reproducible scripts, testable core logic, and a clear separation between raw data, processed data, strategy code, backtest outputs, and reports.

## Scope

The first project scaffold will provide a research pipeline, not a live trading bot. It will be designed around local historical OHLCV data files first, with exchange/API ingestion left as a later extension. The initial code structure should make it easy to add indicators, strategies, and backtest metrics without mixing experiment scripts with reusable library code.

## Architecture

The repository will be a Python package under `src/pj22_btc`. Reusable logic will live in package modules, while command-style workflows will live in `scripts/`. Tests will target the reusable modules directly.

Planned structure:

```text
pj22_btc/
  pyproject.toml
  README.md
  configs/
    default.yaml
  data/
    raw/.gitkeep
    processed/.gitkeep
  reports/
    .gitkeep
  scripts/
    run_backtest.py
  src/
    pj22_btc/
      __init__.py
      backtest.py
      data.py
      indicators.py
      strategy.py
  tests/
    test_backtest.py
    test_indicators.py
```

## Components

`data.py` will load and validate OHLCV data from local files. It should normalize expected columns such as timestamp, open, high, low, close, and volume.

`indicators.py` will contain deterministic indicator calculations, starting with simple moving averages or similarly small functions that are easy to verify.

`strategy.py` will hold strategy rules. The first strategy can be intentionally simple, such as a moving-average crossover, so the project has a working end-to-end example.

`backtest.py` will convert market data and strategy signals into trades, equity values, and summary metrics. The initial version should be conservative and transparent rather than feature-heavy.

`scripts/run_backtest.py` will load config, run the default workflow, and write output into `reports/`.

## Data Flow

Historical OHLCV files are stored in `data/raw/`. Processing functions read raw files, validate schema, optionally write cleaned data into `data/processed/`, then pass data into indicator and strategy functions. The backtest module calculates results and the script writes a human-readable report or CSV output under `reports/`.

## Error Handling

Data-loading functions should fail with clear exceptions when required columns are missing, files are absent, timestamps cannot be parsed, or numeric market columns contain invalid values. Scripts should surface these errors plainly so failed experiments are easy to diagnose.

## Testing

Tests should focus first on pure logic: indicator calculations, signal generation, and core backtest accounting. Small inline DataFrame fixtures are preferred for unit tests, because they keep expected behavior visible and avoid dependence on large market data files.

## Environment

The local virtual environment is `.venv`, created with:

```powershell
C:\Python\Python31313\python.exe -m venv .venv
```

The initial dependency set should stay small: `pandas`, `numpy`, `pyyaml`, `pytest`, and optionally `ruff` for formatting/linting.
