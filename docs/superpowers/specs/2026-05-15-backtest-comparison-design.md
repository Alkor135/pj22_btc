# Дизайн сравнения ordinary backtest и walk-forward

Дата: 2026-05-15
Проект: pj22_btc

## Цель

Добавить в проект нативный отчет сравнения ordinary sentiment backtest и walk-forward backtest по BTCUSDT. Отчет должен помогать быстро понять, насколько результаты правил, подобранных на всей истории, отличаются от walk-forward режима без look-ahead.

## Область работ

Функция будет работать по уже созданным артефактам проекта и не будет сама запускать backtest или walk-forward. Сравнение строится для всех выбранных моделей и для двух target-колонок: `next_body` и `next_open_to_open`. Обе target-колонки попадают в один HTML-файл, но визуально разделяются по секциям; их данные не смешиваются в одном графике.

По умолчанию отчет создается в:

```text
reports/backtest_comparison/backtest_vs_walk_forward.html
```

После успешной сборки CLI открывает этот HTML в Google Chrome в новом окне. Открытие можно отключить флагом `--no-open`.

## Архитектура

Основная логика будет находиться в модуле:

```text
src/pj22_btc/backtest_comparison.py
```

Запускаемый сценарий будет находиться в:

```text
scripts/create_backtest_comparison_report.py
```

Модуль будет независимым от CLI: функции discovery, загрузки данных, расчета метрик и рендера HTML можно будет тестировать напрямую. CLI будет читать `settings.yaml`, выбирать enabled-модели или список из `--models`, строить отчет и открывать файл в Chrome.

## Источники данных

Ordinary backtest берется из текущей структуры sentiment-отчетов:

```text
reports/sentiment/BTCUSDT/<model_key>/backtest/sentiment_backtest_<target_column>_results.xlsx
```

Walk-forward берется из текущей структуры walk-forward результатов:

```text
reports/walk_forward/BTCUSDT/<model_key>/<target_column>/trades.xlsx
```

Если XLSX отсутствует, модуль фиксирует это в секции ошибок и продолжает остальные пары. Основной расчет выполняется только на пересекающихся датах ordinary и walk-forward для конкретной пары `model_key + target_column`.

## Метрики

Для каждой сопоставимой пары считаются:

- период пересечения дат;
- количество общих строк;
- ordinary total P/L;
- walk-forward total P/L;
- delta P/L;
- ordinary max drawdown;
- walk-forward max drawdown;
- ordinary win rate;
- walk-forward win rate;
- процент совпадения сигналов по `action` и `direction`.

В HTML будет сводная таблица по всем парам, отдельные секции для `next_body` и `next_open_to_open`, график equity, график drawdown и таблица метрик для каждой модели.

## CLI

Основной запуск:

```powershell
.\.venv\Scripts\python.exe scripts\create_backtest_comparison_report.py
```

Фильтр моделей:

```powershell
.\.venv\Scripts\python.exe scripts\create_backtest_comparison_report.py --models gemma3_12b,qwen3_14b
```

Сборка без открытия Chrome:

```powershell
.\.venv\Scripts\python.exe scripts\create_backtest_comparison_report.py --no-open
```

Также CLI поддержит явные пути `--settings`, `--walk-forward-dir`, `--reports-dir` и `--output-html`, чтобы отчет можно было пересобрать из нестандартных папок.

## Открытие HTML

Открытие будет использовать существующий подход проекта из `pj22_btc.html_reports`: путь Chrome по умолчанию `DEFAULT_CHROME_PATH` и запуск с `--new-window`. Если Chrome не найден или запуск не удался, отчет остается созданным, а CLI печатает понятное сообщение об ошибке открытия и возвращает успешный код для самой сборки отчета.

## Обработка ошибок

Ошибки по отдельным парам не должны ломать весь отчет. В HTML будет секция "Ошибки и пропуски" со строками `target_column`, `model_key` и текстом причины. Фатальными остаются только ошибки уровня конфигурации: отсутствующий `settings.yaml`, некорректная секция настроек или невозможность записать итоговый HTML.

## Тестирование

Тесты будут добавлены в `tests/test_backtest_comparison.py`. Они проверят:

- discovery expected-путей для двух target-колонок;
- нормализацию и расчет overlap-метрик;
- HTML с раздельными секциями target-колонок;
- сохранение отчета при отсутствующих отдельных файлах с записью ошибок;
- helper открытия Chrome без реального запуска процесса.

## Не входит в объем

Первая версия не будет запускать ordinary backtest или walk-forward автоматически, не будет строить интерактивный web-app и не будет объединять две target-колонки на одном графике. Добавление шага в общий `run_pipeline.py` можно сделать отдельным небольшим расширением после базового отчета.
