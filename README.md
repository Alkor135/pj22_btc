# pj22_btc

Исследовательский Python-проект для торговли BTC на основе исторических данных и sentiment-сигналов локальных LLM.

## Загрузка пятиминутных данных MEXC

Дата начала загрузки находится в [settings.yaml](C:/Users/Alkor/VSCode/pj22_btc/settings.yaml) в поле:

```yaml
mexc:
  start_date: 2025-09-01
```

Чтобы изменить стартовую дату, поменяй это значение в формате `YYYY-MM-DD`.

Запуск загрузки:

```powershell
.\.venv\Scripts\python.exe scripts\download_mexc_klines.py
```

Скрипт скачивает пятиминутные свечи `BTCUSDT` с публичного Spot API MEXC и сохраняет их в SQLite 3 базы по годам:

```text
data/mexc/klines/BTCUSDT/5m/2025.db
data/mexc/klines/BTCUSDT/5m/2026.db
```

При повторном запуске скрипт ищет последнюю сохраненную свечу и докачивает только новые данные. SQLite-файлы исключены из git через `.gitignore`.

## Конвертация 5m в дневные свечи MSK

5m-свечи MEXC сохранены в UTC: `open_time_ms` совпадает с `open_time_utc`.
UTC не имеет перехода на летнее или зимнее время. Для московской сессии
используется IANA-таймзона `Europe/Moscow` через пакет `tzdata`.
Дневные свечи строятся по московской сессии:

```text
[21:00 предыдущего дня MSK; 21:00 текущего дня MSK)
```

Свеча ровно в `21:00 MSK` относится уже к следующей дневной сессии.
Неполные дневные сессии по умолчанию не записываются.

Запуск конвертера:

```powershell
.\.venv\Scripts\python.exe scripts\convert_5m_to_daily.py
```

Результат сохраняется в один SQLite DB-файл:

```text
data/mexc/klines/BTCUSDT/daily_msk.db
```

## Расчет sentiment-оценок

Prompt, список моделей Ollama, кэш и выходная папка задаются в
[settings.yaml](C:/Users/Alkor/VSCode/pj22_btc/settings.yaml) в секции `sentiment`.

Запуск всех моделей с `enabled: true`:

```powershell
.\.venv\Scripts\python.exe scripts\create_sentiment_scores.py
```

Запуск конкретных моделей:

```powershell
.\.venv\Scripts\python.exe scripts\create_sentiment_scores.py --models gemma3_12b,gpt-oss_20b
```

Результат сохраняется отдельно по каждой модели:

```text
data/sentiment/BTCUSDT/<model_key>/sentiment_scores.pkl
```

Во время долгого прогона скрипт показывает progress bar с ETA. PKL-чекпоинт
сохраняется каждые `sentiment.save_every` обработанных файлов, по умолчанию
каждые 10. Если модель вернула нестрогий ответ или запрос упал, скрипт делает
до `sentiment.max_retry_passes` дополнительных проходов по проблемным файлам.
В progress bar также выводится размещение модели из `ollama ps`, например
`processor=100% GPU` или `processor=47%/53% CPU/GPU`.

Для воспроизводимости запросы к Ollama выполняются с `stream=false` и
детерминированными options: `temperature=0`, `top_p=1`, `top_k=1`, `seed=42`.
Эти параметры сохраняются в PKL в колонке `generation_options`.

## Исследование sentiment-логики

После расчета `sentiment_scores.pkl` можно проверить торговую логику по каждой
модели. По умолчанию P/L считается по колонке `next_body`; для сравнения
реакции open-to-open можно указать `--target-column next_open_to_open`.

Групповая статистика по значениям sentiment:

```powershell
.\.venv\Scripts\python.exe scripts\create_sentiment_group_stats.py
.\.venv\Scripts\python.exe scripts\create_sentiment_group_stats.py --models gemma3_12b --target-column next_open_to_open
```

Генерация рекомендованных правил:

```powershell
.\.venv\Scripts\python.exe scripts\create_rules_recommendation.py
.\.venv\Scripts\python.exe scripts\create_rules_recommendation.py --models gemma3_12b --target-column next_open_to_open
```

Backtest по rules YAML:

```powershell
.\.venv\Scripts\python.exe scripts\run_sentiment_backtest.py
.\.venv\Scripts\python.exe scripts\run_sentiment_backtest.py --models gemma3_12b --target-column next_open_to_open
```

Результаты сохраняются отдельно по моделям:

```text
reports/sentiment/BTCUSDT/<model_key>/group_stats/
reports/sentiment/BTCUSDT/<model_key>/rules/
reports/sentiment/BTCUSDT/<model_key>/backtest/
reports/sentiment/BTCUSDT/<model_key>/plots/
```
