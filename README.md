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

Результат сохраняется в годовые SQLite DB:

```text
data/mexc/klines/BTCUSDT/1d_msk/2025.db
data/mexc/klines/BTCUSDT/1d_msk/2026.db
```
