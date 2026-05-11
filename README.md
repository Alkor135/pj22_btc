# pj22_btc

Исследовательский Python-проект для торговли BTC на основе исторических данных и sentiment-сигналов локальных LLM.

## Загрузка минутных данных MEXC

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

Скрипт скачивает минутные свечи `BTCUSDT` с публичного Spot API MEXC и сохраняет их в SQLite 3 базы по месяцам:

```text
data/mexc/klines/BTCUSDT/1m/2025-09.db
data/mexc/klines/BTCUSDT/1m/2025-10.db
```

При повторном запуске скрипт ищет последнюю сохраненную свечу и докачивает только новые данные. SQLite-файлы исключены из git через `.gitignore`.
