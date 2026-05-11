# План реализации загрузчика минутных данных MEXC

> **Для агентной разработки:** обязательный под-навык: использовать `superpowers:subagent-driven-development` (рекомендуется) или `superpowers:executing-plans`, чтобы выполнять этот план по задачам. Шаги оформлены чекбоксами (`- [ ]`) для отслеживания прогресса.

**Цель:** создать скрипт, который скачивает одноминутные свечи BTCUSDT с MEXC начиная с даты из `settings.yaml`, сохраняет их в месячные SQLite-базы и при повторном запуске продолжает загрузку с новой свечи после последней уже сохраненной.

**Архитектура:** загрузчик будет состоять из небольших тестируемых частей в `src/pj22_btc`: чтение настроек, клиент MEXC для kline-данных, месячное SQLite-хранилище и orchestration-логика докачки. `scripts/download_mexc_klines.py` будет тонкой CLI-оберткой. Unit-тесты будут использовать fake-клиент и временные SQLite-директории, поэтому они не зависят от живого API MEXC.

**Технологии:** Python 3.13 standard library, SQLite 3, `unittest`, публичный Spot API MEXC `GET /api/v3/klines`.

---

### Задача 1: Unit-тесты загрузчика

**Файлы:**
- Создать: `tests/test_mexc_downloader.py`

- [ ] **Шаг 1: Написать падающие тесты**

Добавить тесты для выбора месячного DB-файла, идемпотентной записи в SQLite, расчета cursor-а для докачки и записи свечей в разные месячные базы.

- [ ] **Шаг 2: Запустить тесты и подтвердить падение**

Команда: `C:\Python\Python31313\python.exe -m unittest tests.test_mexc_downloader -v`

Ожидаемый результат: тесты падают, потому что `pj22_btc.mexc_downloader` еще не существует.

### Задача 2: Библиотека загрузчика

**Файлы:**
- Создать: `src/pj22_btc/__init__.py`
- Создать: `src/pj22_btc/mexc_downloader.py`

- [ ] **Шаг 1: Реализовать минимальный код под тесты**

Добавить dataclass-и `Kline`, `DownloaderConfig` и `DownloadSummary`; функции работы с датами; `MonthlySQLiteKlineStore`; `MexcKlineClient`; функцию `sync_klines`.

- [ ] **Шаг 2: Запустить unit-тесты**

Команда: `C:\Python\Python31313\python.exe -m unittest tests.test_mexc_downloader -v`

Ожидаемый результат: все тесты загрузчика проходят.

### Задача 3: Конфигурация и CLI

**Файлы:**
- Создать: `settings.yaml`
- Создать: `scripts/download_mexc_klines.py`
- Создать: `pyproject.toml`
- Изменить: `.gitignore`
- Изменить: `README.md`
- Создать: `data/mexc/klines/BTCUSDT/1m/.gitkeep`

- [ ] **Шаг 1: Добавить конфигурацию проекта**

`settings.yaml` должен содержать `start_date: 2025-09-01`, `symbol: BTCUSDT`, `interval: 1m` и директорию для месячных SQLite-файлов. `.gitignore` должен исключать SQLite DB-файлы и служебные sidecar-файлы SQLite.

- [ ] **Шаг 2: Добавить CLI-обертку**

Скрипт должен читать `settings.yaml`, вызывать `sync_klines` и печатать короткую сводку: сколько строк скачано, сколько вставлено, с какого cursor-а началась докачка, какая последняя свеча обработана и где лежат месячные DB-файлы.

- [ ] **Шаг 3: Запустить проверку**

Команда: `C:\Python\Python31313\python.exe -m unittest tests.test_mexc_downloader -v`

Ожидаемый результат: все тесты проходят.

Команда: `C:\Python\Python31313\python.exe scripts\download_mexc_klines.py --help`

Ожидаемый результат: справка CLI печатается успешно и не обращается к MEXC.
