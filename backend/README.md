# Backend ML-сервиса (`backend`)

Готовый к production backend для асинхронной обработки ML-задач в job-based архитектуре.
Backend не выполняет ML-логику: внешние воркеры читают задания из Redis и записывают результат в PostgreSQL.

## Назначение
- Принимать запросы frontend в виде job.
- Хранить состояние job в PostgreSQL (единый источник истины).
- Публиковать события job в Redis Streams для воркеров.
- Отдавать стабильный REST API для опроса статуса и результата.

## Архитектура
- FastAPI: HTTP-контракты и оркестрация.
- PostgreSQL: таблицы `jobs` и `outbox_messages`.
- Redis Streams: только канал доставки (без хранения состояния job).
- Outbox dispatcher: надежная публикация с retry/backoff.

## Основные потоки
1. `POST /api/v1/jobs` создает `jobs(status=pending)` и событие outbox в одной транзакции.
2. Dispatcher публикует событие в Redis Stream и переводит job в `queued`.
3. Воркер обрабатывает job и обновляет PostgreSQL (`running -> done|error`).
4. Frontend опрашивает `GET /api/v1/jobs/{job_id}`.

## Карта документации
- План работ: `tasklist.md`
- Контракты и flow FastAPI: `fastapi/README.md`
- Схема и миграции PostgreSQL: `postgres/README.md`
- Контракт сообщений и гарантии Redis: `redis/README.md`

## Локальный запуск (без Docker)
```bash
cd backend
python -m pip install -e .[test]
alembic upgrade head
uvicorn ml_backend.main:app --reload
```

## Запуск в Docker (рекомендуется)
Из корня репозитория:
```bash
docker compose up --build
```

Точка API:
- `http://localhost:8000/api/v1/jobs`

## Запуск тестов
Unit-тесты:
```bash
pytest backend/tests/unit -q
```

Integration-тесты (нужен запущенный Docker daemon + Testcontainers):
```bash
pytest backend/tests/integration -q
```

Все backend-тесты:
```bash
pytest backend/tests -q
```

## Структура backend
- Код сервиса: `backend/src/ml_backend`
- Миграции: `backend/alembic`
- Тесты: `backend/tests`
- Сборка образа: `backend/Dockerfile`
