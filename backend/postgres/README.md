# Слой PostgreSQL

## Роль в архитектуре
PostgreSQL — единый источник истины для жизненного цикла job и frontend-вывода.

## Схема
Основная таблица: `jobs`
- `jobId` (UUID, PK)
- `job_type` (`job_type` enum): `image_to_text`, `text_to_image`, `text_to_diagram`, `image_to_table`
- `status` (`job_status` enum): `pending`, `queued`, `running`, `done`, `error`
- `meta` (JSONB) — валидированный входной payload
- `result` (JSONB nullable) — frontend-ready результат завершенной job
- `error` (TEXT nullable) — детали терминальной ошибки
- `created_at`, `started_at`, `finished_at` (timestamptz)

Дополнительная таблица: `outbox_messages`
- хранит транзакционные события для отправки в Redis Streams
- гарантирует, что сообщение не потеряется между commit в БД и publish в очередь

## Миграции
Файлы Alembic находятся в `backend/alembic`.
- конфиг: `backend/alembic.ini`
- начальная миграция: `backend/alembic/versions/0001_create_jobs_and_outbox.py`

Запуск миграций:
```bash
cd backend
alembic upgrade head
```

## Контракт переходов статусов
Разрешенные переходы:
- `pending -> queued | error`
- `queued -> running | error`
- `running -> done | error`
- `done`, `error` — терминальные

Правила переходов централизованы в `backend/src/ml_backend/db/status_machine.py`.
