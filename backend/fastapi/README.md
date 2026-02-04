# Слой FastAPI

## Ответственность
FastAPI предоставляет стабильные frontend-контракты и делегирует бизнес-логику сервисам.
В роутерах нет бизнес-логики.

## Версионирование API
Базовый префикс: `/api/v1`

## Эндпоинты
### `POST /api/v1/jobs`
Создает асинхронную job.
- Валидирует вход (`job_type`, `meta`).
- Сохраняет строку `jobs` со статусом `pending`.
- Сохраняет событие outbox в той же транзакции.
- Запускает best-effort отправку из outbox.
- Возвращает `202 Accepted`:
```json
{ "job_id": "<uuid>" }
```

### `GET /api/v1/jobs/{job_id}`
Возвращает состояние job и frontend-ready полезную нагрузку.
- `job_id`
- `job_type`
- `status`
- `result` (когда `done`)
- `error` (когда `error`)
- timestamps (`created_at`, `started_at`, `finished_at`)

### `GET /api/v1/health`
Проверка доступности PostgreSQL и Redis.

## Job-based flow
1. Клиент вызывает `POST /api/v1/jobs`.
2. Backend пишет `jobs(pending)` + `outbox_messages(pending)` в одной транзакции.
3. Dispatcher публикует событие в Redis Streams.
4. При успехе публикации outbox помечается `published`, job переходит в `queued`.
5. Внешний воркер читает stream, выполняет ML и обновляет PostgreSQL (`running -> done/error`).
6. Клиент опрашивает `GET /api/v1/jobs/{job_id}`.

## Границы слоя
- `api/v1/routers/*`: только HTTP-маппинг.
- `services/job_service.py`: orchestration/use-case логика.
- `db/repositories/*`: слой хранения.
- `queue/*`: абстракция Redis.

## Логирование
- Глобально включено структурированное JSON-логирование.
- Контекст запроса (`X-Request-ID` или сгенерированный UUID) попадает в логи и в response header.
- Для событий job добавляется `job_id`, когда он доступен.
