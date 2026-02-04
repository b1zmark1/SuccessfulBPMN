# Слой Redis

## Роль в архитектуре
Redis используется строго как транспортная очередь для доставки задач воркерам.
Состояние job никогда не берется из Redis.

## Тип очереди
Backend использует Redis Streams (`XADD`) с именем stream из env:
- `REDIS_STREAM_NAME` (по умолчанию: `jobs:stream`)

## Контракт сообщения (`v1`)
Публикуемые поля:
- `version` (string): `v1`
- `JobID` (UUID string)
- `job_type` (string enum)
- `Metadata` (compact JSON string)

Реализация producer:
- `backend/src/ml_backend/queue/redis_stream_publisher.py`
- dispatcher: `backend/src/ml_backend/services/outbox_dispatcher.py`

## Идемпотентность публикации
- Ключ идемпотентности — `JobID` (одна job = одно событие очереди).
- Повторная доставка возможна (at-least-once), воркеры должны дедуплицировать по `JobID`.

## Гарантии доставки
- Backend пишет job и outbox-событие в одной транзакции PostgreSQL.
- Dispatcher публикует pending-сообщения в Redis Streams.
- При успехе: outbox помечается `published`, job переходит `pending -> queued`.
- При ошибке: outbox остается `pending` с метаданными повтора (`attempts`, `available_at`, `error`).

Это обеспечивает at-least-once доставку при условии, что PostgreSQL остается источником истины.

## Обработка ошибок
- Ошибки публикации пишутся в структурированные логи.
- Применяется экспоненциальный backoff перед повторной попыткой.
- Нет silent failure: текст ошибки сохраняется в `outbox_messages.error`.
