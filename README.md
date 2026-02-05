# SuccessfulBPMN

SuccessfulBPMN — ML-сервис для преобразования:
- `image_to_text`: BPMN-диаграмма -> структурированное текстовое описание/таблица;
- `text_to_image`: текстовое описание процесса -> диаграмма.

Проект построен как job-based система: frontend создает job, backend кладет событие в очередь, worker обрабатывает, frontend опрашивает статус.

## Архитектура проекта

- `frontend/` — SPA на React + TypeScript.
  - UI сценариев, создание job (`POST /jobs`), polling статуса (`GET /jobs/{job_id}`), отображение результатов.
- `backend/` — FastAPI API.
  - Контракты REST, хранение состояния job, outbox-публикация в Redis Stream.
- `workers/` — воркер обработки job.
  - `image_to_text`: detect/ensemble/ocr/label/graph/narrator pipeline.
  - `text_to_image`: генерация диаграмм и артефактов.
- `preprocanddetect/` — детекция, OCR, merge-разметка.
- `narrator/` — генерация итогового текста/таблицы по графу.
- `graph_builder/` — построение графа процесса из merged ensemble.
- `docker-compose.yml` — единый запуск инфраструктуры (postgres, redis, backend, worker).

## Поток данных (job lifecycle)

1. Frontend отправляет `POST /api/v1/jobs` с `job_type` и `meta`.
2. Backend создает запись job в Postgres и публикует событие в Redis Stream.
3. Worker читает stream, ставит job в `running`, выполняет pipeline.
4. Worker пишет `done`/`error` + `result` в Postgres.
5. Frontend опрашивает `GET /api/v1/jobs/{job_id}` до терминального статуса.

## Быстрый старт в Docker (backend + infra + worker)

Требования:
- Docker Desktop (или Docker Engine + Compose v2).

Из корня проекта:

```bash
docker compose up --build -d
```

Проверка:

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f worker
```

API backend:
- `http://localhost:8000/api/v1`

Сервисы:
- Postgres: `localhost:5432`
- Redis: `localhost:6379`

Остановка:

```bash
docker compose down
```

С удалением volumes (полная очистка БД/кэшей):

```bash
docker compose down -v
```

## Запуск frontend

Frontend запускается отдельно (Vite dev server).

```bash
cd frontend
npm install
npm run dev
```

По умолчанию фронт доступен на:
- `http://localhost:5173`

### Переменные окружения frontend

Создайте `frontend/.env` (если нужно переопределить):

```env
VITE_API_BASE_URL=
VITE_API_PREFIX=/api/v1
```

Если `VITE_API_BASE_URL` пустой, в dev-режиме используется proxy Vite на `http://localhost:8000`.

## Полезные команды

Проверить Redis:

```bash
docker compose exec redis redis-cli ping
```

Проверить Postgres:

```bash
docker compose exec postgres pg_isready -U postgres
```

Посмотреть job в БД:

```bash
docker compose exec postgres psql -U postgres -d ml_jobs -c "SELECT created_at, job_type, status FROM jobs ORDER BY created_at DESC LIMIT 20;"
```

## Где смотреть подробнее

- `frontend/README.md` — frontend архитектура и запуск.
- `frontend/ui/README.md` — структура экранов и UX flow.
- `frontend/api/README.md` — API layer frontend.
- `backend/README.md` — backend детали.
- `workers/dependencies.md` — зависимости worker и runtime нюансы.
