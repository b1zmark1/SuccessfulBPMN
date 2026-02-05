# SuccessfulBPMN

SuccessfulBPMN — асинхронный ML-сервис с job-based архитектурой:

- `image_to_text`: BPMN-изображение -> текст/таблица;
- `text_to_image`: текст -> BPMN/mermaid/plantuml артефакты + изображение.

Frontend создает job, backend пишет статус в PostgreSQL и публикует событие в Redis Stream, worker обрабатывает job.

## 1) Состав проекта

- `frontend/` — React + TypeScript SPA (создание job, polling, результат).
- `backend/` — FastAPI API (`POST /jobs`, `GET /jobs/{job_id}`).
- `workers/` — обработчик очереди (image_to_text/text_to_image pipelines).
- `preprocanddetect/`, `graph_builder/`, `narrator/`, `text_to_diagram/`, `tools/` — ML/рендер пайплайны.
- `docker-compose.yml` — запуск всего стека.

## 2) Требования

Минимум:

- Docker Desktop (или Docker Engine + Compose v2)
- 20+ GB свободного места (образы и зависимости тяжелые)
- 16+ GB RAM желательно

Для GPU (опционально):

- NVIDIA GPU + драйверы
- NVIDIA Container Toolkit (если хотите CUDA в контейнере)

## 3) Запуск проекта в Docker

Есть 2 варианта.

### Быстрый автозапуск через PowerShell (рекомендуется для Windows)

Скрипт проверяет Docker daemon и поднимает сервисы автоматически:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-successfulbpmn.ps1
```

По умолчанию:
- `backend` и `worker` берутся из опубликованных образов Docker Hub;
- `postgres` и `redis` берутся из официальных image;
- выполняется `pull` + `up -d`.

Полезные флаги:

```powershell
# Собирать backend/worker локально вместо published image
powershell -ExecutionPolicy Bypass -File .\scripts\start-successfulbpmn.ps1 -BuildLocal

# Не делать docker pull перед запуском
powershell -ExecutionPolicy Bypass -File .\scripts\start-successfulbpmn.ps1 -NoPull

# После старта сразу смотреть backend/worker логи
powershell -ExecutionPolicy Bypass -File .\scripts\start-successfulbpmn.ps1 -FollowLogs
```

### Вариант A — собрать локально

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

Сервисы:

- backend API: `http://localhost:8000/api/v1`
- postgres: `localhost:5432`
- redis: `localhost:6379`

### Вариант B — запуск из готовых образов (без сборки, вручную)

1. Запуск:

```bash
docker compose -f docker-compose.yml -f docker-compose.published.yml pull
docker compose -f docker-compose.yml -f docker-compose.published.yml up -d
```

> Первый pull/push может идти долго из-за большого слоя моделей/зависимостей.

## 4) Запуск frontend

В отдельном терминале:

```bash
cd frontend
npm install
npm run dev
```

Открыть:

- `http://localhost:5173`

### `frontend/.env`

```env
VITE_API_BASE_URL=
VITE_API_PREFIX=/api/v1
```

Если `VITE_API_BASE_URL` пустой, Vite proxy отправляет API-запросы на `http://localhost:8000`.

## 5) Проверка работоспособности

### Проверка Redis/Postgres

```bash
docker compose exec redis redis-cli ping
docker compose exec postgres pg_isready -U postgres -d ml_jobs
```

### Посмотреть последние job

```bash
docker compose exec postgres psql -U postgres -d ml_jobs -c "SELECT created_at, job_type, status FROM jobs ORDER BY created_at DESC LIMIT 20;"
```

### Проверка text_to_image внутри worker

```bash
docker compose exec worker python -c "from workers.text_to_image_pipeline import run_text_to_image_pipeline as f; r=f('Простой процесс: старт -> проверка -> завершение'); print(r.keys()); print('has_mmd=', bool(r.get('mermaid_mmd'))); print('has_img=', bool(r.get('image_url'))); print('warn=', r.get('render_warning'))"
```

Ожидаемо:

- `has_mmd=True`
- `has_img=True` (или `False`, если рендер недоступен, но тогда будет `render_warning`)

## 6) Остановка и очистка

Остановить сервисы:

```bash
docker compose down
```

Полная очистка (включая БД/кэш volumes):

```bash
docker compose down -v
```

Удалить dangling-образы/кэш:

```bash
docker system prune -f
```

## 7) Частые проблемы

### 1. Worker "обрабатывает снова"

Проверьте, что `WORKER_START_ID` = `$` (а не `0-0`) в `docker-compose.yml`.

### 2. `text_to_image` падает с `node/mmdc/plantuml not installed`

Проверьте в контейнере:

```bash
docker compose exec worker node -v
docker compose exec worker mmdc -V
docker compose exec worker plantuml -version
```

### 3. В браузере ошибка `tainted canvas` при экспорте PNG из Mermaid

Это ограничение браузера. В UI настроен fallback: автоматически скачивается `SVG`

## 8) Дополнительная документация

- `frontend/README.md` — frontend архитектура и запуск
- `frontend/ui/README.md` — UX flow
- `frontend/api/README.md` — API layer фронта
- `backend/README.md` — backend детали
- `workers/dependencies.md` — зависимости worker
