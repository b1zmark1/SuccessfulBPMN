# ML Job Frontend

Production-ready SPA frontend for asynchronous ML jobs.

## Purpose
- Accept user input for supported job types.
- Create jobs through backend REST API.
- Poll job state until terminal status.
- Render result or backend error without local business-state reconstruction.

## Architecture
- `frontend/src/ui`: pure UI screens/components.
- `frontend/src/api`: typed API client and contract guards.
- `frontend/src/state`: lifecycle orchestration (`create -> poll -> done/error`).
- `frontend/src/shared`: shared types/constants.

Data flow:
1. User selects scenario (`image_to_text` or `text_to_image`).
2. UI builds `meta` payload and sends `POST /api/v1/jobs`.
3. Frontend receives `job_id` and starts polling `GET /api/v1/jobs/{job_id}`.
4. Backend status drives UI (`pending|queued|running|done|error` + unknown fallback).
5. On `done`, result is rendered (image preview + download, or extracted text).
6. On `error`, backend message is shown.

## Run
Requirements: Node.js 20+.

```bash
cd frontend
npm install
npm run dev
```

Build:
```bash
npm run build
```

Tests:
```bash
npm run test
```

## Environment
Copy `frontend/.env.example` to `frontend/.env` and adjust values if needed.

- `VITE_API_BASE_URL` (default: `http://localhost:8000`)
- `VITE_API_PREFIX` (default: `/api/v1`)
