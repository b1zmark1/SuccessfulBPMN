# API Layer

Frontend uses backend as source of truth through typed REST methods.

## Endpoints
- `POST /api/v1/jobs`
  - Request:
    - `job_type`: `"image_to_text" | "text_to_image"`
    - `meta`: scenario-specific payload
  - Response (`202`):
    - `job_id: string (UUID)`
- `GET /api/v1/jobs/{job_id}`
  - Response (`200`):
    - `job_id`
    - `job_type`
    - `status`
    - `result`
    - `error`
    - `created_at`
    - `started_at`
    - `finished_at`

## Implementation
- `frontend/src/api/httpClient.ts`
  - Centralized `fetch` wrapper.
  - Handles base URL and API prefix.
  - Maps network failures and HTTP errors (`ApiError`, `NetworkError`).
- `frontend/src/api/jobsApi.ts`
  - `createJob(payload)`
  - `getJob(jobId)`
  - Runtime shape assertions for contract safety.

## Typing and Safety
- Strict TypeScript contracts in `frontend/src/shared/jobTypes.ts`.
- Unknown statuses are accepted as string and handled by UI fallback.
- Frontend does not infer progress; it mirrors backend status only.
