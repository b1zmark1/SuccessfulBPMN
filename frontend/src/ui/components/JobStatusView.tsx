import type { JobResponse } from "../../shared/jobTypes";

const STATUS_LABELS: Record<string, string> = {
  pending: "Ожидает создания",
  queued: "В очереди",
  running: "Выполняется",
  done: "Готово",
  error: "Ошибка",
};

interface JobStatusViewProps {
  jobId: string | null;
  job: JobResponse | null;
  isCreating: boolean;
  isPolling: boolean;
  requestError: string | null;
}

function parseIsoTime(value: string | null | undefined): number | null {
  if (!value) return null;
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? null : ms;
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes > 0) return `${minutes} мин ${seconds} сек`;
  return `${seconds} сек`;
}

export function JobStatusView({ job, isCreating, isPolling, requestError }: JobStatusViewProps) {
  if (!job && !isCreating && !requestError) {
    return <p className="state-text">Отправьте данные, чтобы создать задачу.</p>;
  }

  const status = job?.status;
  const statusLabel = status
    ? (STATUS_LABELS[status] ?? `Неизвестный статус: ${status}`)
    : "Ожидание первого обновления";
  const isUnknown = status !== undefined && !(status in STATUS_LABELS);

  const startedAtMs = parseIsoTime(job?.started_at);
  const finishedAtMs = parseIsoTime(job?.finished_at);
  const createdAtMs = parseIsoTime(job?.created_at);
  const nowMs = Date.now();
  const elapsedMs =
    startedAtMs !== null
      ? (finishedAtMs ?? nowMs) - startedAtMs
      : createdAtMs !== null
        ? (finishedAtMs ?? nowMs) - createdAtMs
        : null;

  return (
    <section className="status-panel" aria-live="polite">
      <p>
        <strong>Статус:</strong> {statusLabel}
      </p>
      {elapsedMs !== null ? (
        <p>
          <strong>Время выполнения:</strong> {formatDuration(elapsedMs)}
        </p>
      ) : null}
      {isCreating ? <p className="state-text">Создаем задачу...</p> : null}
      {isPolling ? <p className="state-text">Обновляем статус с сервера...</p> : null}
      {isUnknown ? <p className="state-text">Сервер вернул неизвестный статус. Опрос продолжается.</p> : null}
      {requestError ? <p className="state-text state-text--error">{requestError}</p> : null}
    </section>
  );
}
