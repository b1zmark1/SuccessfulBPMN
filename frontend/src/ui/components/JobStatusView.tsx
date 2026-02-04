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

export function JobStatusView({ job, isCreating, isPolling, requestError }: JobStatusViewProps) {
  if (!job && !isCreating && !requestError) {
    return <p className="state-text">Отправьте данные, чтобы создать задачу.</p>;
  }

  const status = job?.status;
  const statusLabel = status ? STATUS_LABELS[status] ?? `Неизвестный статус: ${status}` : "Ожидание первого обновления";
  const isUnknown = status !== undefined && !(status in STATUS_LABELS);

  return (
    <section className="status-panel" aria-live="polite">
      <p>
        <strong>Статус:</strong> {statusLabel}
      </p>
      {isCreating ? <p className="state-text">Создаем задачу...</p> : null}
      {isPolling ? <p className="state-text">Обновляем статус с сервера...</p> : null}
      {isUnknown ? <p className="state-text">Сервер вернул неизвестный статус. Опрос продолжается.</p> : null}
      {requestError ? <p className="state-text state-text--error">{requestError}</p> : null}
    </section>
  );
}
