import type { JobResponse, SupportedJobType } from "../../shared/jobTypes";

function asStringRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;
}

function extractImageSrc(result: Record<string, unknown> | null): string | null {
  if (!result) {
    return null;
  }

  const candidates = ["image_url", "url", "download_url", "file_url", "image"];
  for (const key of candidates) {
    const candidate = result[key];
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate;
    }
  }

  return null;
}

function ResultAsJson({ result }: { result: Record<string, unknown> | null }) {
  return <pre className="result-json">{JSON.stringify(result ?? {}, null, 2)}</pre>;
}

interface JobResultViewProps {
  scenario: SupportedJobType;
  job: JobResponse | null;
}

export function JobResultView({ scenario, job }: JobResultViewProps) {
  if (!job) {
    return null;
  }

  if (job.status === "error") {
    return (
      <section className="result-panel">
        <h3>Ошибка выполнения</h3>
        <p className="state-text state-text--error">{job.error ?? "Неизвестная ошибка backend"}</p>
      </section>
    );
  }

  if (job.status !== "done") {
    return null;
  }

  const result = asStringRecord(job.result);

  if (scenario === "image_to_text") {
    const textValue = typeof result?.text === "string" ? result.text : null;
    return (
      <section className="result-panel">
        <h3>Распознанный текст</h3>
        {textValue ? <article className="text-result">{textValue}</article> : <ResultAsJson result={result} />}
      </section>
    );
  }

  const imageSrc = extractImageSrc(result);
  return (
    <section className="result-panel">
      <h3>Сгенерированное изображение</h3>
      {imageSrc ? (
        <>
          <img className="image-preview" src={imageSrc} alt="Сгенерированное изображение" />
          <a className="action-button action-button--secondary" href={imageSrc} download>
            Скачать изображение
          </a>
        </>
      ) : (
        <ResultAsJson result={result} />
      )}
    </section>
  );
}
