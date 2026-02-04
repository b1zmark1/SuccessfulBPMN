import type { JobResponse, SupportedJobType } from "../../shared/jobTypes";

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;
}

function extractImageSrc(result: Record<string, unknown> | null): string | null {
  if (!result) {
    return null;
  }
  for (const key of ["image_url", "url", "download_url", "file_url", "image"]) {
    const value = result[key];
    if (typeof value === "string" && value.trim().length > 0) {
      return value;
    }
  }
  return null;
}

function textDownloadHref(content: string, mime: string): string {
  return `data:${mime};charset=utf-8,${encodeURIComponent(content)}`;
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

  const result = asRecord(job.result);
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
  const bpmnXml = typeof result?.bpmn_xml === "string" ? result.bpmn_xml : null;
  const mermaidMmd = typeof result?.mermaid_mmd === "string" ? result.mermaid_mmd : null;
  const plantumlPuml = typeof result?.plantuml_puml === "string" ? result.plantuml_puml : null;

  return (
    <section className="result-panel">
      <h3>Сгенерированное изображение</h3>
      {imageSrc ? (
        <>
          <img className="image-preview" src={imageSrc} alt="Сгенерированное изображение" />
          <a className="action-button action-button--secondary" href={imageSrc} download>
            Скачать изображение
          </a>
          {bpmnXml ? (
            <a
              className="action-button action-button--secondary"
              href={textDownloadHref(bpmnXml, "application/xml")}
              download="diagram.bpmn"
            >
              Скачать .bpmn
            </a>
          ) : null}
          {mermaidMmd ? (
            <a
              className="action-button action-button--secondary"
              href={textDownloadHref(mermaidMmd, "text/plain")}
              download="diagram.mmd"
            >
              Скачать .mmd
            </a>
          ) : null}
          {plantumlPuml ? (
            <a
              className="action-button action-button--secondary"
              href={textDownloadHref(plantumlPuml, "text/plain")}
              download="diagram.puml"
            >
              Скачать .puml
            </a>
          ) : null}
        </>
      ) : (
        <ResultAsJson result={result} />
      )}
    </section>
  );
}
