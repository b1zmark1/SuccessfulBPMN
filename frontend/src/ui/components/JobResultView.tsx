import type { JobResponse, SupportedJobType } from "../../shared/jobTypes";
import { MermaidEditorPreview } from "./MermaidEditorPreview";

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;
}

function extractImageSrc(result: Record<string, unknown> | null): string | null {
  if (!result) return null;
  for (const key of ["image_url", "url", "download_url", "file_url", "image"]) {
    const value = result[key];
    if (typeof value === "string" && value.trim().length > 0) return value;
  }
  return null;
}

function textDownloadHref(content: string, mime: string): string {
  return `data:${mime};charset=utf-8,${encodeURIComponent(content)}`;
}

function maybeFixMojibake(value: string): string {
  try {
    const bytes = Uint8Array.from(value, (char) => char.charCodeAt(0) & 0xff);
    const decoded = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    const sourceCyr = (value.match(/[А-Яа-яЁё]/g) ?? []).length;
    const decodedCyr = (decoded.match(/[А-Яа-яЁё]/g) ?? []).length;
    return decodedCyr > sourceCyr ? decoded : value;
  } catch {
    return value;
  }
}

function ResultAsJson({ result }: { result: Record<string, unknown> | null }) {
  return <pre className="result-json">{JSON.stringify(result ?? {}, null, 2)}</pre>;
}

interface JobResultViewProps {
  scenario: SupportedJobType;
  job: JobResponse | null;
  uploadedImageSrc?: string | null;
}

export function JobResultView({ scenario, job, uploadedImageSrc }: JobResultViewProps) {
  if (!job) return null;

  if (job.status === "error") {
    return (
      <section className="result-panel">
        <h3>Ошибка выполнения</h3>
        <p className="state-text state-text--error">{job.error ?? "Неизвестная ошибка backend"}</p>
      </section>
    );
  }

  if (job.status !== "done") return null;

  const result = asRecord(job.result);
  if (scenario === "image_to_text") {
    const textValue = typeof result?.text === "string" ? maybeFixMojibake(result.text) : null;
    return (
      <section className="result-panel">
        {uploadedImageSrc ? (
          <div className="source-preview">
            <h3>Исходное изображение</h3>
            <img
              className="source-preview__image"
              src={uploadedImageSrc}
              alt="Загруженная BPMN-диаграмма"
            />
          </div>
        ) : null}
        <h3>Распознанный текст</h3>
        {textValue ? <article className="text-result">{textValue}</article> : <ResultAsJson result={result} />}
      </section>
    );
  }

  const bpmnXml = typeof result?.bpmn_xml === "string" ? maybeFixMojibake(result.bpmn_xml) : null;
  const mermaidMmd = typeof result?.mermaid_mmd === "string" ? maybeFixMojibake(result.mermaid_mmd) : null;
  const plantumlPuml = typeof result?.plantuml_puml === "string" ? maybeFixMojibake(result.plantuml_puml) : null;
  const renderWarning = typeof result?.render_warning === "string" ? maybeFixMojibake(result.render_warning) : null;
  const imageSrc = extractImageSrc(result);

  return (
    <section className="result-panel">
      <h3>Результат text-to-image</h3>
      {renderWarning ? <p className="state-text state-text--error">{renderWarning}</p> : null}
      {mermaidMmd ? <MermaidEditorPreview initialValue={mermaidMmd} /> : null}

      <div className="result-actions">
        {imageSrc ? (
          <a className="action-button action-button--secondary" href={imageSrc} download>
            Скачать PNG BPMN
          </a>
        ) : null}
        {bpmnXml ? (
          <a
            className="action-button action-button--secondary"
            href={textDownloadHref(bpmnXml, "application/xml")}
            download="diagram.bpmn"
          >
            Скачать .bpmn
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
      </div>

      {!mermaidMmd && imageSrc ? <img className="image-preview" src={imageSrc} alt="Сгенерированное изображение" /> : null}
      {!mermaidMmd && !imageSrc ? <ResultAsJson result={result} /> : null}
    </section>
  );
}
