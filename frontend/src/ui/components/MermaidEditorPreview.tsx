import { useEffect, useMemo, useState } from "react";
import mermaid from "mermaid";

interface MermaidEditorPreviewProps {
  initialValue: string;
}

let mermaidInitialized = false;

function ensureMermaidInitialized(): void {
  if (mermaidInitialized) return;
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "default",
    flowchart: {
      htmlLabels: false,
    },
  });
  mermaidInitialized = true;
}

function textDownloadHref(content: string, mime: string): string {
  return `data:${mime};charset=utf-8,${encodeURIComponent(content)}`;
}

function parseSvgSize(svgMarkup: string): { width: number; height: number } | null {
  const parser = new DOMParser();
  const document = parser.parseFromString(svgMarkup, "image/svg+xml");
  const svg = document.documentElement;
  if (svg.tagName.toLowerCase() !== "svg") return null;

  const widthAttr = svg.getAttribute("width");
  const heightAttr = svg.getAttribute("height");
  const viewBoxAttr = svg.getAttribute("viewBox");

  const parseDimension = (value: string | null): number | null => {
    if (!value) return null;
    const parsed = Number.parseFloat(value.replace("px", ""));
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  };

  const width = parseDimension(widthAttr);
  const height = parseDimension(heightAttr);
  if (width && height) return { width, height };

  if (!viewBoxAttr) return null;
  const parts = viewBoxAttr
    .split(/\s+/)
    .map((part) => Number.parseFloat(part))
    .filter((part) => Number.isFinite(part));
  if (parts.length !== 4) return null;

  const vbWidth = parts[2];
  const vbHeight = parts[3];
  if (vbWidth === undefined || vbHeight === undefined || vbWidth <= 0 || vbHeight <= 0) return null;
  return { width: vbWidth, height: vbHeight };
}

function sanitizeSvgForCanvas(svgMarkup: string): string {
  const parser = new DOMParser();
  const doc = parser.parseFromString(svgMarkup, "image/svg+xml");
  const svg = doc.documentElement;
  if (svg.tagName.toLowerCase() !== "svg") {
    return svgMarkup;
  }

  // Prevent canvas taint by stripping only external references,
  // keep labels/styles so exported PNG still contains text.
  svg.querySelectorAll("[href],[xlink\\:href]").forEach((node) => {
    const href = node.getAttribute("href") ?? node.getAttribute("xlink:href") ?? "";
    if (/^https?:\/\//i.test(href)) {
      node.removeAttribute("href");
      node.removeAttribute("xlink:href");
    }
  });
  svg.querySelectorAll("script").forEach((node) => node.remove());

  return new XMLSerializer().serializeToString(svg);
}

async function exportSvgToPng(svgMarkup: string, fileName: string): Promise<void> {
  const safeSvgMarkup = sanitizeSvgForCanvas(svgMarkup);
  const size = parseSvgSize(safeSvgMarkup);
  if (!size) throw new Error("Не удалось определить размеры диаграммы Mermaid.");

  const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(size.width * pixelRatio);
  canvas.height = Math.round(size.height * pixelRatio);
  canvas.style.width = `${size.width}px`;
  canvas.style.height = `${size.height}px`;

  const context = canvas.getContext("2d");
  if (!context) throw new Error("Не удалось создать canvas для экспорта PNG.");

  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, size.width, size.height);

  const svgBlob = new Blob([safeSvgMarkup], { type: "image/svg+xml;charset=utf-8" });
  const svgUrl = URL.createObjectURL(svgBlob);
  const image = new Image();

  try {
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("Не удалось отрисовать Mermaid в изображение."));
      image.src = svgUrl;
    });
  } finally {
    URL.revokeObjectURL(svgUrl);
  }

  context.drawImage(image, 0, 0, size.width, size.height);

  let pngBlob: Blob | null = null;
  try {
    pngBlob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/png");
    });
  } catch {
    throw new Error("Браузер заблокировал экспорт PNG (tainted canvas).");
  }
  if (!pngBlob) throw new Error("Не удалось сформировать PNG файл.");

  const pngUrl = URL.createObjectURL(pngBlob);
  const link = document.createElement("a");
  link.href = pngUrl;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(pngUrl);
}

export function MermaidEditorPreview({ initialValue }: MermaidEditorPreviewProps) {
  const [mmd, setMmd] = useState(initialValue);
  const [svgMarkup, setSvgMarkup] = useState<string | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [isRendering, setIsRendering] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  useEffect(() => {
    setMmd(initialValue);
  }, [initialValue]);

  useEffect(() => {
    if (!mmd.trim()) {
      setSvgMarkup(null);
      setRenderError("Mermaid-код пустой.");
      return;
    }

    ensureMermaidInitialized();
    setIsRendering(true);
    setRenderError(null);

    let cancelled = false;
    const timeout = window.setTimeout(() => {
      const renderId = `mermaid-preview-${crypto.randomUUID()}`;
      mermaid
        .render(renderId, mmd)
        .then((result) => {
          if (cancelled) return;
          setSvgMarkup(result.svg);
          setRenderError(null);
        })
        .catch((error: unknown) => {
          if (cancelled) return;
          const message = error instanceof Error ? error.message : "Ошибка рендера Mermaid.";
          setSvgMarkup(null);
          setRenderError(message);
        })
        .finally(() => {
          if (!cancelled) setIsRendering(false);
        });
    }, 250);

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [mmd]);

  const canExportPng = useMemo(() => !!svgMarkup && !renderError, [svgMarkup, renderError]);

  return (
    <section className="mermaid-panel">
      <h4>Mermaid (.mmd)</h4>
      <textarea
        className="input-form__control input-form__control--textarea mermaid-editor"
        value={mmd}
        onChange={(event) => setMmd(event.target.value)}
        rows={10}
      />

      <div className="result-actions">
        <a className="action-button action-button--secondary" href={textDownloadHref(mmd, "text/plain")} download="diagram.mmd">
          Скачать .mmd
        </a>
        <button
          className="action-button action-button--secondary"
          type="button"
          onClick={() => {
            setExportError(null);
            if (!svgMarkup) {
              setExportError("Нет SVG для экспорта.");
              return;
            }
            exportSvgToPng(svgMarkup, "diagram.png").catch((error: unknown) => {
              const rawMessage = error instanceof Error ? error.message : "Не удалось скачать PNG.";
              const isTainted = /tainted canvas/i.test(rawMessage);
              // Fallback: always download SVG if PNG export fails.
              const link = document.createElement("a");
              link.href = textDownloadHref(svgMarkup, "image/svg+xml");
              link.download = "diagram.svg";
              document.body.appendChild(link);
              link.click();
              link.remove();
              setExportError(
                isTainted
                  ? "PNG заблокирован браузером, скачан SVG."
                  : `PNG не получился, скачан SVG. Причина: ${rawMessage}`,
              );
            });
          }}
          disabled={!canExportPng}
        >
          Скачать PNG из .mmd
        </button>
      </div>

      {isRendering ? <p className="state-text">Обновляем live-preview Mermaid...</p> : null}
      {renderError ? <p className="state-text state-text--error">Ошибка Mermaid: {renderError}</p> : null}
      {exportError ? <p className="state-text state-text--error">{exportError}</p> : null}

      {svgMarkup ? (
        <div className="mermaid-preview" dangerouslySetInnerHTML={{ __html: svgMarkup }} />
      ) : null}
    </section>
  );
}
