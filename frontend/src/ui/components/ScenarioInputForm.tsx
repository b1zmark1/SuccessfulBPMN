import { useRef, useState, type ChangeEvent, type DragEvent, type FormEvent } from "react";
import type { SupportedJobType } from "../../shared/jobTypes";
import { SCENARIO_REGISTRY } from "../jobRegistry";

interface ScenarioInputFormProps {
  scenario: SupportedJobType;
  disabled: boolean;
  onSubmit: (meta: Record<string, unknown>) => Promise<void>;
}

interface ImageInfo {
  width: number;
  height: number;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error("Не удалось прочитать файл"));
    reader.readAsDataURL(file);
  });
}

function readImageInfo(file: File): Promise<ImageInfo | null> {
  return new Promise<ImageInfo | null>((resolve) => {
    const imageUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const result = { width: img.naturalWidth, height: img.naturalHeight };
      URL.revokeObjectURL(imageUrl);
      resolve(result);
    };
    img.onerror = () => {
      URL.revokeObjectURL(imageUrl);
      resolve(null);
    };
    img.src = imageUrl;
  });
}

function triggerDownload(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function ScenarioInputForm({ scenario, disabled, onSubmit }: ScenarioInputFormProps) {
  const config = SCENARIO_REGISTRY[scenario];
  const [value, setValue] = useState("");
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileDataUrl, setFileDataUrl] = useState<string | null>(null);
  const [fileInfo, setFileInfo] = useState<ImageInfo | null>(null);
  const [narratorMode, setNarratorMode] = useState<"text" | "table">("table");
  const [isDragActive, setIsDragActive] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const applyFile = async (file: File | null) => {
    setFormError(null);
    if (!file) {
      setFileName(null);
      setFileDataUrl(null);
      setFileInfo(null);
      return;
    }

    setFileName(file.name);
    try {
      const [dataUrl, info] = await Promise.all([readFileAsDataUrl(file), readImageInfo(file)]);
      setFileDataUrl(dataUrl);
      setFileInfo(info);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Не удалось прочитать файл";
      setFormError(message);
      setFileDataUrl(null);
      setFileInfo(null);
    }
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    await applyFile(event.target.files?.[0] ?? null);
  };

  const handleDrop = async (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragActive(false);
    if (disabled) return;
    const file = event.dataTransfer.files?.[0] ?? null;
    await applyFile(file);
  };

  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    if (!disabled) setIsDragActive(true);
  };

  const handleDragLeave = () => setIsDragActive(false);

  const handleUseSampleImage = async () => {
    if (!config.sampleImageUrl || disabled) return;

    try {
      setFormError(null);
      const response = await fetch(config.sampleImageUrl);
      if (!response.ok) throw new Error(`Не удалось загрузить пример диаграммы (${response.status})`);

      const blob = await response.blob();
      const sampleFileName = config.sampleImageUrl.split("/").pop() || "diagram1.jpg";
      const sampleFile = new File([blob], sampleFileName, { type: blob.type || "image/jpeg" });
      await applyFile(sampleFile);
      triggerDownload(blob, sampleFileName);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Не удалось загрузить пример диаграммы";
      setFormError(message);
    }
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFormError(null);

    try {
      const baseMeta = config.buildMeta({ textValue: value, fileName, fileDataUrl, narratorMode });
      const meta =
        scenario === "image_to_text"
          ? {
              ...baseMeta,
              image_meta: {
                file_name: fileName,
                width: fileInfo?.width ?? null,
                height: fileInfo?.height ?? null,
              },
            }
          : baseMeta;
      await onSubmit(meta);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Некорректный ввод";
      setFormError(message);
    }
  };

  return (
    <form className="input-form" onSubmit={submit}>
      <label className="input-form__label">
        {config.inputLabel}
        {config.inputKind === "textarea" ? (
          <textarea
            className="input-form__control input-form__control--textarea"
            placeholder={config.inputPlaceholder}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            disabled={disabled}
            rows={6}
          />
        ) : config.inputKind === "file" ? (
          <>
            <input
              ref={fileInputRef}
              className="input-form__native-file"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              disabled={disabled}
            />
            <label
              className={`dropzone ${isDragActive ? "dropzone--active" : ""} ${disabled ? "dropzone--disabled" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <strong>Перетащите изображение сюда</strong>
              <span>или нажмите, чтобы выбрать файл</span>
            </label>
            {config.sampleImageUrl ? (
              <button
                className="action-button action-button--secondary"
                type="button"
                onClick={handleUseSampleImage}
                disabled={disabled}
              >
                Пример диаграммы для запуска модели
              </button>
            ) : null}
            <label className="input-form__label">
              <select
                className="input-form__control"
                value={narratorMode}
                onChange={(event) => setNarratorMode(event.target.value as "text" | "table")}
                disabled={disabled}
              >
                <option value="table">Режим отображения: таблица</option>
                <option value="text">Режим отображения: текст</option>
              </select>
            </label>
            {fileName ? (
              <small>
                Выбран файл: {fileName}
                {fileInfo ? ` (${fileInfo.width}x${fileInfo.height})` : ""}
              </small>
            ) : (
              <small>PNG/JPG/WebP.</small>
            )}
          </>
        ) : (
          <input
            className="input-form__control"
            type="url"
            placeholder={config.inputPlaceholder}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            disabled={disabled}
          />
        )}
      </label>

      {config.infoNote ? <pre className="info-note">{config.infoNote}</pre> : null}
      {formError ? <p className="state-text state-text--error">{formError}</p> : null}

      <button className="action-button" type="submit" disabled={disabled}>
        {config.submitLabel}
      </button>
    </form>
  );
}
