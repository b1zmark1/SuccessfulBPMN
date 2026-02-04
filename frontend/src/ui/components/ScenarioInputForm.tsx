import { useState, type ChangeEvent, type FormEvent } from "react";
import type { SupportedJobType } from "../../shared/jobTypes";
import { SCENARIO_REGISTRY } from "../jobRegistry";

interface ScenarioInputFormProps {
  scenario: SupportedJobType;
  disabled: boolean;
  onSubmit: (meta: Record<string, unknown>) => Promise<void>;
}

export function ScenarioInputForm({ scenario, disabled, onSubmit }: ScenarioInputFormProps) {
  const config = SCENARIO_REGISTRY[scenario];
  const [value, setValue] = useState("");
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileDataUrl, setFileDataUrl] = useState<string | null>(null);
  const [narratorMode, setNarratorMode] = useState<"text" | "table">("text");
  const [formError, setFormError] = useState<string | null>(null);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setFormError(null);

    if (!file) {
      setFileName(null);
      setFileDataUrl(null);
      return;
    }

    setFileName(file.name);

    try {
      const reader = new FileReader();
      const dataUrl = await new Promise<string>((resolve, reject) => {
        reader.onload = () => resolve(String(reader.result ?? ""));
        reader.onerror = () => reject(new Error("Не удалось прочитать файл"));
        reader.readAsDataURL(file);
      });
      setFileDataUrl(dataUrl);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Не удалось прочитать файл";
      setFormError(message);
      setFileDataUrl(null);
    }
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFormError(null);

    try {
      const meta = config.buildMeta({
        textValue: value,
        fileName,
        fileDataUrl,
        narratorMode,
      });
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
              className="input-form__control"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              disabled={disabled}
            />
            <label className="input-form__label">
              Режим Narrator
              <select
                className="input-form__control"
                value={narratorMode}
                onChange={(event) => setNarratorMode(event.target.value as "text" | "table")}
                disabled={disabled}
              >
                <option value="text">text</option>
                <option value="table">table</option>
              </select>
            </label>
            {fileName ? <small>Выбран файл: {fileName}</small> : <small>Поддерживаются изображения PNG/JPG/WebP.</small>}
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
      {formError ? <p className="state-text state-text--error">{formError}</p> : null}
      <button className="action-button" type="submit" disabled={disabled}>
        {config.submitLabel}
      </button>
    </form>
  );
}
