import type { SupportedJobType } from "../shared/jobTypes";

export interface ScenarioConfig {
  jobType: SupportedJobType;
  title: string;
  description: string;
  inputLabel: string;
  inputPlaceholder: string;
  inputKind: "url" | "textarea" | "file";
  submitLabel: string;
  buildMeta: (input: {
    textValue: string;
    fileName: string | null;
    fileDataUrl: string | null;
    narratorMode: "text" | "table";
  }) => Record<string, unknown>;
}

function requireValue(value: string, fieldName: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error(`${fieldName} is required`);
  }
  return trimmed;
}

export const SCENARIO_REGISTRY: Record<SupportedJobType, ScenarioConfig> = {
  image_to_text: {
    jobType: "image_to_text",
    title: "Изображение в текст",
    description: "Загрузите изображение, сервис извлечет текст асинхронно.",
    inputLabel: "Файл изображения",
    inputPlaceholder: "",
    inputKind: "file",
    submitLabel: "Распознать текст",
    buildMeta: ({ fileDataUrl, narratorMode }) => ({
      image_url: requireValue(fileDataUrl ?? "", "Файл изображения"),
      narrator_mode: narratorMode,
    }),
  },
  text_to_image: {
    jobType: "text_to_image",
    title: "Текст в изображение",
    description: "Введите промпт, сервис сгенерирует изображение асинхронно.",
    inputLabel: "Промпт",
    inputPlaceholder: "Опишите, какое изображение нужно сгенерировать",
    inputKind: "textarea",
    submitLabel: "Сгенерировать изображение",
    buildMeta: ({ textValue }) => {
      const prompt = requireValue(textValue, "Промпт");
      // Keep backward compatibility if workers still read old key.
      return { prompt, promt: prompt };
    },
  },
};

export const SCENARIOS = Object.values(SCENARIO_REGISTRY);
