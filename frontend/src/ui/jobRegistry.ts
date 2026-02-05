import type { SupportedJobType } from "../shared/jobTypes";

export interface ScenarioConfig {
  jobType: SupportedJobType;
  title: string;
  description: string;
  infoNote?: string;
  sampleImageUrl?: string;
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
  if (!trimmed) throw new Error(`${fieldName} is required`);
  return trimmed;
}

const TEXT_TO_IMAGE_PROMPT_NOTE = `Пример промпта:
Сформируй процесс строго по описанию ниже. Ничего не обобщай и не заменяй на шаблонные фразы.

Участники:
- Клиент
- Оператор
- Отдел безопасности

Шаги:
1) Клиент отправляет заявку на подключение услуги.
2) Оператор проверяет полноту данных.
3) Если данных недостаточно:
   - Оператор запрашивает уточнение у клиента.
   - После получения уточнения вернуться к шагу 2.
4) Если данные корректны:
   - Оператор передает заявку в отдел безопасности.
5) Отдел безопасности принимает решение:
   - Одобрить
   - Отклонить
6) Если одобрено:
   - Оформить договор.
   - Отправить клиенту подтверждение.
7) Если отклонено:
   - Отправить клиенту уведомление с причиной отказа.
8) После отправки любого уведомления процесс завершается.

Требования:
- Сохрани формулировки шагов близко к тексту.
- Добавь явные ветки "если/иначе".
- Не используй названия вида "Шаг 1", "n1", "обработка данных" вместо бизнес-смысла.
- Если есть роли, используй lanes по ролям.`;

export const SCENARIO_REGISTRY: Record<SupportedJobType, ScenarioConfig> = {
  image_to_text: {
    jobType: "image_to_text",
    title: "Изображение в текст",
    description: "Загрузите изображение, сервис извлечет текст.",
    sampleImageUrl: "/preprocanddetect/diagram1.jpg",
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
    description: "Введите промпт, сервис сгенерирует изображение.",
    infoNote: TEXT_TO_IMAGE_PROMPT_NOTE,
    inputLabel: "Промпт",
    inputPlaceholder: "Опишите, какое изображение нужно сгенерировать",
    inputKind: "textarea",
    submitLabel: "Сгенерировать изображение",
    buildMeta: ({ textValue }) => {
      const prompt = requireValue(textValue, "Промпт");
      return { prompt, promt: prompt };
    },
  },
};

export const SCENARIOS = Object.values(SCENARIO_REGISTRY);
