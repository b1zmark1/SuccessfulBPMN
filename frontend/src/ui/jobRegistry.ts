import type { SupportedJobType } from "../shared/jobTypes";

export interface ScenarioConfig {
  jobType: SupportedJobType;
  title: string;
  description: string;
  infoNote?: string;
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

const TEXT_TO_IMAGE_PROMPT_NOTE =
  "Пример промпта:\n" +
  "Сформируй процесс строго по описанию ниже. Ничего не обобщай и не заменяй на шаблонные фразы.\n\n" +
  "Участники:\n" +
  "- Клиент\n" +
  "- Оператор\n" +
  "- Отдел безопасности\n\n" +
  "Шаги:\n" +
  "1) Клиент отправляет заявку на подключение услуги.\n" +
  "2) Оператор проверяет полноту данных.\n" +
  "3) Если данных недостаточно:\n" +
  "   - Оператор запрашивает уточнение у клиента.\n" +
  "   - После получения уточнения вернуться к шагу 2.\n" +
  "4) Если данные корректны:\n" +
  "   - Оператор передает заявку в отдел безопасности.\n" +
  "5) Отдел безопасности принимает решение:\n" +
  "   - Одобрить\n" +
  "   - Отклонить\n" +
  "6) Если одобрено:\n" +
  "   - Оформить договор.\n" +
  "   - Отправить клиенту подтверждение.\n" +
  "7) Если отклонено:\n" +
  "   - Отправить клиенту уведомление с причиной отказа.\n" +
  "8) После отправки любого уведомления процесс завершается.\n\n" +
  "Требования:\n" +
  "- Сохрани формулировки шагов близко к тексту.\n" +
  '- Добавь явные ветки "если/иначе".\n' +
  '- Не используй названия вида "Шаг 1", "n1", "обработка данных" вместо бизнес-смысла.\n' +
  "- Если есть роли, используй lanes по ролям.";

export const SCENARIO_REGISTRY: Record<SupportedJobType, ScenarioConfig> = {
  image_to_text: {
    jobType: "image_to_text",
    title: "Изображение в текст",
    description: "Загрузите изображение, сервис извлечет текст.",
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
