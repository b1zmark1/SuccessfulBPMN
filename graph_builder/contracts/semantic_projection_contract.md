# Semantic Projection Contract (v1)

Этот документ фиксирует контракт слоя `graph -> semantic JSON`, который является входом для Narrator/LLM.

## 1) Схема и версия

- JSON Schema: `graph_builder/contracts/semantic_projection.schema.json`
- Обязательная версия: `meta.schema_version == "semantic-projection.v1"`

## 2) Структура payload

Корневой объект содержит ровно:
- `meta`
- `steps`

### `meta`

Обязательные поля:
- `schema_version` (строго `semantic-projection.v1`)
- `source_graph_schema_version` (строка, например `graph-builder.v1`)
- `direction` (`LR` или `TB`)
- `warnings` (массив строк)

### `steps[]`

Каждый шаг содержит ровно:
- `id` (строка, уникальный ID шага)
- `order` (целое число `>=1`, непрерывная последовательность `1..N`)
- `role` (`start|action|decision|parallel|end`)
- `text` (`string|null`)
- `next_step_ids` (массив уникальных ID следующих шагов)

## 3) Семантические ограничения

- В payload отсутствует геометрия (`bbox`, координаты, `center` и т.д.).
- В `steps[].next_step_ids` разрешены только ссылки на существующие `steps[].id`.
- `steps[].id` уникальны.
- Дополнительные поля не допускаются.

## 4) Политика неполных данных

- Пустой/неизвестный OCR: `steps[].text = null` (допустимо).
- При отсутствии предупреждений: `meta.warnings = []`.
- Отсутствие шага в semantic payload недопустимо: `steps` должен быть непустым.

