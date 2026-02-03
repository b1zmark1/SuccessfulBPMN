# Graph Builder I/O Contract (v1)

Этот документ фиксирует контракт входа/выхода для первого этапа Graph Builder.

## 1) Входной контракт

- Источник: `ensemble` JSON после детекции (YOLOX + EasyOCR).
- Базовая схема: `graph_builder/contracts/input_ensemble.schema.json`.
- Минимально обязательные поля:
  - `image.file_name`
  - `image.width`, `image.height`
  - `detections[]` с полями:
    - `class_name`
    - `bbox_xyxy` (ровно 4 числа)
    - `source` (`yolox` или `easyocr`)
- Допустимые расширения:
  - любые дополнительные поля в `image`, `meta`, `detections[]` сохраняются и не ломают валидацию.

## 2) Выходной контракт

- Базовая схема: `graph_builder/contracts/output_graph.schema.json`.
- Обязательные секции:
  - `meta`
  - `nodes`
  - `edges`
- `meta` обязательно содержит:
  - `schema_version` (строка, напр. `graph-builder.v1`)
  - `direction` (`LR` или `TB`)
  - `warnings` (массив строк, может быть пустым)

## 3) Политика обязательных/опциональных полей

- Для узла (`nodes[]`) обязательны:
  - `id`
  - `type` (`shape|container|text|flow`)
  - `bbox` (массив из 4 чисел `[x1,y1,x2,y2]`)
  - `center` (массив из 2 чисел `[cx,cy]`)
  - `role` (`action|decision|start|end|unknown`)
  - `container_id` (строка или `null`)
  - `text` (строка или `null`)
- Для ребра (`edges[]`) обязательны:
  - `from`
  - `to`
  - `type` (`sequential|conditional|unknown`)

## 4) Значения по умолчанию

- `meta.direction`: `LR` (если направление не удалось определить уверенно).
- `meta.warnings`: `[]`.
- `nodes[].container_id`: `null`.
- `nodes[].text`: `null` (до этапа OCR).

## 5) Версионирование

- Поле `meta.schema_version` обязательно.
- Текущая версия контракта: `graph-builder.v1`.
- Любое несовместимое изменение структуры должно увеличивать версию (`v2`, `v3`, ...).
