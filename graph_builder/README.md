# Graph Builder

## Назначение

`graph_builder` преобразует детекции из `ensemble` JSON (YOLOX + EasyOCR) в notation-agnostic граф процесса, пригодный для передачи в LLM.

Модуль спроектирован как детерминированный пошаговый pipeline:
- устойчив к шуму детекций;
- опирается на геометрию (классы — только подсказки);
- готов к будущему OCR (поле `text` у узлов уже есть).

Контракт входа/выхода зафиксирован в:
- `graph_builder/contracts/README.md`
- `graph_builder/contracts/input_ensemble.schema.json`
- `graph_builder/contracts/output_graph.schema.json`

---

## Что подается на вход

Вход — `ensemble` JSON формата:
- `image`:
  - `file_name`
  - `width`, `height`
  - (опц.) `path`, `resize_ratio`, `geom_angle_deg`
- `detections[]`:
  - `class_name`
  - `bbox_xyxy` `[x1,y1,x2,y2]`
  - `source` (`yolox`/`easyocr`)
  - (опц.) `class_id`, `score`, `bbox_xywh`, ...
- (опц.) `meta`

Пример: `results/benchmark/onnxruntime/detections/BPMN_ensemble.json`

---

## Что выдается на выход

Финальный JSON (пункт 11) содержит только:
- `meta`
  - `schema_version` (сейчас `graph-builder.v1`)
  - `direction` (`LR`/`TB`)
  - `warnings` (краткие диагностические заметки)
- `nodes[]`
  - `id`
  - `type` (`shape|container|text|flow`)
  - `bbox`
  - `center`
  - `role` (`action|decision|start|end|unknown`)
  - `container_id` (`string|null`)
  - `text` (`string|null`, сейчас обычно `null`)
- `edges[]`
  - `from`
  - `to`
  - `type` (`sequential|conditional|unknown`)

Это строго соответствует контракту в `graph_builder/contracts/README.md`.

---

## Что реализовано (по шагам)

### 1) Контракты
- `contracts/*`: зафиксированы схемы и defaults.

### 2) Нормализация
- `normalize.py`
- Парсинг/очистка bbox, клиппинг в границы, отбрасывание невалидных объектов, deterministic dedup.

### 3) Примитивные группы
- `grouping.py`
- Разбиение на `process_shapes/flows/containers/texts/unknown`.
- Добавлены reliability hints по классу/источнику и provenance.

### 4) Направление процесса
- `direction.py`
- Оценка `LR/TB` по геометрии shape + flow, confidence, fallback `LR`, trace в `meta`.

### 5) Узлы
- `nodes.py`
- Построение узлов графа с обязательными полями контракта.
- `text` и `container_id` присутствуют всегда.

### 6) Контейнеры
- `containers.py`
- Назначение `container_id` по геометрическому включению.
- Nested контейнеры отключены (по принятому решению).

### 7) Candidate edges
- `edge_candidates.py`
- Генерация направленных кандидатов по геометрии + flow hints, со score и feature trace.

### 8) Финализация edges
- `edges.py`
- Порог/валидация/ограничение fan-out, типизация (`conditional/sequential/unknown`), dedup.

### 9) Text hooks
- `text_hooks.py`
- Привязка text-узлов к целевым узлам (hooks), `text` placeholder enforcement.

### 9.1) Merge text boxes
- `text_merge.py`
- Склейка соседних text bbox в один region (axis-aware LR/TB), с merge provenance.

### 9.2) Title hints
- `title_hints.py`
- Эвристики `pool_title`, `lane_title`, `diagram_title_candidate`.

### 10) Diagnostics
- `diagnostics.py`
- Сбор uncertainty notes + краткие `meta.warnings`, debug слой по узлам/ребрам.

### 11) Сериализация
- `serialize.py`
- Финальный строгий output JSON по контракту.

### 12) Тесты
- `tests/graph_builder/*`, `tests/fixtures/graph_builder/*`
- Fixture/regression/failure-mode тесты.

---

## Как использовать (pipeline)

Ниже базовый программный вызов всех этапов:

```python
from graph_builder import (
    normalize_ensemble_input,
    group_normalized_detections,
    infer_process_direction,
    build_graph_nodes,
    assign_container_hierarchy,
    build_edge_candidates,
    finalize_edges,
    merge_adjacent_text_nodes,
    attach_text_placeholders_and_hooks,
    assign_title_hints,
    build_uncertainty_and_diagnostics,
    serialize_graph_output,
)

def build_graph(ensemble_json: dict) -> dict:
    x = normalize_ensemble_input(ensemble_json)
    x = group_normalized_detections(x)
    x = infer_process_direction(x)
    x = build_graph_nodes(x)
    x = assign_container_hierarchy(x)
    x = build_edge_candidates(x)
    x = finalize_edges(x)
    x = merge_adjacent_text_nodes(x)
    x = attach_text_placeholders_and_hooks(x)
    x = assign_title_hints(x)
    x = build_uncertainty_and_diagnostics(x)
    return serialize_graph_output(x)
```

---

## Важные архитектурные принципы

- Геометрия приоритетнее классов.
- Классы детектора используются как hints, не как истина.
- Контейнеры — иерархия, не процессные шаги.
- Все решения детерминированы (стабильные сортировки/пороги).
- Выход готов к OCR-расширению: поле `text` уже встроено.
