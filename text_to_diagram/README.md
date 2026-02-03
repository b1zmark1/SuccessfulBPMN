# Text -> Diagram (MVP)

Краткая сводка по шагам реализации. Этот файл — единая точка фиксации контрактов и прогресса.

## Шаг 0 (зафиксировано)

- backend endpoint не меняем (внутренний use-case);
- вход: `text` (ru), опционально `render.enabled`, `render.image_format=png|jpg`;
- выход: inline-артефакты, `status`, `issues`, `meta`;
- soft-fail модель: проблемы фиксируем в `issues`, конвейер продолжаем;
- CPU-only, целевой latency `<= 20s` (ориентир);
- PNG — основной формат, JPG — опциональная конвертация.

## Шаг 1 (зафиксировано): IR контракт `process-ir.v1`

- корень IR: `nodes`, `edges`, `lanes`, `meta`, `issues` (обязательные секции);
- совместимость с текущим проектом: сохраняем `id/type/role/text/container_id`, добавляем `lane_id`;
- versioning: `meta.schema_version = process-ir.v1`, несовместимые изменения -> `v2`;
- минимальная валидность:
  - уникальные id в `nodes/edges/lanes`,
  - ссылочная целостность `edges.from/to`, `lane_id`,
  - минимум один `start` и один `end`,
  - ацикличность только для иерархии `lanes`.

## Шаг 2 (реализовано): LLM-конвейер `text -> IR` (JSON-only)

- `text_to_diagram/llm_prompts.py`:
  - strict JSON-only prompt;
  - repair prompt для re-ask при невалидном ответе.
- `text_to_diagram/llm_config.py`:
  - CPU профиль (`temperature=0.0`, `max_tokens=2200`, `n_ctx=4096`, `timeout_sec=30`);
  - по умолчанию `max_reasks=2`;
  - включен quality gate (`quality_gate_enabled=true`) для отсечения "пустого" IR;
  - переиспользование существующего runtime/provider слоя.
- `text_to_diagram/llm_postprocess.py`:
  - безопасный JSON parse (direct/fenced/extract-object);
  - repair top-level секций и meta defaults под `process-ir.v1`.
- `text_to_diagram/llm_pipeline.py`:
  - `run_text_to_ir_pipeline(...)`;
  - re-ask fallback (`max_reasks`);
  - fallback IR + `issues` при полном провале парсинга (soft-fail).

## Шаг 3 (реализовано): валидация и нормализация IR

- `text_to_diagram/ir_validation.py`:
  - `validate_and_normalize_ir(ir, policy=...)` как единая точка подготовки IR для экспортеров;
  - `normalize_ir(...)`:
    - унификация id (`l1..`, `n1..`, `e1..`);
    - сортировка lane/node/edge для детерминизма;
    - автодополнение `meta` (`process-ir.v1`, `direction`, `source`, `language`);
    - repair некорректных/неполных полей в soft-fail режиме с `issues`.
  - `validate_ir(...)`:
    - ссылочная целостность (`edges.from/to`, `node.lane_id`);
    - доменные правила (`start/end`, цикл в иерархии lanes, self-loop);
    - валидационные предупреждения пишутся в `issues`.
- Политика ошибок:
  - по умолчанию hard-fail только для `IR_ROOT_NOT_OBJECT`;
  - остальное — warning/error в `issues` с продолжением конвейера.
- Тесты:
  - `tests/text_to_diagram/test_ir_validation.py`.

## Шаг 4 (реализовано): экспортер Mermaid (.mmd)

- `text_to_diagram/mermaid_exporter.py`:
  - `export_mermaid(normalized_ir)` как детерминированный экспорт в `.mmd`;
  - mapping ролей в Mermaid-формы:
    - `start/end` -> `(["..."])`,
    - `action` -> `["..."]`,
    - `decision` -> `{"..."}`,
    - `parallel` -> `{ "+" }`,
    - `inclusive` -> `{ "O" }`,
    - `event_intermediate` -> `(("..."))`,
    - unknown role -> fallback в `action` + `issues`.
  - поддержка lanes через `subgraph` по `lane.id`;
  - устойчивый порядок:
    - lanes: по `order`, затем `id`,
    - nodes: по `id`,
    - edges: по `(from, to, id)`.
  - проверки Mermaid-совместимости:
    - invalid direction -> default `LR` + issue;
    - неизвестные/битые ссылки в edge -> edge пропускается + issue;
    - дубли id узлов -> skip duplicate + issue;
    - label sanitation (escape/cleanup) для корректного текста.
- Тесты:
  - `tests/text_to_diagram/test_mermaid_exporter.py` (smoke + degraded cases).

## Шаг 5 (реализовано): экспортер BPMN 2.0 XML (Camunda-compatible)

- `text_to_diagram/bpmn_exporter.py`:
  - на текущем этапе BPMN строится из Mermaid-представления
    (`normalized IR -> mermaid -> mermaid_to_ir -> bpmn_exporter`);
  - `export_bpmn(normalized_ir)` с генерацией BPMN XML через `xml.etree.ElementTree`;
  - корректные namespace: `bpmn`, `camunda`, `xsi`;
  - минимальная Camunda-совместимая структура:
    - `bpmn:definitions`
    - `bpmn:process` (`isExecutable=false`)
    - `bpmn:laneSet` / `bpmn:lane` / `bpmn:flowNodeRef` (если lanes есть)
  - mapping узлов:
    - `start -> startEvent`
    - `end -> endEvent`
    - `action -> task`
    - `decision -> exclusiveGateway`
    - `parallel -> parallelGateway`
    - `inclusive -> inclusiveGateway`
    - `event_intermediate -> intermediateThrowEvent`
    - unknown role -> `task` + `issues`
  - mapping связей:
    - `sequential|conditional -> sequenceFlow`
    - `association -> association`
    - `message|unknown|unsupported -> sequenceFlow` + `issues`
  - для `conditional` добавляется `conditionExpression`;
  - добавляется BPMN DI:
    - `bpmndi:BPMNDiagram` / `bpmndi:BPMNPlane`,
    - `bpmndi:BPMNShape` + `dc:Bounds` для каждого flow node,
    - `bpmndi:BPMNEdge` + `di:waypoint` для каждого rendered flow;
  - deterministic layout MVP:
    - `LR`: x по порядку узлов, y по lane;
    - `TB`: y по порядку узлов, x по lane;
    - размеры: event 36x36, gateway 50x50, task 140x80;
  - детерминированный порядок lanes/nodes/edges и incoming/outgoing по узлам;
  - проверка well-formed XML и DI-валидация
    (`BPMNDiagram`, `BPMNPlane`, количество shape/edge).
- `text_to_diagram/mermaid_to_ir.py`:
  - детерминированный парсер Mermaid в IR-подмножество для BPMN-экспорта.
- Тесты:
  - `tests/text_to_diagram/test_bpmn_exporter.py`.

## Шаг 6 (реализовано): экспортер PlantUML (.puml)

- `text_to_diagram/plantuml_exporter.py`:
  - `export_plantuml(normalized_ir)` как детерминированный текстовый экспортер;
  - принятый профиль MVP: `state-flow.v1` (PlantUML state-style для явных узлов и связей);
  - mapping ролей:
    - `start -> state <<start>>`
    - `end -> state <<end>>`
    - `action -> state`
    - `decision -> state <<decision>>`
    - `parallel -> state <<parallel>>`
    - `inclusive -> state <<inclusive>>`
    - `event_intermediate -> state <<event>>`
    - unknown role -> fallback в обычный `state` + `issues`.
  - lanes через контейнеры:
    - `state "LaneName" as lane_<id> { ... }`
  - mapping связей:
    - `sequential/conditional -> -->` (для `conditional` добавляется label);
    - `message/association -> ..>`;
    - unknown/unsupported -> fallback в `-->` + `issues`.
  - детерминированный порядок lanes/nodes/edges;
  - graceful degradation для неподдерживаемых/битых конструкций через `issues`.
- Тесты:
  - `tests/text_to_diagram/test_plantuml_exporter.py`.

## Шаг 7 (реализовано): render-layer PNG/JPG

- `text_to_diagram/render_layer.py`:
  - единая точка `render_artifact_to_image(artifact_type, artifact_text, image_format)`;
  - поддерживаемые входы:
    - `artifact_type=mermaid` (`.mmd`)
    - `artifact_type=plantuml` (`.puml`)
    - `artifact_type=bpmn` (`.bpmn`)
  - выход (inline-ready):
    - `image_png_base64`
    - `image_jpg_base64` (только при `image_format=jpg`)
    - `issues`, `meta`, `status`.
- Mermaid рендер:
  - локальный `mmdc` (`mermaid-cli`) -> PNG.
- PlantUML рендер:
  - локальный `plantuml` CLI или fallback `java -jar $PLANTUML_JAR` -> PNG.
- BPMN рендер:
  - локальный headless путь через `node tools/bpmn_dataset/render_bpmn_svg.js` -> SVG;
  - затем `cairosvg` конвертирует SVG -> PNG.
- PNG -> JPG:
  - пост-конвертация через Pillow (`RGB`, JPEG quality=90).
- Fallback-политика:
  - любые проблемы рендера/отсутствие рендерера -> `status=degraded`, изображения `null`, причина в `issues`;
  - основной pipeline не ломается.
  - для BPMN добавлена явная диагностика `BPMN_RENDER_NO_DIAGRAM`, если в XML нет BPMN DI.
- Тесты:
  - `tests/text_to_diagram/test_render_layer.py`.

## Шаг 8 (реализовано): внутренняя оркестрация use-case

- `text_to_diagram/llm_service.py`:
  - `TextToDiagramLLMService` как общий сервисный слой для text->IR;
  - переиспользует один LLM provider-инстанс в рамках процесса (без backend-изменений).
- `text_to_diagram/orchestrator.py`:
  - `run_text_to_diagram_use_case(...)`:
    - `text -> LLM -> validate/normalize -> exporters -> (опц.) render`;
    - формирует единый inline-ответ с артефактами:
      - `ir_json`, `mermaid_mmd`, `bpmn_xml`, `plantuml_puml`,
      - `image_png_base64`, `image_jpg_base64`;
    - агрегирует `issues` по всем стадиям, dedupe по ключу issue;
    - поддерживает soft-fail модель (`status=degraded`, но с максимально полным результатом).
  - наблюдаемость:
    - `trace_id`, `trace` по стадиям;
    - `stage_durations_ms`, `total_duration_ms`;
    - `stage_details` (подробные метаданные/диагностика по стадиям, включая render);
    - `error_codes`.
  - trace теперь логирует для каждой стадии:
    - `status` (`started|completed|failed`),
    - `duration_ms`,
    - `input_summary`,
    - `output_summary`,
    - `error_code/error_message` (если есть).
  - `meta.first_fail`:
    - первая стадия сверху вниз, где `status=failed`,
    - или `status=completed`, но появились `severity=error` issues.
  - для BPMN перед render в `render_artifact.input_summary` логируются:
    - `has_bpmn_diagram_tag`, `has_bpmn_plane_tag`,
    - `bpmn_shape_count`, `bpmn_edge_count`,
    - `process_node_count`, `process_edge_count`.
- Тесты:
  - `tests/text_to_diagram/test_orchestrator.py`.

## Шаг 9 (реализовано): тестирование и приемка MVP

- Unit-тесты покрывают:
  - `llm_prompts`, `llm_postprocess`, `llm_pipeline`,
  - `ir_validation`,
  - `mermaid_exporter`, `bpmn_exporter`, `plantuml_exporter`,
  - `render_layer`, `orchestrator`.
- Интеграционный E2E:
  - `tests/text_to_diagram/test_e2e_integration.py` (полный путь `text -> IR -> artifacts`).
- Acceptance-набор русскоязычных сценариев:
  - `tests/text_to_diagram/test_acceptance_suite.py`
  - кейсы: `simple_flow_ru`, `branching_flow_ru`, `lane_flow_ru`.
- Проверка latency-ориентира (<=20 сек):
  - `text_to_diagram/benchmark.py`
  - `tests/text_to_diagram/test_latency_benchmark.py` (синтетический прогон на fake-provider).
- Критерии приемки MVP (чек-лист):
  - есть единый orchestrator и единый IR (`nodes/edges/lanes/meta/issues`);
  - обязательные текстовые артефакты генерируются inline (`.mmd`, `.bpmn`, `.puml`);
  - render-layer возвращает PNG/JPG в base64 с soft-fail fallback;
  - все тесты `tests/text_to_diagram` проходят.

## Текущий статус

- Этап 0: done
- Этап 1: done
- Этап 2: done
- Этап 3: done
- Этап 4: done
- Этап 5: done
- Этап 6: done
- Этап 7: done
- Этап 8: done
- Этап 9: done
