# Narrator module

Эта папка выделена под код Narrator-слоя (генерация финального человеко-читаемого текста из semantic JSON).

Текущий входной контракт для Narrator:
- `graph_builder/contracts/semantic_projection.schema.json`
- `graph_builder/contracts/semantic_projection_contract.md`

## Конфиг Narrator-политик (v1)

Модуль `narrator/policies.py` задаёт runtime-конфиг с дефолтами:
- `max_sentences=10`
- `missing_text_policy="generalize"`
- `branches_policy="cover_all"`
- `output_format="narrative"` (`narrative|table`)

Поддерживается переопределение параметров через `resolve_narrator_policy(overrides)`.

Для проброса в общий результат предусмотрено:
- `build_narrator_meta(policy) -> {"applied_policy": ...}`

## Prompt-пакет Narrator (v1)

Модуль `narrator/prompts.py` формирует версионируемый prompt-пакет:
- `build_narrator_prompts(semantic_payload, policy, cfg)` ->  
  `{"prompt_version", "system_prompt", "user_prompt"}`
- `build_prompt_meta(prompt_version)` -> `{"prompt_version": ...}`

Требования, которые закладываются в prompt:
- опора только на входной semantic JSON;
- соблюдение порядка шагов и связей;
- обязательное покрытие всех ветвей;
- поддержка 2 форматов вывода:
  - `narrative`: связный русский текст;
  - `table`: plain-text таблица `Шаг | Роль`.
- для режима `table`: роль берется из `lane/owner/participants` (если есть во входе); если нет — LLM пробует вывести роль по контексту, иначе `Не указано`.

## Архитектура plug-and-play для смены модели

Цель: обеспечить замену модели с минимальными изменениями в коде.

### Неподвижные инварианты

- Вход Narrator: только `semantic_payload` (по текущему контракту).
- Выход Narrator: готовый текст + `narrator_meta`.
- Один вызов LLM на один граф.
- Prompt-пакет и post-processing не зависят от конкретной модели.

### Минимальная структура модулей

- `narrator/orchestrator.py` — единая точка запуска `run_narration(...)`.
- `narrator/providers/base.py` — интерфейс провайдера модели.
- `narrator/providers/llama_cpp_provider.py` — GGUF/CPU провайдер.
- `narrator/providers/factory.py` — выбор провайдера по конфигу.
- `narrator/config.py` — конфиг Narrator, модели и runtime.

### Контракт провайдера модели

Каждый провайдер обязан реализовывать единый метод:
- `generate(system_prompt, user_prompt, runtime_config) -> str`

Это единственная точка, где модель-специфичная логика допустима.

### Конфиг, который должен быть внешне управляемым

- `provider` (например: `llama_cpp`)
- `model_path`
- `n_ctx`
- `n_threads`
- `n_batch`
- `temperature`
- `max_tokens`
- `cpu_only`

## Runtime инференса Qwen2.5-7B-Instruct-GGUF на CPU (пункт 5)

Реализованные модули:
- `narrator/config.py` — runtime-конфиг и его валидация:
  - `resolve_runtime_config(overrides)`
  - `NarratorRuntimeConfig`
- `narrator/providers/base.py` — единый интерфейс провайдера
- `narrator/providers/llama_cpp_provider.py` — CPU-провайдер на `llama-cpp-python`
- `narrator/providers/factory.py` — выбор провайдера
- `narrator/runtime.py` — `run_single_llm_call(prompt_pack, runtime_cfg)`

Что фиксируется в runtime:
- provider: `llama_cpp`
- cpu-only профиль (`n_gpu_layers=0`)
- параметры генерации: `n_ctx`, `n_threads`, `n_batch`, `temperature`, `max_tokens`, `timeout_sec`
- метаданные результата: `single_call=true`, `duration_ms`, `provider`, `model_path`, `prompt_version`

Гарантия одного вызова LLM на один граф:
- `run_single_llm_call(...)` делает ровно один вызов `provider.generate(...)`.

## Отказоустойчивость и fallback (пункт 7)

В `narrator/orchestrator.py` введена деградационная ветка ответа:
- при любой критической ошибке возвращается техническая заглушка для пользователя;
- `status` устанавливается в `degraded`;
- причина фиксируется в `errors[]` и в `trace`.

Коды ошибок по этапам:
- `PROJECTION_ERROR`
- `SEMANTIC_CONTRACT_ERROR`
- `POLICY_CONFIG_ERROR`
- `PROMPT_BUILD_ERROR`
- `RUNTIME_CONFIG_ERROR`
- `LLM_TIMEOUT`
- `LLM_CONTEXT_OVERFLOW`
- `LLM_INVALID_OUTPUT`
- `LLM_RUNTIME_ERROR`
- `POSTPROCESS_ERROR`
- `OUTPUT_GUARDRAIL_VIOLATION`

## Post-processing и guardrails финального текста (пункт 8)

Модуль `narrator/postprocess.py` применяет единые проверки и нормализацию:
- мягкая нормализация (очистка markdown-маркеров списков/заголовков, пробелов, пустых строк);
- hard-check, что выход не является JSON/markdown code fence;
- запрет терминов BPMN/UML;
- запрет внутренних фраз модели (например, self-reference формулировок).

Поведение в orchestrator:
- при мягких нарушениях текст авто-нормализуется и возвращается как `status=ok`;
- при hard-нарушениях включается деградация с технической заглушкой (`OUTPUT_GUARDRAIL_VIOLATION`).

## Наблюдаемость и диагностические метрики (пункт 9)

В `narrator/observability.py` реализованы:
- `generate_trace_id()` — короткий `trace_id` для корреляции запуска;
- `build_observability_meta(...)` — безопасный observability-блок без утечки полного текста.

В `orchestrator` этот блок возвращается в `meta.observability` и содержит:
- `trace_id`
- `status` (`ok`/`degraded`)
- `degraded` (bool)
- `stage_durations_ms`
- `total_duration_ms`
- `semantic_schema_version`
- `source_graph_schema_version`
- `provider`
- `prompt_version`
- `error_codes`

Принцип безопасности:
- в observability-данные не пишется полный сгенерированный текст и не дублируется входной semantic JSON.

## Acceptance-набор для projection + narrator (пункт 10)

Добавлен acceptance-слой проверки:
- `narrator/acceptance.py` — `evaluate_acceptance_case(case_name, semantic_payload, narration_result)`
- `tests/narrator_layer/test_acceptance_suite.py` — прогон репрезентативных кейсов `graph -> semantic -> text`

Покрываемые сценарии:
- `linear_flow`
- `multiple_branches`
- `parallel_block`
- `empty_text`
- `noise_incomplete`

Чеклист качества для каждого кейса:
- допустимый статус (`ok` или `degraded`)
- текст/заглушка не пустые
- текст выглядит русскоязычным
- выход не JSON
- нет терминов BPMN/UML
- ветвления отражены в тексте (эвристика покрытия всех ветвей)

## Интеграционная валидация end-to-end (пункт 11)

Добавлен модуль:
- `narrator/e2e_validation.py` — `run_e2e_validation(cases, runtime_overrides)`

Возвращаемый отчёт содержит:
- `summary` с метриками:
  - `total_cases`
  - `ok_cases`
  - `degraded_cases`
  - `success_rate`
  - `fallback_rate`
  - `text_or_stub_rate`
  - `projection_valid_rate_ok`
  - `single_call_rate_ok`
  - `avg_total_duration_ms`
- `results` с результатом по каждому кейсу

Назначение:
- подтвердить стабильность `projection` (валидная semantic schema в успешных кейсах),
- подтвердить один LLM-вызов на граф в успешных кейсах (`single_call=true`),
- подтвердить, что система всегда возвращает либо финальный текст, либо техническую заглушку.

### Целевой пайплайн orchestrator

1. Валидация `semantic_payload` по контракту.
2. Применение Narrator-политики.
3. Формирование prompt-пакета.
4. Вызов `provider.generate(...)`.
5. Post-processing/guardrails.
6. Возврат `text` и `narrator_meta`.

Реализованные точки входа:
- `run_narration(graph_payload, ...)` — строго `graph -> semantic -> text`
- `run_narration_from_ensemble(ensemble_payload, ...)` — `ensemble -> graph -> semantic -> text`

Обе функции возвращают единый DTO:
- `text`
- `status`
- `errors`
- `meta` (включая `projection`, `narrator`, `trace`)

### Что должно попадать в `narrator_meta`

- `applied_policy`
- `prompt_version`
- `provider`
- `model_id`/`model_path` (без утечки чувствительных данных при необходимости)

### Политика замены модели

При замене модели в первую очередь меняются:
- `provider`
- `model_path`
- runtime-параметры генерации

Остальные слои (контракты, prompt-логика, orchestrator-API) остаются без изменений.
