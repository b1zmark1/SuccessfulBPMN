# Tasklist: точечные доработки по UI и worker

## 1) Проверка и выравнивание пайплайнов

- [X] Сравнить `workers/image_to_text_pipeline.py` с логикой `run_full_bpmn_pipeline.ps1` по шагам.
- [X] Зафиксировать, что Narrator для `image_to_text` работает всегда.
- [X] Поддержать режимы Narrator: `text` и `table` (UI -> `meta` -> worker).

## 2) Исправление `text_to_image` ошибки

- [X] Диагностировать, почему `Text-to-image pipeline did not return PNG output`.
- [X] Исправить worker, чтобы стабильно возвращал `result.image_url` при успешном рендере.
- [X] Улучшить сообщение об ошибке из worker для UI.

## 3) Правки текущего UI-поведения

- [x] Убрать отображение `Job ID` из UI.
- [x] Убрать слово "асинхронно" из описаний.
- [x] Добавить длинный пример промпта в `text_to_image` как информационную сноску (не как автозаполнение поля).

## 4) Image-to-text UX

- [x] Сделать красивую drag-and-drop зону вместо обычного file input.
- [x] Оставить fallback на клик/выбор файла.
- [x] Добавить переключатель режима Narrator: `text` / `table`.
- [x] Сохранить масштаб и качество изображения при загрузке (без ухудшения на frontend).

## 5) Редизайн (дерзкий, меньше градиентов)

- [x] Перейти на акцент `#F36523`.
- [x] Перестроить стартовый экран в split-layout на весь экран.
- [x] Зафиксировать desktop split строго `50/50`.
- [x] Добавить анимированное раскрытие выбранной половины.
- [x] Упростить фон и снизить количество градиентов.

## 6) Проверка целостности

- [ ] Прогнать unit-тесты UI/API/lifecycle.
- [ ] Ручной smoke-тест обоих сценариев с живым backend+worker.

  Обновить документацию в `frontend/README.md` и `frontend/ui/README.md`.

  7) Добавь логотип из frontend\logo.png
     8. непонятно работает ли redis и postgres
     9. Очередь для изображений чтобы была

     10.интегрирование draw.io(подробнее описать)

## 11) Image Queue (up to 5) + UI update

- [ ] Lock architecture: for `image_to_text`, each image is a separate backend job; UI batch limit is 5 files.
- [ ] Keep backend contract unchanged: use current `POST /jobs` and `GET /jobs/{job_id}` (no batch endpoint).
- [ ] Add client queue state model: `queued_local` -> `creating` -> backend status (`pending/queued/running/done/error`).
- [ ] Implement sequential enqueue policy (1 by 1) with retry for network errors and clear stop rules for validation errors.
- [ ] Update `ScenarioInputForm` for multi-file upload (`multiple`) + drag-and-drop with validation (type/size/max=5).
- [ ] Add queue UI list (1..5): thumbnail, filename, status badge, actions (`remove`, `retry`, `start all`).
- [ ] Update lifecycle hook/state to support multiple `job_id` polling and proper timer cleanup per item.
- [ ] Add result rendering per queue item (text result/error), including copy/download action for each result.
- [ ] Add UX guardrails: disable adding >5, show "queued N of 5", show aggregate progress.
- [ ] Tests: unit tests for queue reducer/hook + integration tests for 5-file flow with msw success/error/timeout.
- [ ] Docs: update `frontend/README.md` and `frontend/ui/README.md` with queue behavior and edge cases.
