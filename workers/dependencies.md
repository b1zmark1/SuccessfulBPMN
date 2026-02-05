# Worker Dependencies (фиксировано по чату)

Этот файл фиксирует зависимости, обсужденные в чате, чтобы пересборка worker была воспроизводимой.

## Python (`workers/requirements-worker.txt`)
- asyncpg
- redis
- sqlalchemy[asyncio]
- pydantic
- pydantic-settings
- numpy
- opencv-python
- pillow
- pyyaml
- cairosvg
- cairocffi
- cssselect2
- torch
- openvino
- tabulate
- easyocr
- pytesseract
- llama-cpp-python

## System packages (`workers/Dockerfile`)
- build-essential
- cmake
- ninja-build
- pkg-config
- nodejs
- npm
- tesseract-ocr
- tesseract-ocr-rus
- libgl1
- libglib2.0-0
- libgomp1

## Node/tooling for BPMN render
- Используется `tools/bpmn_dataset/render_bpmn_svg.js` (Puppeteer + bpmn-js).
- Требуется установленный Chrome для Puppeteer.

Команда для ручной установки Chrome внутри уже запущенного container:

```bash
docker compose exec worker sh -lc "cd /app/tools/bpmn_dataset && npx puppeteer browsers install chrome --install-deps"
```

## Runtime files expected by worker
- OpenVINO model:
  - `results/yolox_tiny_bpmn_openvino.xml`
  - `results/yolox_tiny_bpmn_openvino.bin`
- YOLOX exp file:
  - `results/yolox_tiny_bpmn.py`
- Dataset classes:
  - `datasets/bpmn_full/classes.txt`
- Narrator GGUF model:
  - `narrator/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf`

## OCR engine
- Worker использует `tesseract` для `image_to_text`.

## GPU/CPU auto mode
- Worker автоматически пытается использовать GPU для этапов детекции, если CUDA доступна.
- При отсутствии GPU пайплайн автоматически переключается на CPU без падения.
- Для `detect_res` флаг `--gpu` включается только если доступен CUDA runtime и `DETECT_RES_GPU` не отключен.

## Warmup at container startup
- В `docker-compose.yml` для `worker` включен прогрев:
  - `WARMUP_ENABLED=1`
  - `WARMUP_SAMPLE_IMAGE=/app/preprocanddetect/diagram1.jpg`
  - `WARMUP_NARRATOR_MODE=table`
- Прогрев запускает `workers/warmup.py` перед `job_worker.py` и не блокирует старт worker при ошибке warmup.
- Для ускорения повторных стартов добавлены volume-кэши:
  - `worker_cache` -> `/root/.cache`
