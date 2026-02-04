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
