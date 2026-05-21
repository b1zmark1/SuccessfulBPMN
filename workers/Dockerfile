FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF" \
    FORCE_CMAKE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    nodejs \
    npm \
    chromium \
    default-jre-headless \
    graphviz \
    plantuml \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

WORKDIR /app

COPY requirements-worker.txt /tmp/requirements-worker.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r /tmp/requirements-worker.txt

# Render toolchain for text->image pipelines.
RUN npm install -g @mermaid-js/mermaid-cli

CMD ["python", "workers/job_worker.py", "--start-id", "0-0"]
