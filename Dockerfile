# Full version
FROM python:3.10-slim AS full

# Common dependencies
RUN apt-get update -qqy && \
    apt-get install -y --no-install-recommends \
        ssh \
        git \
        gcc \
        g++ \
        poppler-utils \
        libpoppler-dev \
        unzip \
        curl \
        cargo \
        tesseract-ocr \
        tesseract-ocr-jpn \
        libsm6 \
        libxext6 \
        libreoffice \
        ffmpeg \
        libmagic-dev

# Setup args
ARG TARGETPLATFORM
ARG TARGETARCH

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV TARGETARCH=${TARGETARCH}
ENV PDFJS_PREBUILT_DIR="/app/src/ui/kso_rag_ui/assets/prebuilt/pdfjs-dist"

# Create working directory
WORKDIR /app

# --- Layer 1: Copy only what pip needs (rarely changes) ---
# So when you change app code, only the final COPY is re-run, not pip install
COPY pyproject.toml ./
COPY src ./src

# Install pip packages (this layer is cached until pyproject or src deps change)
RUN --mount=type=ssh  \
    --mount=type=cache,target=/root/.cache/pip  \
    pip install -e "src/core[adv]" \
    && pip install -e "src/ui" \
    && pip install "pdfservices-sdk@git+https://github.com/niallcm/pdfservices-python-sdk.git@bump-and-unfreeze-requirements"

# Install graphrag for amd64
RUN --mount=type=ssh  \
    --mount=type=cache,target=/root/.cache/pip  \
    if [ "$TARGETARCH" = "amd64" ]; then pip install "graphrag<=0.3.6" future; fi

# Install torch and torchvision for unstructured
RUN --mount=type=ssh  \
    --mount=type=cache,target=/root/.cache/pip  \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional pip packages
RUN --mount=type=ssh  \
    --mount=type=cache,target=/root/.cache/pip  \
    pip install unstructured[all-docs] \
    && pip install "huggingface_hub<0.25.0"

# Install lightRAG
ENV USE_LIGHTRAG=true
RUN --mount=type=ssh  \
    --mount=type=cache,target=/root/.cache/pip  \
    pip install aioboto3 nano-vectordb ollama xxhash "lightrag-hku<=1.3.0" \
    && pip install "pydantic<=2.10.6"

RUN --mount=type=ssh  \
    --mount=type=cache,target=/root/.cache/pip  \
    pip install "docling<=2.5.2"

# Fix: ensure pyparsing>=3.0 is installed so httplib2 can use DelimitedList
RUN --mount=type=cache,target=/root/.cache/pip  \
    pip install --force-reinstall "pyparsing>=3.0,<4" "httplib2>=0.22.0"

# Download NLTK data from LlamaIndex
RUN python -c "from llama_index.core.readers.base import BaseReader"

# Clean up
RUN apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache

# --- Layer 2: Copy app code and assets (changes often) ---
# Only this layer rebuilds when you edit code; pip layer stays cached
COPY . /app

# Download pdfjs (runs after COPY so script and paths exist)
RUN chmod +x /app/scripts/download_pdfjs.sh \
    && bash /app/scripts/download_pdfjs.sh $PDFJS_PREBUILT_DIR

ENTRYPOINT ["sh", "/app/launch.sh"]

# Ollama-bundled version
FROM full AS ollama

# Install ollama
RUN --mount=type=ssh  \
    --mount=type=cache,target=/root/.cache/pip  \
    curl -fsSL https://ollama.com/install.sh | sh

# RUN nohup bash -c "ollama serve &" && sleep 4 && ollama pull qwen2.5:7b
RUN nohup bash -c "ollama serve &" && sleep 4 && ollama pull nomic-embed-text

ENTRYPOINT ["sh", "/app/launch.sh"]
