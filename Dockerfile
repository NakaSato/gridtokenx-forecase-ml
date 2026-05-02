FROM python:3.11-slim

WORKDIR /app

# System deps for LightGBM + scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies (cached layer)
COPY pyproject.toml requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Source code
COPY config.yaml evaluate.py main.py ./
COPY api/       ./api/
COPY models/    ./models/
COPY optimizer/ ./optimizer/
COPY data/      ./data/
COPY results/   ./results/

ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
