FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY config.yaml ./
COPY models/ ./models/
COPY optimizer/ ./optimizer/
COPY api/ ./api/
COPY data/ ./data/
COPY results/ ./results/

ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
