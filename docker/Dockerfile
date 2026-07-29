FROM python:3.11-slim

# Fail fast, no .pyc clutter, unbuffered logs so container output is live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_PORT=8000 \
    DATA_DIR=/app/memory/data

WORKDIR /app

# curl is used by the container healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so application edits don't invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user — the previous image ran everything as root, so a
# code-execution bug in a request handler owned the whole container.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p "$DATA_DIR" \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${APP_PORT}/health/live" || exit 1

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${APP_PORT}"]
