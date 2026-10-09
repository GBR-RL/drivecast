# The scoring service with the champion model baked in.
# Build context: the repository, with the exported model in ./model (drivecast export-champion).
FROM python:3.11-slim
LABEL org.opencontainers.image.source="https://github.com/GBR-RL/drivecast" \
      org.opencontainers.image.description="drivecast: 30-day hard-drive failure risk from SMART data" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
COPY model ./model

USER app
ENV DRIVECAST_MODEL_DIR=/app/model PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "drivecast.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
