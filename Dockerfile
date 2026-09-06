FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG INSTALL_HEAVY=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-heavy.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_HEAVY" = "true" ]; then pip install --no-cache-dir -r requirements-heavy.txt; fi

COPY app/ ./app
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "server:app", "--app-dir", "/app/app", "--host", "0.0.0.0", "--port", "8000"]
