FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=5010

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/tmp_audio \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5010

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5010/api/v1/health', timeout=3).read()"

CMD ["gunicorn", "--bind", "0.0.0.0:5010", "--workers", "1", "--threads", "8", "--timeout", "300", "app:create_app()"]
