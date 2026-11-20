FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config.yaml ./
RUN pip install --no-cache-dir .
RUN mkdir -p /data
EXPOSE 4001
CMD ["uvicorn", "sidestep_audit.app:app", "--host", "0.0.0.0", "--port", "4001"]
