##########  build stage  ##########
FROM python:3.11-slim AS build
WORKDIR /app

# 1. lock-based install for reproducible deps
COPY app/pyproject.toml app/poetry.lock ./
RUN pip install --no-cache-dir poetry && \
    poetry install --only main --no-interaction --no-ansi

##########  runtime stage ##########
FROM python:3.11-slim
ENV PATH="/root/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=build /root/.local /root/.local   # copy installed deps
COPY app/ ./app                               # copy source code

CMD ["uvicorn", "app.main:app",
     "--host", "0.0.0.0", "--port", "8000"]
