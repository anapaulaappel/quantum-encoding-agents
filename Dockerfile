# Imagem para OpenShift / testes locais (API + UI em /chat)
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --upgrade pip && pip install ".[api]"

# OpenShift: UID arbitrário no grupo root (0) com permissão de leitura na app
RUN chgrp -R 0 /app && chmod -R g=u /app
USER 1001

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn llama_qiskit_agents.api.app:app --host 0.0.0.0 --port ${PORT}"]
