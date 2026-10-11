# Failsafe — deterministic core + CLI + evaluation harness.
# The image runs everything locally without external services.
FROM python:3.11-slim

# Never bake secrets into the image; pass them at runtime.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FAILSAFE_DB_URL=sqlite:////data/failsafe.db \
    FAILSAFE_LLM_ENABLED=false

WORKDIR /app

# System deps (none required beyond pip for the core)
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY examples ./examples
COPY evals ./evals
COPY ui ./app-ui
COPY scripts ./scripts

RUN pip install --upgrade pip && \
    pip install ".[dev]" && \
    pip install streamlit langgraph langchain-core

RUN useradd -m failsafeuser && mkdir -p /data && chown -R failsafeuser /data
USER failsafeuser

VOLUME ["/data"]

# Default: show CLI help. Override for other commands, e.g.
#   docker run failsafe failsafe evaluate --agent good
ENTRYPOINT ["failsafe"]
CMD ["--help"]
