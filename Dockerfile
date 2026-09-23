# Forecast API image (SPEC Phase C1). The production path only: no torch, image or LLM stack.
#
#   docker build -t nss-forecaster .
#   docker run -p 8080:8080 -v /path/to/registry:/registry -e NSS_REGISTRY_ROOT=/registry nss-forecaster
#
# No model is baked in: the service loads NSS_ARTIFACT_DIR, or the champion of NSS_REGISTRY_ROOT
# (a mounted path now, gs://... in C2). With neither, /healthz answers 503 with the reason.

# --- deps: resolve the locked `serving` group into a venv; cached until the lockfile changes ---
FROM python:3.13-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group serving --no-install-project

# --- runtime ---
FROM python:3.13-slim
# LightGBM's wheel links against OpenMP
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=deps /opt/venv /opt/venv
COPY src/ /app/src/
COPY configs/ /app/configs/
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "nss.prod.api:app", "--host", "0.0.0.0", "--port", "8080"]
