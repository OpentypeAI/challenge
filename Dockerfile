# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
# Two targets from one file:
#   server  the Cortex challenge container (contract v1): non-root 65532, read-only root.
#   worker  the B300 duel worker: the pinned vLLM nightly (PR 57250 structured reads) plus
#           the pinned structured_server.py and this package.
# No secret enters any layer: tokens are mounted at run time.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.17@sha256:03bdc89bb9798628846e60c3a9ad19006c8c3c724ccd2985a33145c039a0577b
ARG VLLM_IMAGE=vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423@sha256:ed3c505d2cf4b62b0ca8d0b87591bcb6fbff2a3abcc6d25732343e19b4d74f09

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
# uv compiles bytecode with one interpreter per core; lift the 1024 soft fd limit for it.
RUN ulimit -n "$(ulimit -Hn)" \
 && uv venv /opt/venv --python /usr/local/bin/python3.12 \
 && uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/requirements.txt \
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache -r /tmp/requirements.txt
COPY src ./src
RUN ulimit -n "$(ulimit -Hn)" \
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache --no-deps . \
 && uv build --wheel --out-dir /wheels

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS server
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/OpentypeAI/challenge" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="opentype challenge" \
      io.cortex.challenge.slug="opentype" \
      io.cortex.challenge.contract="1"
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/* \
 && install -d -o 65532 -g 65532 -m 0750 /data
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHALLENGE_SLUG=opentype \
    CHALLENGE_STATE_DIR=/data \
    CHALLENGE_INTERNAL_TOKEN_FILE=/run/secrets/internal.token \
    CHALLENGE_ADMIN_TOKEN_FILE=/run/secrets/admin.token \
    CHALLENGE_WORKER_TOKEN_FILE=/run/secrets/worker.token
USER 65532:65532
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8000
# No HEALTHCHECK: the Cortex supervisor probes /version (liveness) and /health (readiness).
ENTRYPOINT ["opentype-challenge", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
FROM ${VLLM_IMAGE} AS worker
ARG VERSION=dev
ARG REVISION=unknown
ARG STRUCTURED_SERVER_URL=https://raw.githubusercontent.com/vllm-project/vllm/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/examples/features/structured_diffusion/structured_server.py
ARG STRUCTURED_SERVER_SHA256=7cd9aa0081090c064eaac28db0f54f812749eeb3ae787d7f7653d2e35d8a938f
ARG VLLM_IMAGE
LABEL org.opencontainers.image.source="https://github.com/OpentypeAI/challenge" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="opentype duel worker" \
      io.cortex.challenge.slug="opentype" \
      io.cortex.challenge.contract="1"
ADD --checksum=sha256:${STRUCTURED_SERVER_SHA256} ${STRUCTURED_SERVER_URL} /opt/opentype/structured_server.py
COPY --from=build /wheels /tmp/wheels
# The vLLM image already ships fastapi, httpx, pydantic, uvicorn and huggingface_hub; install
# only what it lacks so vLLM's own pins never move.
RUN echo "${STRUCTURED_SERVER_SHA256}  /opt/opentype/structured_server.py" | sha256sum -c - \
 && chmod 0644 /opt/opentype/structured_server.py \
 && uv pip install --system --no-cache "py-sr25519-bindings>=0.2.3,<0.3" \
 && uv pip install --system --no-cache --no-deps /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels \
 && printf '{"vllm_image": "%s"}\n' "${VLLM_IMAGE}" > /opt/opentype/build.json \
 && chmod 0444 /opt/opentype/build.json \
 && python3 -c "import fastapi, httpx, huggingface_hub, pydantic, sr25519, uvicorn, opentype_challenge"
ENV OPENTYPE_STRUCTURED_SERVER=/opt/opentype/structured_server.py \
    OPENTYPE_WORKER_IMAGE="ghcr.io/opentypeai/challenge-worker:sha-${REVISION}" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1
ENTRYPOINT ["opentype-challenge", "worker"]
