# OrbitBot: FastAPI + React UI in one image, for a Hugging Face Docker Space (or any container host).
#
#   docker build -t orbitbot .
#   docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY=... orbitbot      # then open http://localhost:7860
#
# The API key is passed at run time (a Space secret), never baked into the image.

# ---------- 1. build the React UI ----------
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# shown as the "Source code on GitHub" button in the About panel
ARG VITE_GITHUB_URL=https://github.com/kaushik-hadiya/OrbitBot
ENV VITE_GITHUB_URL=$VITE_GITHUB_URL
RUN npm run build

# ---------- 2. the Python app ----------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# Spaces run the container as uid 1000; Chroma writes to its directory even when only reading
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# dependencies first: this layer is reused until pyproject.toml or uv.lock change
COPY --chown=user pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user main.py api.py agent_tools.py decisions.py ./
COPY --chown=user documents/ documents/
COPY --chown=user data/ data/

# Download the embedding + reranker models (~210 MB) and build the Chroma index now, so a cold start
# only loads them. main.py requires the key to exist at import; nothing is called with this placeholder.
RUN OPENROUTER_API_KEY=build-only-placeholder python -c "import main"

COPY --from=ui --chown=user /ui/dist frontend/dist

# demo defaults; each can be overridden in the Space's Variables settings
ENV DECIDER=jev \
    GRADER=jev \
    USEFUL=jev \
    SUPPORT=jev \
    PROXY_HOPS=1 \
    LANGSMITH_TRACING=false
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
