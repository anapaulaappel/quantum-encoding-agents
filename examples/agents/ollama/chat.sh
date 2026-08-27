#!/usr/bin/env bash
# Exemplo: Ollama OpenAI-compatible + tools locais via harness CLI
set -euo pipefail
export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
export AGENT_MODEL="${AGENT_MODEL:-qwen2.5-coder:14b}"
export AGENT_BACKEND="${AGENT_BACKEND:-ollama}"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
exec python3 "$ROOT/scripts/run_agent_harness.py" --persona "${1:-default}" "${2:-Qual encoding para classificação com 4 features contínuas?}"
