# Integração agentica — opções flexíveis

O **núcleo quântico** é determinístico; a **camada agentica** escolhe *quais* tools chamar e *como* narrar.

Todas as opções abaixo usam a **mesma lista de tools** (`GET /v1/tools` ou `tool_registry.dispatch_tool`).

## Comparativo rápido

| Opção | Onde roda o LLM | Onde rodam as tools | Personas | Canais (Telegram, etc.) |
|-------|------------------|---------------------|----------|-------------------------|
| **A. Harness CLI** | Local (Ollama/vLLM) | In-process Python | default / expert / mentor | Terminal |
| **B. POST /v1/agent/chat** | Servidor (env AGENT_*) | In-process no pod API | default / expert / mentor | Qualquer HTTP client |
| **C. OpenClaw** | Gateway OpenClaw | HTTP → `/v1/tools/dispatch` | SOUL.md por agente | WhatsApp, Slack, web, … |
| **D. Llama Stack** | Llama Stack server | In-process ou HTTP | Via agent config LS | Apps LS |
| **E. HTTP dispatch puro** | Seu agente (LangGraph, CrewAI, Cursor SDK) | POST `/v1/tools/dispatch` | Você define | Qualquer |
| **F. REST one-shot** | Nenhum (templates) | `/v1/recommend/explain` | N/A | curl, Kubeflow |
| **G. Web UI `/chat`** | Modo Agente: servidor (`AGENT_*`) | `/v1/agent/chat` ou direto `/v1/compare/csv` | default / expert / mentor | Browser |

## Tools disponíveis (9)

`analyze_data`, `recommend_embedding_strategy`, `generate_qiskit_circuit`, `simulate_circuit`,
`compare_embeddings_report`, `compare_csv_embeddings`, `compute_quantum_kernel`,
`explain_tradeoffs`, `scenarios_guide`

## A — Harness CLI (desenvolvimento local)

```bash
# Só tools (sem LLM)
python scripts/run_agent_harness.py --dispatch recommend_embedding_strategy \
  '{"dataset_or_description":"8 continuous features, classification"}'

# Com Ollama (default)
export OLLAMA_HOST=http://localhost:11434
export AGENT_MODEL=qwen2.5-coder:14b
python scripts/run_agent_harness.py --persona expert "Qual encoding para QSVM com 6 features?"

# Llama Stack
export AGENT_BACKEND=llama_stack
export LLAMA_STACK_CLIENT_BASE_URL=http://localhost:8321
python scripts/run_agent_harness.py "Compare todos os encodings para [0.1,0.2,0.3,0.4]"
```

Ver também: `ollama/chat.sh`

## B — API server-side agent

```bash
curl -s http://localhost:8080/v1/agent/chat -H 'Content-Type: application/json' -d '{
  "message": "Tenho labels no CSV, quero ranking por KTA",
  "persona": "mentor",
  "max_tool_rounds": 6
}'
```

Configure no deployment: `AGENT_BACKEND`, `AGENT_MODEL`, `OPENAI_API_BASE`, `OLLAMA_HOST`.

## C — OpenClaw (produção multi-canal)

1. Suba a API: `docker compose up` ou OpenShift `deploy/openshift/`
2. Copie `openclaw/SKILL-quantum-encoding.md` para a skill do agente
3. Use `openclaw/agents/circuit/SOUL.md` ou `quanta/SOUL.md` como persona
4. Skill chama `POST http://llama-qiskit-agents:8080/v1/tools/dispatch`

Documentação gerada pela API: `GET /v1/tools/openclaw-skill.md?api_base=http://...`

Alternativa: skill nativa OpenClaw com `$http` ou exec — ver `openclaw/config-snippet.json`.

## D — Llama Stack

Registre as tools de `GET /v1/tools?format=openai` no agente Llama Stack e aponte handlers para
`dispatch_tool` (Python in-process) ou proxy HTTP para `/v1/tools/dispatch`.

Cliente mínimo: `src/llama_qiskit_agents/agents/client.py` + `harness.py`.

## E — HTTP dispatch (qualquer framework)

```python
import requests
r = requests.post("http://localhost:8080/v1/tools/dispatch", json={
    "name": "compute_quantum_kernel",
    "arguments": {
        "data": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "encoding_name": "iqp",
        "labels": [0, 0, 1],
    },
})
print(r.json()["result"])
```

Exemplo completo: `http/client_example.py`

## F — Sem agente (automação)

Kubeflow, CI e usuários avançados continuam com `/v1/recommend/explain`, `/v1/compare/csv`, `/v1/kernel`.

## G — Web UI (`/chat`)

1. Suba a API: `docker compose up --build`
2. Abra `http://localhost:8080/chat`
3. Modo **Agente** (padrão): multi-turn + personas + CSV → `/v1/agent/chat`
4. Modo **Direto**: relatório determinístico → `/v1/compare/csv` (sem LLM)

Requer Ollama acessível quando `AGENT_BACKEND=ollama` (ver `docker-compose.yml`).

## Variáveis de ambiente (harness / /v1/agent/chat / UI)

| Variável | Default | Descrição |
|----------|---------|-----------|
| `AGENT_BACKEND` | `ollama` | `ollama` \| `openai_compatible` \| `llama_stack` \| `remote` |
| `AGENT_MODEL` | `qwen2.5-coder:14b` | Modelo no backend |
| `OPENAI_API_BASE` | `{OLLAMA_HOST}/v1` | Base OpenAI-compatible |
| `OLLAMA_HOST` | `http://localhost:11434` | Host Ollama |
| `OPENAI_API_KEY` | — | Se o backend exigir |
