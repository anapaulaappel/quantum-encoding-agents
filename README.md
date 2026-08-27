# Quantum Encoding Agents

Agentes de IA especializados em **encoding de dados para Quantum Machine Learning**,
com personalidades distintas por audiência. Detectam idioma automaticamente (PT-BR / EN),
recomendam a estratégia de encoding ideal, justificam com métricas reais do circuito
e entregam código Qiskit completo e copiável.

---

## O que faz

Dado um dataset (numérico, CSV ou descrição em linguagem natural) e contexto QML
opcional (tarefa, algoritmo, problema, hardware alvo), o sistema:

1. Analisa o perfil estrutural do dado (dimensão, tipo, distribuição)
2. Recomenda o melhor entre **7 encodings quânticos**
3. Ajusta a recomendação para o hardware alvo (gate error rate, max depth, conectividade)
4. Justifica a escolha citando métricas concretas: qubits, profundidade de circuito
5. Gera código Python/Qiskit completo e copiável
6. Simula via `AerSimulator` (CPU, sem hardware real)
7. Responde no idioma do input (PT-BR ou EN)

---

## Os 7 encodings suportados

| Encoding | Qubits | Profundidade | Ideal para |
|---|---|---|---|
| `amplitude` | ⌈log₂(n)⌉ | profunda | vetor grande, mínimo de qubits |
| `angle` | n_features | 1 | dados contínuos, ≤4 features, protótipo |
| `dense_angle` | ⌈n/2⌉ | 2 | 5–12 features, Ry·Rz por qubit (survey Sammartino 2026) |
| `iqp` | n_features | ~3n | 8–16 features, kernel quântico (Havlíček 2019) |
| `basis` | n_bits | mínima | dados binários ou categóricos |
| `data_reuploading` | n_features | média | VQC, QNN, alta expressividade |
| `custom_feature_map` | n_features | profunda | QSVM, kernel quântico arbitrário |

### Hardware-aware: limiar p* ≈ 10⁻³

Passe `hardware_profile` no request para recomendação NISQ-realista:

```json
{
  "data": [0.31, 0.72, 0.55, 0.18, 0.9],
  "task": "kernel",
  "hardware_profile": {
    "gate_error_rate": 5e-3,
    "connectivity": "heavy-hex",
    "max_depth_budget": 20,
    "backend_name": "ibm_eagle"
  }
}
```

Acima de `gate_error_rate = 1e-3` (limiar p* de Sammartino arXiv:2606.05387),
encodings profundos são automaticamente substituídos por alternativas mais rasas.

---

## Os agentes

Três agentes especializados em encoding, com personalidades distintas:

| Agente | Nome | Para quem |
|---|---|---|
| `quantum_encoding` | QiskitAgent | Uso geral — recomenda + código sem pressuposto de audiência |
| `qiskit_expert` | **Circuit** | Quem já conhece QML — resposta direta, métricas concretas, sem explicar o óbvio |
| `qiskit_mentor` | **Quanta** | Quem está aprendendo — constrói intuição física antes do código |

Mais dois agentes de suporte:

| Agente | Nome | Para quem |
|---|---|---|
| `meu_agente` | Meu Agente | Assistente geral em português |
| `quantum_news` | QubitinhoBot | Monitor de notícias de computação quântica |

### Arquitetura: microserviço + tools + orquestração flexível

| Camada | O que é | Onde |
|---|---|---|
| **Core quântico** | Preprocess, 7 encodings, KTA, otimização de mapeamento de features, simulação | `src/llama_qiskit_agents/quantum/` |
| **Tools** | 9 funções + `dispatch_tool` (fonte única) | `agents/tool_registry.py` |
| **REST one-shot** | Pipelines fixos (Kubeflow, curl) | `/v1/recommend/explain`, `/v1/compare`, … |
| **Agent harness** | LLM + function calling → tools | `scripts/run_agent_harness.py`, `/v1/agent/chat` |
| **OpenClaw** (opcional) | Personas + canais | `examples/agents/openclaw/` |

OpenClaw **não é obrigatório**. Outras opções: Ollama local, Llama Stack, LangGraph/CrewAI via
`POST /v1/tools/dispatch`. Guia completo: [`examples/agents/README.md`](examples/agents/README.md).

```bash
# Listar tools (schema OpenAI)
curl -s http://localhost:8080/v1/tools | python3 -m json.tool

# Executar tool (qualquer agente externo)
curl -s -X POST http://localhost:8080/v1/tools/dispatch \
  -H 'Content-Type: application/json' \
  -d '{"name":"recommend_embedding_strategy","arguments":{"dataset_or_description":"6 continuous features"}}'

# Harness CLI + Ollama
python3 scripts/run_agent_harness.py --persona expert "Qual encoding para QSVM?"
```

---

## Início rápido — local (Docker Compose + Ollama)

### Pré-requisitos

- Docker Desktop (Mac/Windows) ou Docker Engine (Linux)
- 16 GB RAM para `qwen2.5-coder:14b` (recomendado) | 8 GB para `:7b` | 24 GB para Mistral 24B

### Subir a stack

```bash
git clone https://github.com/anapaulaappel/quantum-encoding-agents
cd quantum-encoding-agents

# Setup automático (baixa o modelo + sobe tudo)
cd openclaw-openshift/local
./setup.sh                          # usa qwen2.5-coder:14b por padrão
# ./setup.sh qwen2.5-coder:7b       # versão leve
# ./setup.sh mistral-small3.2:24b   # versão mais capaz
```

O script:
- Gera o token de acesso automaticamente
- Faz `docker compose up --build`
- Baixa o modelo via Ollama (uma vez)
- Exibe a URL de acesso

### Acessar

```
http://localhost:18789/?token=<TOKEN-gerado-pelo-setup>
```

---

## API REST (sem OpenClaw)

A API pode ser usada diretamente, sem o chat:

```bash
# Subir só a API
docker compose up -d

# Endpoint principal: recomendação + explicação + código Qiskit
curl -s -X POST http://localhost:8080/v1/recommend/explain \
  -H "Content-Type: application/json" \
  -d '{
    "data": [0.31, 0.72, 0.55, 0.18, 0.9],
    "description": "features contínuas para QSVM",
    "algorithm": "QSVM",
    "include_bloch": true
  }' | python3 -m json.tool
```

Resposta inclui:
- `recommended_encoding` — encoding escolhido
- `explanation` — justificativa em linguagem natural no idioma detectado
- `qiskit_code` — código Python completo e copiável
- `circuit_depth` / `circuit_qubits` — métricas reais do circuito simulado
- `lang` — idioma detectado (`pt` ou `en`)
- `hardware_constraints_applied` — `true` se `hardware_profile` influenciou a recomendação
- `bloch_sphere_b64` — PNG base64 da esfera de Bloch por qubit (quando `include_bloch=true`)
- `bloch_caption` — legenda explicativa da esfera de Bloch no idioma detectado

### Outros endpoints

| Método | Path | Descrição |
|---|---|---|
| `GET` | `/health` `/healthz` | Health check |
| `POST` | `/v1/recommend/explain` | **Principal** — recomendação + explicação + código Qiskit + Bloch sphere |
| `POST` | `/v1/recommend` | Recomendação estruturada (JSON) |
| `POST` | `/v1/analyze` | Perfil do dado (DataProfile) |
| `POST` | `/v1/compare` | Ranking comparativo dos 7 encodings |
| `POST` | `/v1/compare/csv` | Upload CSV multipart (+ `optimize_features=true` com labels: ordem/seleção/peso) |
| `POST` | `/v1/kernel` | Matriz de kernel K_{ij} + KTA + heatmap |
| `GET` | `/v1/tools` | Schemas de tools (OpenAI / OpenClaw) |
| `POST` | `/v1/tools/dispatch` | Executa uma tool (integração agentes) |
| `GET` | `/v1/tools/openclaw-skill.md` | Skill Markdown para OpenClaw |
| `POST` | `/v1/agent/chat` | Turno agentico (LLM + tools, env AGENT_*) |
| `POST` | `/v1/circuit` | Diagrama ASCII do circuito |
| `POST` | `/v1/simulate` | Histograma de medições |
| `GET` | `/v1/tradeoffs` | Trade-offs de todos os encodings |
| `GET` | `/v1/scenarios-guide` | Guia de cenários QML |
| `GET` | `/v1/analyze/text?q=` | Análise rápida por query string |
| `GET` | `/chat` | **UI web** — chat agentico (`/v1/agent/chat`) ou compare CSV direto |
| `GET` | `/docs` | Swagger interativo |

### Exemplo: matriz de kernel com labels de classe

```bash
curl -s -X POST http://localhost:8080/v1/kernel \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[0.1,0.2],[0.15,0.25],[2.5,2.8],[2.6,2.9]],
    "encoding_name": "angle",
    "labels": [0, 0, 1, 1],
    "lang": "pt"
  }' | python3 -m json.tool
```

Resposta inclui `kernel_matrix` (N×N), `stats` (com KTA), `heatmap_b64` (PNG base64)
e `caption` — tudo que precisa para avaliar se o encoding separa as classes antes de
treinar qualquer classificador quântico.

---

## Interface web (`/chat`)

Com a API rodando (`docker compose up` ou `python scripts/run_api.py`):

```
http://localhost:8080/chat
```

| Modo | Endpoint | Requisitos |
|------|----------|------------|
| **Agente** (padrão) | `POST /v1/agent/chat` | Ollama ou LLM configurado (`AGENT_*`) |
| **Direto** | `POST /v1/compare/csv` | Só CSV + descrição (sem LLM) |

A UI suporta **personas** (Default, Circuit, Quanta), **histórico multi-turn**, anexo CSV
(com ranking KTA via tool `compare_csv_embeddings` no modo Agente) e coluna de label opcional.

```bash
# Docker + Ollama no host (macOS/Linux)
docker compose up --build
# Acesse http://localhost:8080/chat

# Ou API local + Ollama nativo
export OLLAMA_HOST=http://localhost:11434
export AGENT_MODEL=qwen2.5-coder:14b
pip install -e ".[api]"
python scripts/run_api.py
```

---

## CLI (sem Docker, sem OpenClaw)

```bash
pip install -e ".[api]"

# Descrição em texto
python3 scripts/run_encoding_agent.py "dados contínuos com 5 features"

# CSV
python3 scripts/run_encoding_agent.py --csv dados.csv

# Dados numéricos com contexto QML
python3 scripts/run_encoding_agent.py --data 0.1 0.2 0.3 --task kernel --algorithm QSVM

# Apenas trade-offs
python3 scripts/run_encoding_agent.py --tradeoffs-only

# Harness agentico (Ollama + tools locais)
python3 scripts/run_agent_harness.py "Compare encodings para dados contínuos com 5 features"
```

---

## Kubeflow Pipeline (RHOAI)

Pipeline de 5 etapas para Red Hat OpenShift AI com Data Science Pipelines:

```
CSV → analyze → recommend → compare → kernel(KTA) → MLflow log
```

```bash
cd deploy/rhoai

# Compilar o pipeline YAML
python pipeline.py

# Submeter ao cluster RHOAI
DSP_HOST=$(oc get route -n rhoai-dsp data-science-pipelines-api -o jsonpath='{.spec.host}')
python submit.py \
  --dsp-endpoint "https://$DSP_HOST" \
  --csv dados.csv \
  --labels "0,0,1,1" \
  --task kernel \
  --alg QSVM \
  --hw-error 5e-3 \
  --connectivity heavy-hex
```

Ver guia completo em [`deploy/rhoai/RHOAI-PIPELINE.md`](deploy/rhoai/RHOAI-PIPELINE.md).

---

## Deploy no OpenShift

Ver guia completo em [`openclaw-openshift/GUIA-INSTALACAO.md`](openclaw-openshift/GUIA-INSTALACAO.md).

Resumo:

```bash
# Build da imagem no cluster
oc new-build --name=quantum-encoding-agents --binary -n openclaw
oc start-build quantum-encoding-agents --from-dir=. --follow -n openclaw

# Aplicar manifests do microserviço
oc apply -f deploy/openshift/

# Aplicar manifests do OpenClaw (com os agentes)
cd openclaw-openshift
oc apply -f 00-namespace.yaml
# ... (ver GUIA-INSTALACAO.md para ordem completa)
```

---

## Benchmarks (Iris, Breast Cancer, Wine)

Suite reprodutível para **KTA**, ranking de encodings e **otimização ordem/seleção/peso** ([2512.02422](https://arxiv.org/abs/2512.02422)).

```bash
pip install -e ".[benchmark]"
python scripts/run_benchmarks.py --quick          # Iris + sintético (~1 min)
python scripts/run_benchmarks.py                  # suite completa
python scripts/run_benchmarks.py --dataset iris_binary --json out.json
pytest tests/test_benchmarks.py -q                # smoke CI
```

| Dataset | Origem | Uso |
|---------|--------|-----|
| `iris_binary` | UCI Iris (2 classes) | Rápido, CI |
| `breast_cancer_top6` | Wisconsin BC, 6 features | Público, médio |
| `wine_binary` | UCI Wine (2 classes) | 13 features |
| `synthetic_order` | Gerado | Sensível a permutação de colunas |

Detalhes: [`benchmarks/README.md`](benchmarks/README.md).  
**Roteiro para apresentar oralmente:** [`benchmarks/PRESENTATION.md`](benchmarks/PRESENTATION.md).  
**Hardware IBM (~10 min/mês):** [`benchmarks/HARDWARE.md`](benchmarks/HARDWARE.md) — `python scripts/run_hardware_benchmarks.py --preset week1_iris`.

---

## Arquitetura

```
Usuário (browser /chat, OpenClaw, curl, Kubeflow)
        │
   ┌────┴──── OpenClaw (opcional) ──── LLM + SOUL.md (Circuit / Quanta)
   │              │
   │              └──► POST /v1/tools/dispatch  ou  /v1/agent/chat
   │
   └──► GET /chat  ──► POST /v1/agent/chat  (UI web, personas)
        │
   LLM (Ollama / vLLM / Llama Stack)     ← AGENT_* env
        │
             llama-qiskit-agents (FastAPI)
                    ├── GET /v1/tools + POST /v1/tools/dispatch  (9 tools)
                    ├── POST /v1/agent/chat                        (harness server-side)
                    ├── POST /v1/recommend/explain                 (one-shot, sem LLM)
                    ├── infer_data_profile() → recommend_encoding()
                    ├── simulate_encoding_circuit() → compute_kernel() (K_{ij}, KTA)
                    └── dispatch_tool()  ← fonte única quantum/*
```

---

## Estrutura do repositório

```
quantum-encoding-agents/
├── src/llama_qiskit_agents/
│   ├── quantum/
│   │   ├── encodings.py         # 7 circuit builders + EncodingType enum
│   │   ├── data_analysis.py     # DataProfile + recommend_encoding
│   │   ├── problem_context.py   # MLTask, ProblemContext, refine_recommendation
│   │   ├── hardware_profile.py  # HardwareProfile, limiar p*=1e-3 (NISQ-aware)
│   │   ├── explanation.py       # detect_language + narrativas PT/EN + código Qiskit
│   │   ├── visualization.py     # Bloch sphere via StatevectorSimulator + matplotlib
│   │   ├── kernel.py            # FidelityStatevectorKernel, KTA, heatmap
│   │   ├── encoding_ranking.py  # ranking formatado para relatório
│   │   └── simulate.py          # AerSimulator orchestration
│   ├── api/
│   │   ├── app.py               # FastAPI — REST + /v1/tools + /v1/agent/chat
│   │   ├── schemas.py           # Pydantic models
│   │   └── static/chat.html     # UI web agentica
│   └── agents/
│       ├── tool_registry.py     # 9 tools + dispatch (fonte única)
│       ├── harness.py           # ReAct loop (Ollama / Llama Stack)
│       └── encoding_agent.py    # Re-exports retrocompatíveis
├── examples/agents/             # OpenClaw, Ollama, HTTP — guia integração
├── deploy/openshift/            # Manifests do microserviço (API)
├── scripts/
│   ├── run_agent_harness.py     # CLI agentico
│   └── run_encoding_agent.py    # CLI pipeline quântico
├── Dockerfile
└── docker-compose.yml           # API + env AGENT_* para /chat
```

---

## Variáveis de ambiente

| Variável | Padrão | Uso |
|---|---|---|
| `PORT` | `8080` | Porta da API |
| `CORS_ORIGINS` | `*` | Origins CORS |
| `AGENT_BACKEND` | `ollama` | Backend do `/v1/agent/chat` e UI `/chat` |
| `AGENT_MODEL` | `qwen2.5-coder:14b` | Modelo LLM |
| `OLLAMA_HOST` | `http://localhost:11434` | Host Ollama |
| `OPENAI_API_BASE` | `{OLLAMA_HOST}/v1` | API OpenAI-compatible |
| `LLAMA_STACK_CLIENT_API_KEY` | — | Opcional: auth Llama Stack |
| `LLAMA_STACK_CLIENT_BASE_URL` | — | Opcional: URL servidor Llama Stack |

---

## Licença

MIT
