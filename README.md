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
| `GET` | `/healthz` | Health check |
| `POST` | `/v1/recommend/explain` | **Principal** — recomendação + explicação + código |
| `POST` | `/v1/recommend` | Recomendação estruturada (JSON) |
| `POST` | `/v1/compare` | Ranking comparativo dos 7 encodings |
| `POST` | `/v1/compare/csv` | Upload CSV multipart |
| `POST` | `/v1/circuit` | Diagrama ASCII do circuito |
| `POST` | `/v1/simulate` | Simula e retorna histograma de medições |
| `GET` | `/v1/tradeoffs` | Trade-offs de todos os encodings |
| `GET` | `/docs` | Swagger interativo |

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
```

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

## Arquitetura

```
Usuário (chat ou curl)
        │
   OpenClaw (Node.js)          ← personalidade via SOUL.md + AGENTS.md
        │
   LLM (Ollama local           ← qwen2.5-coder:14b / mistral-small3.2:24b
        ou vLLM no OpenShift)
        │
        └──► POST /v1/recommend/explain
                    │
             quantum-encoding-agents (FastAPI)
                    ├── infer_data_profile()
                    ├── recommend_encoding()
                    ├── detect_language()          ← PT-BR / EN automático
                    ├── build_natural_explanation() ← cita DataProfile + métricas reais
                    ├── generate_qiskit_code()      ← código copiável por encoding
                    └── simulate_encoding_circuit() ← AerSimulator (CPU)
```

---

## Estrutura do repositório

```
quantum-encoding-agents/
├── src/llama_qiskit_agents/
│   ├── quantum/
│   │   ├── encodings.py        # 5 circuit builders + EncodingType enum
│   │   ├── data_analysis.py    # DataProfile + recommend_encoding
│   │   ├── problem_context.py  # MLTask, ProblemContext, refine_recommendation
│   │   ├── explanation.py      # detect_language + frases narrativas + código Qiskit
│   │   ├── encoding_ranking.py # ranking formatado para relatório
│   │   └── simulate.py         # AerSimulator orchestration
│   ├── api/
│   │   ├── app.py              # FastAPI endpoints
│   │   └── schemas.py          # Pydantic models (inclui ExplainRequest/Response)
│   └── agents/
│       └── encoding_agent.py   # 6 tool functions para Llama Stack
├── openclaw-openshift/
│   ├── local/                  # Stack local (Docker Compose + Ollama)
│   │   ├── docker-compose.yml
│   │   ├── setup.sh
│   │   └── config/agents/      # SOUL.md de cada agente
│   ├── 04-configmap-openclaw.yaml
│   ├── 05-configmap-agent-*.yaml
│   ├── 10-deployment.yaml
│   └── GUIA-INSTALACAO.md
├── deploy/openshift/           # Manifests do microserviço (API)
├── scripts/                    # CLI runners
├── Dockerfile
└── DEMO.md                     # Guia de demonstração para conferências
```

---

## Variáveis de ambiente

| Variável | Padrão | Uso |
|---|---|---|
| `PORT` | `8080` | Porta da API |
| `CORS_ORIGINS` | `*` | Origins CORS |
| `LLAMA_STACK_CLIENT_API_KEY` | — | Opcional: auth Llama Stack |
| `LLAMA_STACK_CLIENT_BASE_URL` | — | Opcional: URL servidor Llama Stack |

---

## Licença

MIT
