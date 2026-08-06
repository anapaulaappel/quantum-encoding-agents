# Project Context: llama-qiskit-agents

> Documento de referência rápida para retomar o projeto sem reler o código inteiro.

---

## Propósito

Pacote Python que conecta **Llama Stack** (LLMs com function-calling) ao **Qiskit + Qiskit-Aer** para recomendar, gerar e simular estratégias de **encoding de dados quânticos** (Quantum Machine Learning). Dado um dataset (CSV, array numérico, ou descrição textual) + contexto QML opcional, o agente:

1. Analisa o perfil do dado
2. Recomenda a melhor estratégia de encoding
3. Gera o circuito Qiskit correspondente
4. Simula via `AerSimulator` (CPU, sem hardware real)
5. Produz relatório comparativo das 5 estratégias

Exposto via **FastAPI** (`/v1/…`) e **UI web** (`/chat`).

---

## Layout

```
llama-qiskit-agents/
├── src/llama_qiskit_agents/
│   ├── agents/
│   │   ├── client.py          # LlamaStackClient factory + chat_completion helper
│   │   └── encoding_agent.py  # 6 tool functions + tool defs JSON + dispatch_tool
│   ├── api/
│   │   ├── app.py             # FastAPI: todos os endpoints
│   │   ├── schemas.py         # Pydantic models request/response
│   │   └── static/chat.html   # UI web (dark, form → /v1/compare/csv)
│   └── quantum/
│       ├── encodings.py       # EncodingType enum + 5 circuit builders
│       ├── data_analysis.py   # CSV load, DataProfile, recommend_encoding
│       ├── problem_context.py # MLTask, ProblemContext, refine_recommendation
│       ├── encoding_ranking.py# Formatação do ranking + nota de medições
│       ├── simulate.py        # AerSimulator orchestration + format_comparison_report
│       └── circuits.py        # Bell + simple circuit (demo apenas)
├── scripts/
│   ├── run_encoding_agent.py  # CLI principal (sem Llama Stack)
│   ├── run_api.py             # Inicia uvicorn
│   └── run_quantum_example.py # Demo Qiskit puro
├── demo_script.py             # Demo Llama Stack RAG (independente do quantum)
├── config.yaml                # Referência provider Ollama (não carregado pelo app)
├── deploy/openshift/          # deployment.yaml, service.yaml, route.yaml
├── Dockerfile                 # python:3.12-slim, rootless (UID 1001), OpenShift-ready
└── docker-compose.yml         # porta 8080, CORS_ORIGINS=*
```

---

## Estruturas de Dados Chave

| Struct | Arquivo | Tipo | Campos principais |
|--------|---------|------|-------------------|
| `EncodingType` | `encodings.py:11` | `str+Enum` | `amplitude`, `angle`, `basis`, `data_reuploading`, `custom_feature_map` |
| `DataProfile` | `data_analysis.py:61` | `@dataclass` | `n_samples`, `n_features`, `is_binary`, `is_categorical`, `is_continuous`, `has_negative`, `description` |
| `MLTask` | `problem_context.py:10` | `str+Enum` | `classification`, `clustering`, `encoding_only`, `kernel_method`, `variational`, `unknown` |
| `ProblemContext` | `problem_context.py:21` | `@dataclass` | `task`, `algorithm`, `raw_hints`, `inferred_note`, `has_explicit_info()` |
| `SimulationResult` | `simulate.py:28` | `@dataclass` | `encoding_type`, `circuit`, `depth`, `num_qubits`, `counts: dict[str,int]`, `shots` |

---

## Os 5 Encodings

| Tipo | Qubits | Profundidade | Melhor para | Gates principais |
|------|--------|-------------|-------------|-----------------|
| `amplitude` | `ceil(log2(n))` | Profundo | Vetor grande normalizado; poucos qubits | `StatePreparation` |
| `angle` | n_features | Raso | Dados contínuos baixa dimensão; prototipagem | `Ry(x[i])` por qubit |
| `basis` | n_bits | Mínima | Dados binários/categóricos | `X` por bit=1 |
| `data_reuploading` | n_features | Médio | VQC, QNN, alta expressividade | `Ry × n_layers` + `CX` chain |
| `custom_feature_map` | n_features | Profundo | Kernels quânticos (QSVM) | `H + Rz + Ry` + `CZ` pareado |

---

## Algoritmo → Encoding

| Keyword | MLTask | Encoding preferido |
|---------|--------|--------------------|
| `qsvm`, `quantum svm`, `quantum kernel`, `qkernel` | KERNEL_METHOD | `custom_feature_map` |
| `vqc`, `variational classifier`, `qnn` | VARIATIONAL | `data_reuploading` |
| `vqe`, `qaoa` | VARIATIONAL | None (problema-específico) |
| `qgan` | VARIATIONAL | `angle` |
| `qpca` | CLUSTERING | `amplitude` |

---

## Fluxo de Dados Principal

```
CSV / array / texto
  → infer_data_profile()          → DataProfile
  → recommend_encoding()
      → infer_problem_context()   → ProblemContext
      → _recommend_encoding_from_data() → EncodingType base
      → refine_recommendation()   → EncodingType final + razão
  → para cada EncodingType:
      build_encoding_circuit()    → QuantumCircuit
      simulate_encoding_circuit() → SimulationResult
  → format_comparison_report()    → str (plain text)
      → format_encoding_ranking_section()
      → format_measurements_note_section()
      → get_encoding_tradeoffs()
```

---

## Endpoints API

| Método + Path | Descrição |
|---------------|-----------|
| `GET /health` / `/healthz` | Health check |
| `GET /` | Info do serviço |
| `POST /v1/analyze` | Perfil do dado (`DataInput` JSON) → `ProfileResponse` |
| `POST /v1/recommend` | Recomendação (`DataInput` JSON) → `RecommendResponse` |
| `POST /v1/compare` | Relatório completo (`CompareRequest` JSON) → plain text |
| `POST /v1/compare/csv` | Upload CSV multipart → relatório plain text |
| `GET /v1/tradeoffs` | Trade-offs de todos os encodings |
| `GET /v1/scenarios-guide` | Guia de cenários QML |
| `POST /v1/circuit` | Gera circuito (`CircuitRequest` JSON) → diagrama texto |
| `POST /v1/simulate` | Simula circuito (`SimulateRequest` JSON) → contagens |
| `GET /v1/analyze/text?q=` | Análise rápida por query string (legacy) |
| `GET /chat` | Serve `chat.html` |

---

## Tools do Agente LLM (encoding_agent.py)

6 funções expostas como tools OpenAI-style:

1. `analyze_data(dataset_or_description)` → perfil formatado
2. `recommend_embedding_strategy(dataset_or_description, task?, algorithm?, problem_description?)` → recomendação
3. `generate_qiskit_circuit(encoding_name, data, n_qubits?)` → diagrama do circuito
4. `simulate_circuit(encoding_name, data, n_qubits?, shots=1024)` → top-10 outcomes
5. `compare_embeddings_report(data, n_qubits?, shots?, task?, algorithm?, problem_description?)` → relatório completo
6. `explain_tradeoffs()` → texto de trade-offs

Roteador: `dispatch_tool(name, arguments)` → despacha para a função certa.

---

## Variáveis de Ambiente

| Variável | Padrão | Uso |
|----------|--------|-----|
| `LLAMA_STACK_CLIENT_API_KEY` | — | Auth Llama Stack (opcional para pipeline quantum) |
| `LLAMA_STACK_CLIENT_BASE_URL` | — | URL servidor Llama Stack |
| `PORT` | `8080` | Porta uvicorn |
| `CORS_ORIGINS` | `*` | Origins CORS (`,` separados ou `*`) |
| `UVICORN_RELOAD` | — | `1`/`true`/`yes` = hot reload |

---

## Comandos Úteis

```bash
# Instalar
pip install -e ".[api]"

# Rodar API
python scripts/run_api.py

# CLI direto
python scripts/run_encoding_agent.py "dados contínuos com 5 features"
python scripts/run_encoding_agent.py --csv dados.csv
python scripts/run_encoding_agent.py --data 0.1 0.2 0.3 --task kernel --algorithm QSVM
python scripts/run_encoding_agent.py --tradeoffs-only

# Docker
docker compose up --build

# Demo Qiskit puro
python scripts/run_quantum_example.py
```

---

## Decisões de Design Notáveis

- **Llama Stack é opcional** — todo o pipeline quantum funciona sem servidor LLM
- **AerSimulator apenas** — sem hardware quântico real; `build-essential` no Docker para compilar extensões nativas
- **Falhas silenciosas** — em `compare_embeddings()`, encodings que falham são pulados (`except Exception: continue`)
- **Relatório plain text** — `PlainTextResponse`; fácil de exibir em terminal, UI e contexto LLM
- **Bilingue PT-BR + EN** — detecção de keywords em ambos os idiomas em `data_analysis.py` e `problem_context.py`
- **OpenShift rootless** — `chgrp/chmod g=u` + `USER 1001` no Dockerfile
- **CSV tolerante** — células não-numéricas e cabeçalhos texto são ignorados automaticamente

---

## Tarefa Pendente (contexto da sessão anterior)

**Geração de frases em linguagem natural explicando o ranking de encodings.**

O objetivo era: dado que o controller ranqueou um encoding acima dos outros, gerar frases dinâmicas que citem explicitamente:
- Características estruturais do dado (`DataProfile`: n_features, is_binary, is_continuous, etc.)
- Restrições de hardware (`SimulationResult`: depth, num_qubits)
- Por que essas características favorecem o encoding escolhido sobre as alternativas

Ponto de entrada mais relevante: `encoding_ranking.py` — especialmente `_alternativa_rationale()` (linha 37) e `format_encoding_ranking_section()` (linha 66). A lógica de recomendação final está em `problem_context.py:refine_recommendation()` (linha 178).
