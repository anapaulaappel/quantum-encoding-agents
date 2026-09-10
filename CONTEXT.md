# Project Context: llama-qiskit-agents

> Documento de referência rápida para retomar o projeto sem reler o código inteiro.

---

## Propósito

Pacote Python que conecta **Llama Stack** (LLMs com function-calling) ao **Qiskit + Qiskit-Aer** para recomendar, gerar e simular estratégias de **encoding de dados quânticos** (Quantum Machine Learning). Dado um dataset (CSV, array numérico, ou descrição textual) + contexto QML opcional, o agente:

1. Analisa o perfil do dado e, com tabela 2-D, estima D2 → orçamento `q*` via FD-ASE
2. Recomenda a melhor estratégia de encoding (por padrão no recorte `q*`)
3. Gera o circuito Qiskit correspondente
4. Simula via `AerSimulator` (CPU). Histogramas e kernel-lite no IBM Quantum: `hardware_run.py` + `scripts/run_hardware_benchmarks.py`
5. Produz relatório comparativo das 7 estratégias (KTA se labels, kernel-alive, sweep de `q`)

Exposto via **FastAPI** (`/v1/…`), **UI web agentica** (`/chat` → `/v1/agent/chat`) e tools HTTP (`/v1/tools/dispatch`).

---

## Layout

```
llama-qiskit-agents/
├── src/llama_qiskit_agents/
│   ├── agents/
│   │   ├── tool_registry.py   # 9 tools + schemas + dispatch_tool (fonte única)
│   │   ├── encoding_agent.py  # Re-exports retrocompatíveis
│   │   ├── harness.py         # ReAct loop (Ollama / Llama Stack / OpenAI-compatible)
│   │   └── client.py          # LlamaStackClient factory + chat_completion helper
│   ├── api/
│   │   ├── app.py             # FastAPI: todos os endpoints
│   │   ├── schemas.py         # Pydantic models request/response
│   │   └── static/chat.html   # UI web → /v1/agent/chat (+ modo direto /v1/compare/csv)
│   └── quantum/
│       ├── encodings.py        # EncodingType enum + 7 circuit builders
│       ├── data_analysis.py    # CSV load, DataProfile, recommend_encoding
│       ├── fractal.py          # D2 (LiBOC) + FD-ASE + FractalBudget
│       ├── qubit_sweep.py      # sweep q: FD-ASE vs PCA vs prefixo (probe = angle)
│       ├── preprocessing.py    # min-max ângulo, L2 amplitude, avisos
│       ├── problem_context.py  # MLTask, ProblemContext, refine_recommendation
│       ├── hardware_profile.py # HardwareProfile, limiar p*=1e-3 (NISQ-aware)
│       ├── hardware_run.py     # Sampler IBM (week1_iris / kernel-lite)
│       ├── explanation.py      # detect_language + narrativas PT/EN + código Qiskit
│       ├── visualization.py    # Bloch sphere via StatevectorSimulator + matplotlib
│       ├── kernel.py           # FidelityStatevectorKernel, KTA, kernel-alive
│       ├── encoding_optimization.py  # ordem/seleção/peso por KTA
│       ├── encoding_ranking.py # Formatação do ranking + nota de medições
│       ├── simulate.py         # AerSimulator orchestration + format_comparison_report
│       └── circuits.py         # Bell + simple circuit (demo apenas)
├── scripts/
│   ├── run_encoding_agent.py  # CLI pipeline quântico (sem LLM)
│   ├── run_agent_harness.py   # CLI agentico (LLM + tools)
│   ├── run_api.py             # Inicia uvicorn
│   └── run_quantum_example.py # Demo Qiskit puro
├── examples/agents/           # OpenClaw, Ollama, HTTP — guia de integração
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
| `EncodingType` | `encodings.py:11` | `str+Enum` | `amplitude`, `angle`, `dense_angle`, `iqp`, `basis`, `data_reuploading`, `custom_feature_map` |
| `DataProfile` | `data_analysis.py` | `@dataclass` | tipo + D2 + `q*` + `selected_columns` (FD-ASE) + `fractal_selection_applied` |
| `MLTask` | `problem_context.py:10` | `str+Enum` | `classification`, `clustering`, `encoding_only`, `kernel_method`, `variational`, `unknown` |
| `ProblemContext` | `problem_context.py:21` | `@dataclass` | `task`, `algorithm`, `raw_hints`, `inferred_note`, `has_explicit_info()` |
| `SimulationResult` | `simulate.py:28` | `@dataclass` | `encoding_type`, `circuit`, `depth`, `num_qubits`, `counts: dict[str,int]`, `shots` |

---

## Os 7 Encodings

| Tipo | Qubits | Profundidade | Melhor para | Gates principais |
|------|--------|-------------|-------------|-----------------|
| `amplitude` | `ceil(log2(n))` | Profundo | Vetor grande normalizado; poucos qubits | `StatePreparation` |
| `angle` | n_features | 1 | Dados contínuos ≤4 features; prototipagem | `Ry(x[i])` por qubit |
| `dense_angle` | `ceil(n/2)` | 2 | 5–12 features; metade dos qubits do angle | `Ry(x[2i])·Rz(x[2i+1])` |
| `iqp` | n_features | ~3n | 8–16 features; kernel quântico | `H + Rz(x²) + Rzz(xᵢ·xⱼ)` |
| `basis` | n_bits | Mínima | Dados binários/categóricos | `X` por bit=1 |
| `data_reuploading` | n_features | Médio | VQC, QNN, alta expressividade | `Ry × n_layers` + `CX` chain |
| `custom_feature_map` | n_features | Profundo | Kernels quânticos (QSVM) arbitrários | `H + Rz + Ry` + `CZ` pareado |

### Hardware-aware: `HardwareProfile`

Novo módulo `hardware_profile.py`. Limiar crítico `p* ≈ 1e-3` (Sammartino arXiv:2606.05387):
- `gate_error_rate >= p*` → encodings profundos (amplitude, custom_feature_map, IQP) são substituídos
- `connectivity: "heavy-hex" | "linear"` → aviso de overhead de SWAP para custom_feature_map
- `max_depth_budget` → aviso quando depth estimado excede o budget
- `max_qubits` → aviso de qubits insuficientes

### Fractal + kernel-alive

- `fractal.py`: D2 via LiBOC; FD-ASE devolve **índices de colunas originais** (não misturas PCA). `q* = max(2, ceil(D2))`. Precisa N≥32 amostras e ≥3 features para FD-ASE.
- `apply_fractal_budget=false`: D2 e a lista FD-ASE ficam no perfil; o circuito usa a largura original.
- `kernel.py`: KTA (rótulo) e kernel-alive (geometria, independente de rótulo) na mesma `K`. Alive se near≥0.25, near/far≥2, média off-diag≥0.03 (arXiv:2609.00475).
- `qubit_sweep.py`: curva de `q` com probe = angle, três vistas (FD-ASE / PCA / prefixo CSV).

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
  → infer_data_profile(apply_fractal_budget=True)
      → DataProfile (+ D2, q*, selected_columns)
  → recommend_encoding()                    # angle/IQP limitados a q* se recorte ligado
      → infer_problem_context()   → ProblemContext
      → _recommend_encoding_from_data() → EncodingType base
      → refine_recommendation()   → EncodingType final + razão
  → compare_embeddings():
      recorte FD-ASE (se ligado) → KTA + kernel-alive por encoding
      → run_qubit_budget_sweep() (FD-ASE vs PCA vs prefixo)
  → format_comparison_report()    → str (plain text)
```

---

## Endpoints API

| Método + Path | Descrição |
|---------------|-----------|
| `GET /health` / `/healthz` | Health check |
| `GET /` | Info do serviço |
| `POST /v1/analyze` | Perfil do dado (`DataInput` JSON) → `ProfileResponse` (D2, q*, FD-ASE) |
| `POST /v1/recommend` | Recomendação (`DataInput` JSON) → `RecommendResponse` |
| `POST /v1/recommend/explain` | **Principal** — recomendação + explicação PT/EN + código Qiskit + Bloch sphere |
| `POST /v1/compare` | Ranking + KTA + kernel-alive + sweep q (`CompareRequest`) → plain text |
| `POST /v1/compare/csv` | Upload CSV multipart (`optimize_features`, `apply_fractal_budget`, `sweep_qubit_budget`) |
| `POST /v1/kernel` | Matriz K_ij + KTA + kernel-alive + heatmap PNG |
| `GET /v1/tools` | Schemas OpenAI / OpenClaw |
| `POST /v1/tools/dispatch` | Executa tool (integração agentes) |
| `POST /v1/agent/chat` | Turno LLM + tools (env AGENT_*) |
| `GET /v1/tradeoffs` | Trade-offs de todos os encodings |
| `GET /v1/scenarios-guide` | Guia de cenários QML |
| `POST /v1/circuit` | Gera circuito (`CircuitRequest` JSON) → diagrama texto |
| `POST /v1/simulate` | Simula circuito (`SimulateRequest` JSON) → contagens |
| `GET /v1/analyze/text?q=` | Análise rápida por query string (legacy) |
| `GET /chat` | UI web: chat agentico (default) ou compare CSV direto |

---

## Tools do agente (`tool_registry.py`)

9 tools — fonte única para REST, harness e OpenClaw:

1. `analyze_data` — perfil do dado
2. `recommend_embedding_strategy` — recomendação (+ hardware_profile opcional)
3. `generate_qiskit_circuit` — diagrama ASCII
4. `simulate_circuit` — contagens Aer
5. `compare_embeddings_report` — ranking 7 encodings + KTA + kernel-alive + sweep q
6. `compare_csv_embeddings` — CSV texto + KTA se labels + flags fractal/sweep
7. `compute_quantum_kernel` — K_ij + stats + KTA + kernel-alive
8. `explain_tradeoffs`
9. `scenarios_guide`

Roteador: `dispatch_tool(name, arguments)`. Harness: `agents/harness.py` + `scripts/run_agent_harness.py`.
Integração: `examples/agents/README.md`.

---

## Variáveis de Ambiente

| Variável | Padrão | Uso |
|----------|--------|-----|
| `LLAMA_STACK_CLIENT_API_KEY` | — | Auth Llama Stack (opcional para pipeline quantum) |
| `LLAMA_STACK_CLIENT_BASE_URL` | — | URL servidor Llama Stack |
| `PORT` | `8080` | Porta uvicorn |
| `CORS_ORIGINS` | `*` | Origins CORS (`,` separados ou `*`) |
| `UVICORN_RELOAD` | — | `1`/`true`/`yes` = hot reload |
| `AGENT_BACKEND` | `ollama` | `ollama` \| `llama_stack` \| `remote` |
| `AGENT_MODEL` | `qwen2.5-coder:14b` | Modelo LLM |
| `OPENAI_API_BASE` | `{OLLAMA_HOST}/v1` | API OpenAI-compatible |
| `OLLAMA_HOST` | `http://localhost:11434` | Host Ollama |

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
# UI: http://localhost:8080/chat

# Harness agentico (CLI)
python scripts/run_agent_harness.py --persona expert "Qual encoding para QSVM?"

# Demo Qiskit puro
python scripts/run_quantum_example.py
```

---

## Decisões de Design Notáveis

- **Llama Stack é opcional** — pipeline quantum e modo Direto da UI funcionam sem LLM; modo Agente em `/chat` requer `AGENT_*` + Ollama/vLLM
- **Simulador por padrão** — Aer/statevector no compare e no kernel; QPU opcional via `.[ibm]` (`ibm_fez`, histogramas + kernel-lite, cota ~10 min/mês)
- **Falhas silenciosas** — em `compare_embeddings()`, encodings que falham são pulados (`except Exception: continue`)
- **Relatório plain text** — `PlainTextResponse`; fácil de exibir em terminal, UI e contexto LLM
- **Bilingue PT-BR + EN** — detecção de keywords em ambos os idiomas em `data_analysis.py` e `problem_context.py`
- **OpenShift rootless** — `chgrp/chmod g=u` + `USER 1001` no Dockerfile
- **CSV tolerante** — células não-numéricas e cabeçalhos texto são ignorados automaticamente
- **FD-ASE ≠ PCA** — devolve colunas originais; `apply_fractal_budget=false` só desliga o recorte no circuito

---

## Estruturas de Dados Chave (atualizado)

| Struct | Arquivo | Tipo | Campos principais |
|--------|---------|------|-------------------|
| `DataProfile` | `data_analysis.py` | `@dataclass` | tipo + D2 + `q*` + `selected_columns` + `fractal_selection_applied` |
| `HardwareProfile` | `hardware_profile.py` | `@dataclass` | `gate_error_rate`, `max_depth_budget`, `max_qubits`, `connectivity`, `backend_name` |
| `FractalBudget` | `fractal.py` | `@dataclass` | `intrinsic_dimension`, `qubit_budget`, `selected_columns` |
| `KernelGeometry` | `kernel.py` | `@dataclass` | `fid_near`, `fid_far`, `near_far_ratio`, `mean_offdiag`, `kernel_alive` |
| `MLTask` | `problem_context.py` | `str+Enum` | `classification`, `clustering`, `encoding_only`, `kernel_method`, `variational`, `unknown` |
| `ProblemContext` | `problem_context.py` | `@dataclass` | `task`, `algorithm`, `raw_hints`, `inferred_note`, `has_explicit_info()` |
| `SimulationResult` | `simulate.py` | `@dataclass` | `encoding_type`, `circuit`, `depth`, `num_qubits`, `counts: dict[str,int]`, `shots` |
| `HardwareProfileInput` | `api/schemas.py` | Pydantic | espelho do dataclass para API HTTP |

## Estado atual do roadmap

- [x] **A1a** — `dense_angle` encoding (Sammartino 2026)
- [x] **A1b** — `IQP` encoding (Havlíček 2019)
- [x] **A2** — `hardware_profile` NISQ-aware (p* threshold)
- [x] **B1** — Bloch sphere visualization (`matplotlib` + statevector)
- [x] **B2** — `POST /v1/kernel` (FidelityStatevectorKernel + heatmap + kernel-alive)
- [x] **C1** — Kubeflow Pipeline para RHOAI (deploy/rhoai/)
- [x] **C11** — Otimização ordem/seleção/peso de features por KTA
- [x] **D2 / FD-ASE** — orçamento `q*` antes do encoding e do KTA (arXiv:2609.00475)
- [x] **Sweep q** — FD-ASE vs PCA vs prefixo no `/v1/compare`
- [x] **IBM Quantum** — `ibm_fez`: histogramas Iris (2026-09-09) + kernel-lite (2026-08-30)

Referências chave: arXiv:2606.05387 (Sammartino survey), Nature 2019 (Havlíček kernel), arXiv:2609.00475 (D2 / kernel-alive).
