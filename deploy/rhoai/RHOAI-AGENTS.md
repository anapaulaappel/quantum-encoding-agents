# Kubeflow Pipeline — variante agentica (RHOAI)

Pipeline **quantum-encoding-analysis-agents** para Red Hat OpenShift AI (RHOAI).
Estende o pipeline clássico (`RHOAI-PIPELINE.md`) com:

- **Compare via tool dispatch** — `POST /v1/tools/dispatch` → `compare_csv_embeddings` (ranking KTA + simulabilidade)
- **Resumo agentico** — `POST /v1/agent/chat` (ReAct + tools; requer LLM configurado na API)
- **Consolidação de artefatos** — JSON de run + métricas KFP (**sem etapa MLflow** nesta variante)

---

## Arquitetura

```
CSV + parâmetros QML
        │
        ▼
┌───────────────────┐
│  1. analyze_data  │  POST /v1/analyze
└────────┬──────────┘
         │
         ▼
┌───────────────────────────┐
│  2. recommend_encoding    │  POST /v1/recommend/explain
└────────┬──────────────────┘
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌─────────────────┐     ┌────────────────────────────┐
│ 3. agent_chat   │     │ 4. compare_csv (dispatch)  │
│ /v1/agent/chat  │     │ /v1/tools/dispatch         │
└────────┬────────┘     └─────────────┬──────────────┘
         │                            │
         │         ┌──────────────────┘
         │         ▼
         │   ┌────────────────────┐
         │   │ 5. compute_kernel  │  POST /v1/kernel → K_{ij} + KTA
         │   └─────────┬──────────┘
         │             │
         └──────┬──────┘
                ▼
      ┌─────────────────────┐
      │ 6. emit_run_summary │  run_summary.json + agent_reply.txt
      └─────────────────────┘
```

---

## Pré-requisitos

1. API `llama-qiskit-agents` deployada no cluster (ver `deploy/openshift/`).
2. **Variante agentica:** variáveis `AGENT_*` no Deployment da API apontando para um Model Serving RHOAI (vLLM) ou Ollama sidecar — ver `deploy/openshift/agent-env.example.yaml`.
3. `pip install kfp>=2.0` no ambiente de compilação/submissão.

Sem `AGENT_*` configurado, a etapa 3 falha; o pipeline clássico (`--variant classic`) continua válido para runs só determinísticos + MLflow.

---

## Compilar e submeter

```bash
cd deploy/rhoai

# Compilar YAML agentico
python pipeline_agents.py
# → quantum_encoding_pipeline_agents.yaml

# Submeter (requer oc login + DSP endpoint)
python submit.py --variant agents \
  --dsp-endpoint https://data-science-pipelines-api-rhoai-dsp.apps.<cluster> \
  --csv dados.csv \
  --labels "0,0,1,1" \
  --problem "Classificação binária tabular" \
  --agent-persona expert

# Só compilar
python submit.py --variant agents --compile-only
```

Parâmetros extras da variante agentica:

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `agent_persona` | `expert` | Persona passada a `/v1/agent/chat` |
| `label_column` | `""` | Coluna de rótulo no CSV (se houver header) |
| `max_tool_rounds` | `6` | Limite de rodadas ReAct no agente |

---

## Artefatos de saída

| Artefato | Conteúdo |
|----------|----------|
| `run_summary.json` | Recomendação, top-K compare, KTA, trecho do reply agentico |
| `agent_reply.txt` | Texto completo retornado por `/v1/agent/chat` |
| Métricas KFP | `recommended_encoding`, `kta_score`, `compare_top_encoding`, `agent_tool_calls` |

---

## MLflow (planejado, não implementado nesta variante)

O pipeline **clássico** registra params/metrics/artefatos via `log_to_mlflow.py`.
Para a variante agentica, a integração MLflow fica **documentada** para uma fase posterior:

| Campo MLflow | Origem prevista |
|--------------|-----------------|
| `params.recommended_encoding` | recommend |
| `metrics.kta_score` | kernel |
| `metrics.compare_top_kta` | compare dispatch |
| `metrics.agent_tool_calls` | agent chat |
| `artifacts.run_summary.json` | emit_run_summary |
| `artifacts.agent_reply.txt` | emit_run_summary |

**Implementação futura:** adicionar componente `log_mlflow_agents.py` que aceite `agent_output` + `run_summary`, ou reutilizar `log_mlflow.py` com inputs opcionais. Não incluído agora para manter o escopo da variante agentica focado em `/v1/agent/chat` + tools.

---

## Model Serving (vLLM) no RHOAI

Fluxo típico:

1. Importar modelo no RHOAI Model Registry (ex.: Llama 3.x instruct).
2. Criar **Single Model Serving Runtime** (vLLM) no namespace do projeto.
3. Configurar na API:

```yaml
env:
  - name: AGENT_BACKEND
    value: openai_compatible
  - name: OPENAI_API_BASE
    value: http://vllm-serving.<namespace>.svc.cluster.local:8000/v1
  - name: AGENT_MODEL
    value: meta-llama/Llama-3.1-8B-Instruct
  - name: OPENAI_API_KEY
    value: "not-used"
```

4. Validar: `curl -s $API/v1/agent/chat -H 'Content-Type: application/json' -d '{"message":"teste","persona":"expert"}'`

---

## Relação com o artigo

- Compare dispatch expõe o mesmo ranking KTA + kernel-alive descrito no artigo (`K_ij`, simulabilidade, FD-ASE).
- O agente resume resultados para o pesquisador — alinhado à §4.1 (tools + `/chat`).
- Otimização de **ordem/peso de features** ([arXiv:2512.02422](https://arxiv.org/abs/2512.02422)) está no core (`encoding_optimization.py`) e na tool `compare_csv_embeddings` (`optimize_features=true`).

---

## Ver também

- [RHOAI-PIPELINE.md](./RHOAI-PIPELINE.md) — pipeline clássico + MLflow
- `deploy/openshift/agent-env.example.yaml` — snippet de env para Deployment
- `examples/agents/` — integração OpenClaw / Ollama / HTTP
