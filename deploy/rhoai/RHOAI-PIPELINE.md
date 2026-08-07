# Kubeflow Pipeline — Quantum Encoding Analysis

Pipeline para **Red Hat OpenShift AI (RHOAI)** com Data Science Pipelines (DSP).
Orquestra análise de dados quânticos, recomendação de encoding, comparação,
cálculo de kernel e logging no MLflow — tudo usando a API `llama-qiskit-agents`.

---

## Arquitetura do pipeline

```
CSV input + parâmetros QML
        │
        ▼
┌───────────────────┐
│  1. analyze_data  │  POST /v1/analyze → DataProfile JSON
└────────┬──────────┘
         │
         ▼
┌───────────────────────────┐
│  2. recommend_encoding    │  POST /v1/recommend/explain
│  (+ Bloch sphere opt-in)  │  → encoding, explanation, qiskit_code
└────────┬──────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌────────────────────┐
│ 3.      │ │ 4. compute_kernel  │  POST /v1/kernel → K[i,j] + KTA
│ compare │ │ (se labels         │
│ 7 encs  │ │  fornecidos)       │
└────┬────┘ └────────┬───────────┘
     │               │
     └───────┬────────┘
             │
             ▼
   ┌──────────────────┐
   │ 5. log_to_mlflow │  MLflow: params, metrics, heatmap, código
   └──────────────────┘
```

---

## Pré-requisitos

```bash
# SDK KFP (compile local, submeta via CLI ou UI)
pip install kfp>=2.0

# Acesso ao cluster OpenShift
oc login --server=https://<SEU_CLUSTER>:6443

# A API llama-qiskit-agents deve estar rodando no cluster (deploy/openshift/)
oc get svc llama-qiskit-agents -n openclaw
```

---

## Uso rápido

### Compilar o pipeline

```bash
cd deploy/rhoai
python pipeline.py
# Gera: quantum_encoding_pipeline.yaml
```

### Submeter via script

```bash
# Obter endpoint do DSP
DSP_HOST=$(oc get route -n rhoai-dsp data-science-pipelines-api \
  -o jsonpath='{.spec.host}')

# Exemplo básico (dados de exemplo, sem labels)
python submit.py --dsp-endpoint "https://$DSP_HOST"

# Com CSV real + contexto QML + hardware IBM Eagle
python submit.py \
  --dsp-endpoint "https://$DSP_HOST" \
  --csv meus_dados.csv \
  --labels "0,0,0,1,1,1" \
  --task kernel \
  --alg QSVM \
  --hw-error 5e-3 \
  --connectivity heavy-hex \
  --max-depth 20 \
  --mlflow-uri "https://mlflow.apps.<CLUSTER>/api/2.0"

# Apenas compilar (sem submeter)
python submit.py --compile-only
```

### Submeter via RHOAI UI

1. Acesse `Data Science Pipelines` no RHOAI
2. `Import pipeline` → selecione `quantum_encoding_pipeline.yaml`
3. Configure os parâmetros e clique em `Create run`

---

## Parâmetros do pipeline

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `csv_content` | 3 amostras de exemplo | CSV como string (linhas separadas por `\n`) |
| `labels_csv` | `""` | Rótulos por amostra (`"0,0,1,1"`). Vazio = sem cálculo de kernel |
| `task` | `""` | Tarefa QML: `classification`, `kernel`, `variational`, etc. |
| `algorithm` | `""` | Algoritmo alvo: `QSVM`, `VQC`, `QNN`, etc. |
| `problem_description` | `""` | Texto livre sobre o problema |
| `gate_error_rate` | `0.0` | Taxa de erro de porta. `>= 1e-3` ativa ajuste NISQ |
| `connectivity` | `all-to-all` | Topologia do chip: `heavy-hex`, `linear`, `grid` |
| `max_depth_budget` | `0` | Profundidade máxima de circuito (`0` = sem limite) |
| `api_url` | URL interna do cluster | Endpoint da API `llama-qiskit-agents` |
| `mlflow_tracking_uri` | `""` | URI do MLflow. Vazio = só métricas KFP nativas |
| `experiment_name` | `quantum-encoding` | Nome do experimento no MLflow |
| `lang` | `pt` | Idioma das respostas: `pt` ou `en` |
| `shots` | `512` | Shots para simulação (etapa de comparação) |
| `max_kernel_samples` | `20` | Limite de amostras para o kernel (custo O(N²)) |

---

## Artefatos gerados por etapa

| Etapa | Artefato | Conteúdo |
|---|---|---|
| analyze | `profile_output` | DataProfile JSON (n_features, is_binary, etc.) |
| recommend | `recommendation_output` | Encoding, explanation, qiskit_code, bloch_sphere_b64 |
| compare | `comparison_output` | Relatório de ranking dos 7 encodings (plain text) |
| kernel | `kernel_output` | Kernel N×N, stats (KTA), heatmap PNG base64 |
| log | `mlflow_metrics` | Métricas KFP nativas (circuit_depth, kta, etc.) |

### No MLflow (quando `mlflow_tracking_uri` configurado)

| Tipo | Item |
|---|---|
| Params | `recommended_encoding`, `task_interpreted`, `n_features`, `is_continuous`, `hw_constraints` |
| Metrics | `circuit_depth`, `circuit_qubits`, `kernel_kta`, `kernel_separability_hint` |
| Artifacts | `explanation.md`, `circuit_code.py`, `visualizations/kernel_heatmap.png`, `visualizations/bloch_sphere.png` |

---

## Casos de uso

### 1. Exploração inicial (sem contexto)

```bash
python submit.py --csv dados.csv
```
→ Recomenda encoding baseado só no perfil dos dados.

### 2. QSVM com hardware IBM Eagle

```bash
python submit.py \
  --csv features.csv \
  --labels "0,0,0,1,1,1" \
  --task kernel \
  --alg QSVM \
  --hw-error 5e-3 \
  --connectivity heavy-hex \
  --max-depth 20
```
→ Ajusta recomendação para hardware NISQ (gate_error >= p*),
   calcula KTA para avaliar separabilidade, salva tudo no MLflow.

### 3. VQC com restrição de qubits

```bash
python submit.py \
  --csv features.csv \
  --task variational \
  --alg VQC \
  --problem "classificar espectros NIR com poucos qubits"
```
→ Recomenda `data_reuploading` ou `dense_angle` conforme o perfil.

---

## Troubleshooting

### Pipeline não consegue chamar a API

```bash
# Verificar se o serviço está acessível no cluster
oc exec -it deployment/openclaw -n openclaw -c gateway -- \
  curl http://llama-qiskit-agents.openclaw.svc.cluster.local:8080/healthz
```

### `kfp.Client` não conecta ao DSP

```bash
# Verificar a rota
oc get route -n rhoai-dsp data-science-pipelines-api

# Autenticar (RHOAI usa bearer token do OpenShift)
TOKEN=$(oc whoami -t)
python submit.py --dsp-endpoint "https://$DSP_HOST" \
  # kfp.Client usa KUBECONFIG automaticamente quando disponível
```

### Etapa de kernel muito lenta

Reduza `--max-kernel-samples`. A complexidade é O(N²) — 50 amostras = 2500 pares.
Para datasets grandes, use amostragem estratificada antes de passar ao pipeline.
