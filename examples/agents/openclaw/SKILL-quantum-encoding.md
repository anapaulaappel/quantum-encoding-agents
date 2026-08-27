# Skill: quantum-encoding (OpenClaw)

Use esta skill quando o usuário pedir recomendação de encoding quântico, comparação de
feature maps, kernel K_ij, KTA, simulação Qiskit ou QML.

## API base

Substitua `{{QISKIT_API_URL}}` pelo URL interno da API (ex.: `http://llama-qiskit-agents.openclaw.svc.cluster.local:8080`).

## Descobrir tools

```http
GET {{QISKIT_API_URL}}/v1/tools?format=openai
```

## Executar tool

Sempre use:

```http
POST {{QISKIT_API_URL}}/v1/tools/dispatch
Content-Type: application/json

{"name": "<tool_name>", "arguments": { ... }}
```

## Fluxo sugerido

1. `analyze_data` — perfil do dataset ou descrição
2. `recommend_embedding_strategy` — com `task`, `algorithm`, `hardware_profile` se souber
3. Se houver CSV com labels → `compare_csv_embeddings` (ranking KTA)
4. Se precisar kernel explícito → `compute_quantum_kernel`
5. `generate_qiskit_circuit` + `simulate_circuit` para validar

## One-shot (sem multi-step)

Para resposta única com código Qiskit + explicação template (sem loop LLM):

```http
POST {{QISKIT_API_URL}}/v1/recommend/explain
```

## Health

```http
GET {{QISKIT_API_URL}}/healthz
```

## Web UI (browser)

Para testes manuais multi-turn (personas Circuit/Quanta):

```
GET {{QISKIT_API_URL}}/chat
```

Modo Agente usa `POST /v1/agent/chat`; modo Direto usa `/v1/compare/csv`.
