# Benchmark em hardware IBM Quantum (~10 min/mês)

Este roteiro separa **simulador** (suite completa) de **hardware** (validação enxuta).

## Instalação

```bash
pip install -e ".[benchmark,ibm]"
ibm-quantum-login   # ou export QISKIT_IBM_TOKEN=...
export IBM_QUANTUM_BACKEND=ibm_fez   # opcional
```

## Estratégia para 10 min/mês

| Semana | Preset | O quê | Jobs QPU | Tempo QPU (est.) |
|--------|--------|-------|----------|------------------|
| 1 | `week1_iris` | Iris, **angle** (depth 1) | 4 | ~2–5 min + fila |
| 2 | `week2_synthetic` | Sintético, **dense_angle** | 4 | ~2–5 min + fila |
| 3 | `week3_breast` | Breast cancer top-6, **angle** | 4 | ~3–6 min + fila |
| 4 | — | Só simulador / artigo | 0 | 0 |

Cada preset executa **4 circuits**:

- classe 0 — plano CSV (baseline)
- classe 0 — plano otimizado (KTA)
- classe 1 — baseline
- classe 1 — otimizado

Isso valida que **ambos os planos rodam no chip** e compara histogramas de medição — sem tentar matriz kernel N×N completa no QPU (proibitivo na cota).

## Comandos

```bash
# Planejar (transpile no backend — não gasta QPU se falhar auth, só avisa)
python scripts/run_hardware_benchmarks.py --preset week1_iris --dry-run

# Esta semana — Iris no hardware
python scripts/run_hardware_benchmarks.py --preset week1_iris --execute \
  --json benchmarks/results/hardware_iris.json

# Mês que vem — sintético (ordem de colunas)
python scripts/run_hardware_benchmarks.py --preset week2_synthetic --execute
```

## Simulador vs hardware neste benchmark

| Etapa | Onde | Métrica |
|-------|------|---------|
| Otimização ordem/peso | Simulador (statevector) | KTA baseline vs otimizado |
| Transpile | Backend IBM (local API) | depth, 2q gates, layout |
| Execução | QPU | histograma, job_id, tempo wall |

**KTA completo no hardware** exigiria fidelity/Swap tests para cada par \(K_{ij}\) — centenas de jobs. Fora do budget de 10 min. O paper [2512.02422](https://arxiv.org/abs/2512.02422) usa pipelines maiores; nós documentamos validação **NISQ-realista** enxuta.

## Encodings recomendados no hardware

| Encoding | Depth típico | Quando usar no preset |
|----------|--------------|------------------------|
| `angle` | 1 | Default — máximo de shots úteis |
| `dense_angle` | 2 | Sintético / Iris com 4 features |
| `basis` | ≤ d | Só dados binários |
| `iqp` | alto | **Evitar** na cota mensal |

## O que dizer na apresentação

> “No simulador otimizamos ordem de colunas por KTA. No IBM Quantum rodamos **4 circuits** — baseline vs plano otimizado, uma amostra por classe — para provar que o mapeamento transpila e executa no NISQ real. KTA completo no chip ficaria fora da cota de 10 min/mês; validamos execução + coerência com o ganho simulado.”

## Suite simulador (complementar)

```bash
python scripts/run_benchmarks.py --quick
python scripts/run_benchmarks.py --dataset breast_cancer_top6 --no-compare
```

## Troubleshooting

| Problema | Ação |
|----------|------|
| `qiskit-ibm-runtime não instalado` | `pip install -e ".[ibm]"` |
| Token inválido | `ibm-quantum-login` ou `QISKIT_IBM_TOKEN` |
| Fila longa | `--backend` com fila menor; rodar em horário off-peak |
| Job falhou | Preferir `angle`; reduzir `--shots 256` |

## Future work

- Kernel-lite no hardware: **feito** (Iris angle, 6 pares, 1 job Sampler em `ibm_fez`, 2026-08-30; `benchmarks/results/hardware_kernel_lite.json`)
- Integrar `job_id` e counts no MLflow / artigo como tabela hardware vs sim — histogramas em `benchmarks/results/hardware_iris.json`
