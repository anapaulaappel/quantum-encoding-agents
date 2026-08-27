# Benchmark suite

Conjunto reprodutível para avaliar **KTA**, **ranking de encodings** e **otimização de ordem/seleção/peso** ([arXiv:2512.02422](https://arxiv.org/abs/2512.02422)).

## Instalação

```bash
pip install -e ".[benchmark]"
```

Requer `scikit-learn` (extra opcional, não entra no core da API).

## Datasets

| Slug | Origem | Amostras (default) | Features | Notas |
|------|--------|-------------------|----------|-------|
| `iris_binary` | UCI Iris via sklearn | 50 | 4 | Setosa vs versicolor — **rápido**, bom para CI |
| `breast_cancer_top6` | Wisconsin BC | 40 | 6 (top variância) | Binário; reduz custo O(d²) do IQP |
| `wine_binary` | UCI Wine | 40 | 13 | Classes 0 vs 1 |
| `synthetic_order` | Gerado | 24 | 4 | Ruído antes do sinal — sensível a permutação |

Todos usam **labels binários** (KTA requer ≥2 classes; comparamos pares de classes em Iris/Wine).

## Executar

```bash
# Suite completa (~ minutos, depende da máquina)
python scripts/run_benchmarks.py

# Rápido: só Iris + sintético, só métricas KTA/otimização
python scripts/run_benchmarks.py --quick

# Um dataset
python scripts/run_benchmarks.py --dataset iris_binary

# Exportar JSON (artigo / CI artifacts)
python scripts/run_benchmarks.py --json benchmarks/results/latest.json
```

## Métricas reportadas

Por dataset:

- **baseline_best_kta** — melhor KTA entre os 7 encodings (ordem CSV original)
- **optimization_delta** — ganho KTA na busca ordem/seleção/peso (`optimize_feature_encoding`)
- **compare_delta** — melhor KTA no ranking do compare com vs. sem `optimize_features`

## Testes (pytest)

```bash
pip install -e ".[benchmark,dev]"
pytest tests/test_benchmarks.py -q
```

Smoke test usa apenas `iris_binary` com amostras reduzidas.

## Hardware IBM (cota ~10 min/mês)

Validação enxuta no QPU — **não** replica KTA N×N no chip (caro demais).

```bash
pip install -e ".[benchmark,ibm]"
python scripts/run_hardware_benchmarks.py --preset week1_iris --dry-run
python scripts/run_hardware_benchmarks.py --preset week1_iris --execute
```

Presets mensais: `week1_iris`, `week2_synthetic`, `week3_breast` — ver [`HARDWARE.md`](HARDWARE.md).

## Uso no artigo

Cite os slugs e parâmetros (`max_kta_samples`, top-6 variância no breast cancer) para reproducibilidade. Resultados variam levemente com versão Qiskit/Aer; fixe seed (default 42).
