# Roteiro de apresentação — otimização de colunas / qubits

Use esta página junto com a seção **"Otimização de mapeamento de features"** do relatório
(`/v1/compare/csv` com `optimize_features=true` ou `python scripts/run_benchmarks.py`).

---

## Conceito em 30 segundos

> No QML, o CSV não é só entrada clássica: **a ordem das colunas decide qual feature vai para qual qubit**.
> Trocar colunas de lugar **não muda os números na tabela**, mas **muda o circuito** e o kernel \(K(x,x')\).
> Medimos isso com **KTA** (alinhamento kernel–classes) e buscamos a melhor ordem/seleção/peso **antes** de treinar QSVM.
> Referência: Fioravanti et al., [arXiv:2512.02422](https://arxiv.org/abs/2512.02422).

---

## Como ler o relatório (bloco a bloco)

| Bloco no relatório | O que dizer |
|--------------------|-------------|
| **Ordem ORIGINAL no CSV** | "Assim o arquivo chegou: coluna 0, 1, 2…" |
| **Mapeamento OTIMIZADO** | "Depois da busca, a feature X (col. CSV 2) vai para o qubit 0…" |
| **Colunas NÃO usadas** | "Estas colunas foram descartadas porque pioravam ou não ajudavam o KTA." |
| **Em uma frase** | Leia literalmente — resumo para slide ou oral. |
| **Log da busca** | "Primeiro permutamos, depois eliminamos colunas, depois ajustamos pesos." |

### Angle / IQP / custom (1 feature → 1 qubit)

- Posição 0 no vetor mapeado → qubit 0  
- Posição 1 → qubit 1  
- No **IQP**, pares de qubits geram termos \(R_{zz}(x_i \cdot x_j)\): **quem está ao lado importa**.

### Dense angle (2 features → 1 qubit)

- Posição 0 → qubit 0, rotação **Ry**  
- Posição 1 → qubit 0, rotação **Rz**  
- Posição 2 → qubit 1, **Ry** …

Por isso no benchmark sintético convém colocar **feat_a** na posição 0 (primeiro Ry do primeiro qubit).

---

## Exemplo 1 — Sintético (`synthetic_order`) — ganho grande

**Setup:** CSV com `noise_x, noise_y, feat_a, feat_b` — ruído **antes** do sinal.

**Baseline:** ruído ocupa qubits 0–1; sinal fica atrás → KTA ~0,55.

**Otimizado (típico):** `feat_a → noise_y → feat_b → noise_x` → KTA ~0,67 (**+0,12**).

**Frase pronta:**

> "Colocamos features informativas na frente do mapeamento e empurramos ruído para posições
> que menos prejudicam o kernel. Só reordenar colunas elevou o KTA em ~22%, sem mudar os dados."

---

## Exemplo 2 — Iris (`iris_binary`) — ganho modesto

**Setup:** 4 medidas botânicas, ordem padrão do sklearn.

**Otimizado (típico):** reordena pétalas/sepalas; pode **dropar** uma coluna redundante.

**Frase pronta:**

> "Em dados reais bem comportados, a ordem original já era quase boa; a otimização refina
> e remove uma feature — ganho pequeno (+2%), o que valida que o método não infla KTA à toa."

---

## Exemplo 3 — Breast Cancer (`breast_cancer_top6`) — ganho mínimo

**Setup:** 6 features de maior variância, maligno vs benigno.

**Otimizado (típico):** mantém `area error`, `worst perimeter`, etc.; descarta 2 colunas.

**Frase pronta:**

> "No Wisconsin, top features por variância já capturam o sinal; reorder + seleção dá +0,006 KTA —
> o pipeline confirma quais colunas importam para o encoding IQP escolhido."

---

## Perguntas que podem surgir

**"Isso não é só feature selection clássica?"**  
Parcialmente — também **ordem** e **peso**, específicos do circuito (pares ZZ no IQP, pares Ry/Rz no dense_angle).

**"Por que KTA e não acurácia?"**  
KTA avalia o **kernel antes** de treinar classificador; alinhado ao paper e ao `/v1/kernel`.

**"Funciona em hardware?"**  
Simulado aqui; paper [2512.02422] mostra ganho em hardware real — nosso future work é validar no IBM Quantum.

---

## Comandos para gerar o relatório na demo

```bash
python scripts/run_benchmarks.py --dataset synthetic_order --no-compare
# ou API:
curl -F "file=@dados.csv" -F "optimize_features=true" http://localhost:8080/v1/compare/csv
```
