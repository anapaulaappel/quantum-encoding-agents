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

## Exemplo 3 — Breast Cancer (`breast_cancer_fdase` / `breast_cancer_top6`)

**Default da suite (`breast_cancer_fdase`):** 30 colunas originais; o runner recorta FD-ASE (`q*` ≈ 3) **antes** do KTA — protocolo do artigo.

**Variante (`breast_cancer_top6`):** 6 features de maior variância (protocolo das Tabelas 1–3 do paper). Otimizado típico: mantém `area error`, `worst perimeter`; descarta 2 colunas; ganho ~+0,006 KTA.

**Frase pronta (FD-ASE):**

> "No Wisconsin completo, D2 ≈ 2,5 → três qubits. FD-ASE escolhe colunas originais, não PCA;
> o kernel-alive nasce perto de q* e morre na largura PCA-95%."

---

## Perguntas que podem surgir

**"Isso não é só feature selection clássica?"**  
Parcialmente — também **ordem** e **peso**, específicos do circuito (pares ZZ no IQP, pares Ry/Rz no dense_angle).

**"Por que KTA e não acurácia?"**  
KTA avalia o **kernel antes** de treinar classificador; alinhado ao paper e ao `/v1/kernel`.
O relatório também traz **kernel-alive** (geometria near/far, sem rótulo): KTA alto em kernel morto é um aviso.

**"Funciona em hardware?"**  
Sim. No `ibm_fez` rodamos os 4 circuitos Iris angle (baseline vs plano KTA, uma amostra por classe) como um Sampler job (`dagota8mhr3c73e5fb20`, 9 Sep 2026, 512 shots): o bitstring dominante bate com o Aer; o pico QPU fica 5–11 pontos mais baixo. Kernel-lite (6 pares) no mesmo backend em 30 Aug 2026: mini-KTA 0.688 vs 0.691 exato. Detalhes: [`HARDWARE.md`](HARDWARE.md).

**"FD-ASE e a busca KTA são a mesma coisa?"**  
Não. FD-ASE recorta colunas **originais** pelo D2 **antes** do encoding (`q*`). A busca KTA depois reordena/seleciona/pesa no recorte. `breast_cancer_fdase` é o default da suite; `breast_cancer_top6` é o protocolo antigo por variância.

---

## Comandos para gerar o relatório na demo

```bash
python scripts/run_benchmarks.py --dataset synthetic_order --no-compare
# ou API:
curl -F "file=@dados.csv" -F "optimize_features=true" http://localhost:8080/v1/compare/csv
```
