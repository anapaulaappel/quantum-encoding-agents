"""
Otimização clássica de mapeamento de features antes do ansatz.

Inspirado em Fioravanti et al. (arXiv:2512.02422): ordem, seleção e ponderação
de features alteram o kernel implícito K(x,x') mesmo com o mesmo circuito U_φ.

Busca heurística em três fases (greedy), maximizando KTA quando labels existem:
  1. Permutação de colunas (exaustiva se d ≤ 7, hill-climbing por trocas caso contrário)
  2. Eliminação de colunas (backward selection)
  3. Ponderação por feature (grid {0.5, 1.0, 2.0})
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.kernel import compute_kta_by_encoding
from llama_qiskit_agents.quantum.preprocessing import FeatureBounds

WEIGHT_CANDIDATES: tuple[float, ...] = (0.5, 1.0, 2.0)
EXHAUSTIVE_ORDER_LIMIT: int = 7


@dataclass
class FeatureEncodingPlan:
    """Plano clássico: subconjunto ordenado de colunas + pesos opcionais."""

    column_indices: list[int]
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [1.0] * len(self.column_indices)
        if len(self.weights) != len(self.column_indices):
            raise ValueError("weights deve ter o mesmo tamanho que column_indices")

    @classmethod
    def identity(cls, n_features: int) -> FeatureEncodingPlan:
        return cls(column_indices=list(range(n_features)))

    def copy(self) -> FeatureEncodingPlan:
        return FeatureEncodingPlan(
            column_indices=list(self.column_indices),
            weights=list(self.weights),
        )

    def apply(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        sub = x[:, self.column_indices]
        w = np.asarray(self.weights, dtype=float)
        return sub * w

    def apply_vector(self, row: np.ndarray) -> np.ndarray:
        return self.apply(np.asarray(row, dtype=float).reshape(1, -1))[0]

    def _col_label(self, col_idx: int, column_names: list[str] | None) -> str:
        if column_names and 0 <= col_idx < len(column_names):
            return column_names[col_idx]
        return f"col_{col_idx}"

    def describe(self, column_names: list[str] | None = None) -> str:
        parts: list[str] = []
        for idx, w in zip(self.column_indices, self.weights):
            name = self._col_label(idx, column_names)
            if abs(w - 1.0) < 1e-9:
                parts.append(name)
            else:
                parts.append(f"{name}×{w:g}")
        return " → ".join(parts)

    def dropped_columns(self, n_original: int) -> list[int]:
        kept = set(self.column_indices)
        return [i for i in range(n_original) if i not in kept]

    def format_csv_baseline_order(self, column_names: list[str] | None = None) -> list[str]:
        """Ordem das colunas no CSV antes de qualquer otimização."""
        n = max(self.column_indices) + 1 if self.column_indices else 0
        if column_names:
            n = len(column_names)
        lines: list[str] = []
        for i in range(n):
            lines.append(f"    posição CSV {i} → {self._col_label(i, column_names)}")
        return lines

    def format_position_to_circuit_map(
        self,
        column_names: list[str] | None,
        encoding_type: EncodingType,
    ) -> list[str]:
        """
        Após o plano: posição no vetor mapeado → coluna de origem → qubit(s) no circuito.
        """
        lines: list[str] = []
        enc = encoding_type

        if enc == EncodingType.DENSE_ANGLE:
            lines.append(
                "    (dense_angle: 2 features consecutivas por qubit — Ry na 1ª, Rz na 2ª)"
            )
            for pos, (col_idx, w) in enumerate(zip(self.column_indices, self.weights)):
                qubit = pos // 2
                role = "Ry (1ª do par)" if pos % 2 == 0 else "Rz (2ª do par)"
                wt = "" if abs(w - 1.0) < 1e-9 else f", peso×{w:g}"
                lines.append(
                    f"    posição {pos} → {self._col_label(col_idx, column_names)} "
                    f"(col. CSV {col_idx}) → qubit {qubit} ({role}{wt})"
                )
        elif enc in {
            EncodingType.ANGLE,
            EncodingType.IQP,
            EncodingType.DATA_REUPLOADING,
            EncodingType.CUSTOM_FEATURE_MAP,
            EncodingType.BASIS,
        }:
            hint = {
                EncodingType.ANGLE: "Ry(x) no qubit i",
                EncodingType.IQP: "Rz(x²) no qubit i; pares (i,j) geram Rzz(x_i·x_j)",
                EncodingType.DATA_REUPLOADING: "Ry(x) por camada no qubit i",
                EncodingType.CUSTOM_FEATURE_MAP: "Ry/Rz no qubit i + entrelaçamento",
                EncodingType.BASIS: "bit no qubit i",
            }.get(enc, "feature no qubit i")
            lines.append(f"    (1 feature por qubit — {hint})")
            for pos, (col_idx, w) in enumerate(zip(self.column_indices, self.weights)):
                wt = "" if abs(w - 1.0) < 1e-9 else f", peso×{w:g}"
                lines.append(
                    f"    posição {pos} → {self._col_label(col_idx, column_names)} "
                    f"(col. CSV {col_idx}) → qubit {pos}{wt}"
                )
        else:
            lines.append("    (amplitude: vetor completo após seleção/peso — sem mapeamento 1:1 por qubit)")
            for pos, (col_idx, w) in enumerate(zip(self.column_indices, self.weights)):
                wt = "" if abs(w - 1.0) < 1e-9 else f", peso×{w:g}"
                lines.append(
                    f"    posição {pos} → {self._col_label(col_idx, column_names)} "
                    f"(col. CSV {col_idx}){wt}"
                )
        return lines


@dataclass
class FeatureOptimizationResult:
    plan: FeatureEncodingPlan
    baseline_plan: FeatureEncodingPlan
    encoding_type: EncodingType
    baseline_kta: float
    optimized_kta: float
    search_log: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        return self.optimized_kta - self.baseline_kta

    @property
    def improvement_pct(self) -> float:
        if abs(self.baseline_kta) < 1e-12:
            return 0.0
        return 100.0 * self.improvement / abs(self.baseline_kta)


def _kta_for_plan(
    X: np.ndarray,
    labels: list[int],
    plan: FeatureEncodingPlan,
    encoding_type: EncodingType,
    *,
    n_qubits: int | None,
    max_samples: int,
) -> float | None:
    mapped = plan.apply(X)
    if mapped.shape[1] < 1:
        return None
    bounds = FeatureBounds.fit(mapped) if mapped.shape[0] >= 2 else None
    scores = compute_kta_by_encoding(
        mapped,
        labels,
        encoding_types=[encoding_type],
        n_qubits=n_qubits,
        max_samples=max_samples,
        feature_bounds=bounds,
    )
    return scores.get(encoding_type)


def _optimize_order(
    X: np.ndarray,
    labels: list[int],
    plan: FeatureEncodingPlan,
    encoding_type: EncodingType,
    *,
    n_qubits: int | None,
    max_samples: int,
    exhaustive_limit: int,
) -> tuple[FeatureEncodingPlan, float, list[str]]:
    log: list[str] = []
    n = len(plan.column_indices)
    if n <= 1:
        score = _kta_for_plan(X, labels, plan, encoding_type, n_qubits=n_qubits, max_samples=max_samples)
        return plan, score if score is not None else float("-inf"), log

    best_plan = plan.copy()
    best_score = _kta_for_plan(
        X, labels, best_plan, encoding_type, n_qubits=n_qubits, max_samples=max_samples
    )
    if best_score is None:
        return best_plan, float("-inf"), log

    if n <= exhaustive_limit:
        log.append(f"Ordem: busca exaustiva sobre {n}! permutações.")
        for perm in itertools.permutations(range(n)):
            trial = FeatureEncodingPlan(
                column_indices=[plan.column_indices[i] for i in perm],
                weights=[plan.weights[i] for i in perm],
            )
            score = _kta_for_plan(
                X, labels, trial, encoding_type, n_qubits=n_qubits, max_samples=max_samples
            )
            if score is not None and score > best_score:
                best_score = score
                best_plan = trial
    else:
        log.append(f"Ordem: hill-climbing por trocas (d={n} > {exhaustive_limit}).")
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    trial = best_plan.copy()
                    trial.column_indices[i], trial.column_indices[j] = (
                        trial.column_indices[j],
                        trial.column_indices[i],
                    )
                    trial.weights[i], trial.weights[j] = trial.weights[j], trial.weights[i]
                    score = _kta_for_plan(
                        X, labels, trial, encoding_type, n_qubits=n_qubits, max_samples=max_samples
                    )
                    if score is not None and score > best_score + 1e-6:
                        best_score = score
                        best_plan = trial
                        improved = True

    return best_plan, best_score, log


def _optimize_selection(
    X: np.ndarray,
    labels: list[int],
    plan: FeatureEncodingPlan,
    encoding_type: EncodingType,
    *,
    n_qubits: int | None,
    max_samples: int,
) -> tuple[FeatureEncodingPlan, float, list[str]]:
    log: list[str] = []
    best_plan = plan.copy()
    best_score = _kta_for_plan(
        X, labels, best_plan, encoding_type, n_qubits=n_qubits, max_samples=max_samples
    )
    if best_score is None:
        return best_plan, float("-inf"), log

    log.append("Seleção: eliminação backward de colunas.")
    while len(best_plan.column_indices) > 2:
        improved = False
        for drop in range(len(best_plan.column_indices)):
            trial = FeatureEncodingPlan(
                column_indices=[
                    c for k, c in enumerate(best_plan.column_indices) if k != drop
                ],
                weights=[w for k, w in enumerate(best_plan.weights) if k != drop],
            )
            score = _kta_for_plan(
                X, labels, trial, encoding_type, n_qubits=n_qubits, max_samples=max_samples
            )
            if score is not None and score > best_score + 1e-6:
                best_score = score
                best_plan = trial
                improved = True
                log.append(
                    f"  Removida posição {drop} (col original "
                    f"{plan.column_indices[drop] if drop < len(plan.column_indices) else '?'}), "
                    f"KTA={score:.4f}"
                )
                break
        if not improved:
            break

    return best_plan, best_score, log


def _optimize_weights(
    X: np.ndarray,
    labels: list[int],
    plan: FeatureEncodingPlan,
    encoding_type: EncodingType,
    *,
    n_qubits: int | None,
    max_samples: int,
) -> tuple[FeatureEncodingPlan, float, list[str]]:
    log: list[str] = ["Pesos: grid greedy por feature."]
    best_plan = plan.copy()
    best_score = _kta_for_plan(
        X, labels, best_plan, encoding_type, n_qubits=n_qubits, max_samples=max_samples
    )
    if best_score is None:
        return best_plan, float("-inf"), log

    for pos in range(len(best_plan.column_indices)):
        local_best_w = best_plan.weights[pos]
        local_best_score = best_score
        for w in WEIGHT_CANDIDATES:
            trial = best_plan.copy()
            trial.weights[pos] = w
            score = _kta_for_plan(
                X, labels, trial, encoding_type, n_qubits=n_qubits, max_samples=max_samples
            )
            if score is not None and score > local_best_score + 1e-6:
                local_best_score = score
                local_best_w = w
        if local_best_score > best_score + 1e-6:
            best_plan.weights[pos] = local_best_w
            best_score = local_best_score
            log.append(
                f"  Feature pos {pos}: peso {local_best_w:g}, KTA={best_score:.4f}"
            )

    return best_plan, best_score, log


def optimize_feature_encoding(
    X: np.ndarray,
    labels: list[int],
    encoding_type: EncodingType,
    *,
    n_qubits: int | None = None,
    max_samples: int = 25,
    exhaustive_order_limit: int = EXHAUSTIVE_ORDER_LIMIT,
    column_names: list[str] | None = None,
) -> FeatureOptimizationResult | None:
    """
    Otimiza ordem, seleção e pesos de features maximizando KTA para um encoding fixo.
    Retorna None se KTA baseline não puder ser calculado.
    """
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or len(labels) != x.shape[0]:
        return None
    if len(set(labels)) < 2:
        return None

    baseline_plan = FeatureEncodingPlan.identity(x.shape[1])
    baseline_kta = _kta_for_plan(
        x, labels, baseline_plan, encoding_type, n_qubits=n_qubits, max_samples=max_samples
    )
    if baseline_kta is None:
        return None

    search_log: list[str] = [
        f"Encoding alvo: {encoding_type.value}",
        f"KTA baseline (ordem CSV): {baseline_kta:.4f}",
    ]

    plan, score, log1 = _optimize_order(
        x,
        labels,
        baseline_plan,
        encoding_type,
        n_qubits=n_qubits,
        max_samples=max_samples,
        exhaustive_limit=exhaustive_order_limit,
    )
    search_log.extend(log1)
    search_log.append(f"Após ordem: KTA={score:.4f}")

    plan, score, log2 = _optimize_selection(
        x, labels, plan, encoding_type, n_qubits=n_qubits, max_samples=max_samples
    )
    search_log.extend(log2)
    search_log.append(f"Após seleção: KTA={score:.4f}")

    plan, score, log3 = _optimize_weights(
        x, labels, plan, encoding_type, n_qubits=n_qubits, max_samples=max_samples
    )
    search_log.extend(log3)
    search_log.append(f"Após pesos: KTA={score:.4f}")

    if column_names:
        search_log.append(f"Mapeamento: {plan.describe(column_names)}")

    return FeatureOptimizationResult(
        plan=plan,
        baseline_plan=baseline_plan,
        encoding_type=encoding_type,
        baseline_kta=baseline_kta,
        optimized_kta=score,
        search_log=search_log,
    )


def pick_encoding_for_optimization(
    X: np.ndarray,
    labels: list[int],
    *,
    preferred: EncodingType | None = None,
    n_qubits: int | None = None,
    max_samples: int = 25,
) -> EncodingType:
    """Escolhe encoding para otimizar: preferido se válido, senão melhor KTA baseline."""
    if preferred is not None:
        return preferred
    scores = compute_kta_by_encoding(
        X, labels, encoding_types=list(EncodingType), n_qubits=n_qubits, max_samples=max_samples
    )
    if scores:
        return max(scores.items(), key=lambda kv: kv[1])[0]
    return EncodingType.IQP


def _presentation_summary(
    result: FeatureOptimizationResult,
    column_names: list[str] | None,
    n_original_features: int,
) -> str:
    """Frase única para apresentação oral / slides."""
    enc = result.encoding_type.value
    delta = result.improvement
    pct = result.improvement_pct
    dropped = result.plan.dropped_columns(n_original_features)
    drop_part = ""
    if dropped and column_names:
        names = [result.plan._col_label(i, column_names) for i in dropped]
        drop_part = f" Removemos {', '.join(names)}."
    elif dropped:
        drop_part = f" Removemos colunas CSV {dropped}."
    return (
        f"Reordenamos as colunas do CSV antes do encoding {enc}: KTA subiu "
        f"{result.baseline_kta:.3f} → {result.optimized_kta:.3f} (Δ={delta:+.3f}, {pct:+.0f}%)."
        f"{drop_part} A ordem no CSV não muda os dados clássicos, mas muda qual feature "
        f"ocupa cada qubit — e portanto o kernel quântico K(x,x')."
    )


def format_feature_optimization_section(
    result: FeatureOptimizationResult,
    column_names: list[str] | None = None,
    *,
    n_original_features: int | None = None,
) -> list[str]:
    n_orig = n_original_features or (
        len(column_names) if column_names else max(result.baseline_plan.column_indices, default=-1) + 1
    )
    dropped = result.plan.dropped_columns(n_orig)
    weighted = [
        (result.plan._col_label(c, column_names), w)
        for c, w in zip(result.plan.column_indices, result.plan.weights)
        if abs(w - 1.0) >= 1e-9
    ]

    lines = [
        "=== Otimização de mapeamento de features (ordem / seleção / peso) ===",
        "Referência: Fioravanti et al., Quantum feature encoding optimization (arXiv:2512.02422).",
        "",
        "  O que isto significa (para apresentar):",
        "  • Cada coluna numérica do CSV vira uma feature no circuito.",
        "  • A posição da coluna define qual qubit (ou par, no dense_angle) recebe aquele valor.",
        "  • Permutar colunas = remapear features → qubits = kernel K(x,x') diferente.",
        "  • KTA mede se esse kernel separa as classes; a busca maximiza KTA antes do treino QSVM.",
        "",
        f"  Encoding otimizado: {result.encoding_type.value}",
        f"  KTA baseline (ordem do CSV): {result.baseline_kta:.4f}",
        f"  KTA otimizado: {result.optimized_kta:.4f} "
        f"(Δ={result.improvement:+.4f}, {result.improvement_pct:+.1f}%)",
        f"  Sequência otimizada: {result.plan.describe(column_names)}",
        "",
        "  Ordem ORIGINAL no CSV (como veio no arquivo):",
    ]
    lines.extend(result.baseline_plan.format_csv_baseline_order(column_names))
    lines.extend([
        "",
        "  Mapeamento OTIMIZADO (posição no vetor → coluna de origem → qubit no circuito):",
    ])
    lines.extend(
        result.plan.format_position_to_circuit_map(column_names, result.encoding_type)
    )

    if dropped:
        lines.append("")
        lines.append("  Colunas do CSV NÃO usadas após seleção:")
        for col_idx in dropped:
            lines.append(f"    • {result.plan._col_label(col_idx, column_names)} (col. CSV {col_idx})")

    if weighted:
        lines.append("")
        lines.append("  Pesos aplicados antes do scaling [0, π]:")
        for name, w in weighted:
            lines.append(f"    • {name} × {w:g}")

    lines.extend([
        "",
        "  Em uma frase:",
        f"    {_presentation_summary(result, column_names, n_orig)}",
        "",
        "  Log da busca (ordem → seleção → peso):",
    ])
    for entry in result.search_log:
        lines.append(f"    {entry}")
    lines.append("")
    return lines
