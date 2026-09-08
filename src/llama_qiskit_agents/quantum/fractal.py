"""Dimensão fractal de correlação (D2) e seleção de atributos FD-ASE.

Estimador LiBOC: grade multi-resolução esparsa e inclinação de log S(r) vs log r.
FD-ASE escolhe colunas originais cuja dimensão fractal parcial recupera D2
(não é PCA: devolve índices, não misturas).

Referências: Sousa et al. (2002); SAC/PKDD 2007; Appel, arXiv:2609.00475.

Este módulo não inclui o sampler BBS (amostragem de linhas).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil

import numpy as np


MIN_SAMPLES_FOR_D2 = 32
MIN_FEATURES_FOR_FDASE = 3
QUBIT_BUDGET_FLOOR = 2


def normalize_unit_cube(X: np.ndarray) -> np.ndarray:
    """Normaliza cada atributo para o cubo unitário [0, 1)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X deve ter forma (n_amostras, n_atributos).")
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.where(hi - lo < 1e-15, 1.0, hi - lo)
    scaled = (X - lo) / span
    return np.clip(scaled, 0.0, 1.0 - 1e-15)


def child_code(x: np.ndarray, level: int) -> int:
    """Código do octante relativo ao pai no nível ``level``."""
    bins = np.minimum(np.floor(x * (1 << level)).astype(np.int64), (1 << level) - 1)
    bits = bins & 1
    code = 0
    for d, bit in enumerate(bits):
        if bit:
            code |= 1 << d
    return int(code)


@dataclass
class GridNode:
    """Nó da MG-tree: ocupância e filhos esparsos."""

    level: int
    occupancy: int = 0
    children: dict[int, GridNode] = field(default_factory=dict)


class MultiResolutionGrid:
    """Árvore de grade multi-resolução usada pelo LiBOC."""

    def __init__(self, n_levels: int = 5) -> None:
        if n_levels < 2:
            raise ValueError("n_levels deve ser pelo menos 2.")
        self.n_levels = n_levels
        self.root = GridNode(level=0)
        self.n_points = 0
        self.n_features = 0

    def insert_all(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        n, e = X.shape
        self.n_points = n
        self.n_features = e
        self.root.occupancy = n
        for i in range(n):
            node = self.root
            point = X[i]
            for level in range(1, self.n_levels):
                code = child_code(point, level)
                child = node.children.get(code)
                if child is None:
                    child = GridNode(level=level)
                    node.children[code] = child
                child.occupancy += 1
                node = child

    def occupancy_squares_by_level(self) -> list[float]:
        """S(r) = soma dos C_i^2 em cada nível 0 .. n_levels-1."""
        totals = [0.0] * self.n_levels

        def walk(node: GridNode) -> None:
            totals[node.level] += float(node.occupancy) ** 2
            for child in node.children.values():
                walk(child)

        walk(self.root)
        return totals


def correlation_fractal_dimension(
    X: np.ndarray,
    n_levels: int = 10,
    min_level: int = 0,
    occupancy_floor: float = 1.5,
) -> float:
    """Estima D2 como a inclinação de log S(r) versus log r (box-count LiBOC)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] < 2 or X.shape[1] == 0:
        return 0.0

    grid = MultiResolutionGrid(n_levels=n_levels)
    grid.insert_all(normalize_unit_cube(X))
    s_values = grid.occupancy_squares_by_level()
    n = float(X.shape[0])
    floor = occupancy_floor * n

    log_r: list[float] = []
    log_s: list[float] = []
    last = min(n_levels, len(s_values))
    for level in range(min_level, last):
        s = s_values[level]
        if s <= 0:
            continue
        if level > 0 and s <= floor:
            continue
        r = 0.5**level
        log_r.append(float(np.log(r)) if r > 0 else 0.0)
        log_s.append(float(np.log(s)))

    if len(log_r) < 2:
        log_r = []
        log_s = []
        for level in range(min_level, last):
            s = s_values[level]
            if s <= 0:
                continue
            r = 0.5**level
            log_r.append(float(np.log(r)) if r > 0 else 0.0)
            log_s.append(float(np.log(s)))
    if len(log_r) < 2:
        return 0.0
    slope = float(np.polyfit(np.asarray(log_r), np.asarray(log_s), 1)[0])
    return float(np.clip(slope, 0.0, float(X.shape[1])))


class FDASE:
    """Seleção de atributos pela dimensão fractal de correlação (forward greedy)."""

    def __init__(self, eps: float = 0.15, n_levels: int = 8) -> None:
        self.eps = float(eps)
        self.n_levels = int(n_levels)
        self.embedding_dimension_: int | None = None
        self.intrinsic_dimension_: float | None = None
        self.selected_: np.ndarray | None = None
        self.correlation_groups_: list[list[int]] | None = None
        self.correlation_bases_: list[list[int]] | None = None

    def _pD(self, X: np.ndarray, cols: Sequence[int]) -> float:
        if not cols:
            return 0.0
        return correlation_fractal_dimension(X[:, list(cols)], n_levels=self.n_levels)

    def _drop_constants(self, X: np.ndarray, attrs: list[int]) -> list[int]:
        kept: list[int] = []
        for a in attrs:
            if self._pD(X, [a]) > self.eps:
                kept.append(a)
        return kept

    def _correlation_base(self, X: np.ndarray, group: list[int]) -> list[int]:
        base = list(group)
        if len(base) <= 1:
            return base
        d_group = self._pD(X, base)
        changed = True
        while changed and len(base) > 1:
            changed = False
            for attr in list(base):
                trial = [x for x in base if x != attr]
                if abs(self._pD(X, trial) - d_group) <= self.eps:
                    base = trial
                    changed = True
                    break
        return base

    def _fit_greedy(self, X: np.ndarray, remaining: list[int], d_full: float) -> None:
        selected: list[int] = []
        groups: list[list[int]] = []
        bases: list[list[int]] = []
        pD = 0.0
        leftover = list(remaining)
        while leftover and pD < d_full - self.eps:
            best_attr = leftover[0]
            best_pD = -1.0
            for attr in leftover:
                trial_pD = self._pD(X, selected + [attr])
                if trial_pD > best_pD + 1e-12:
                    best_pD = trial_pD
                    best_attr = attr
            gain = best_pD - pD
            if gain <= self.eps:
                correlated = [best_attr]
                leftover.remove(best_attr)
                still = list(leftover)
                for attr in still:
                    extra = self._pD(X, selected + correlated + [attr])
                    if extra - best_pD <= self.eps:
                        correlated.append(attr)
                        leftover.remove(attr)
                group = selected[-1:] + correlated if selected else correlated
                groups.append(group)
                base = self._correlation_base(X, group)
                bases.append(base)
                for attr in group:
                    if attr not in selected:
                        selected.append(attr)
                pD = self._pD(X, selected)
                continue
            leftover.remove(best_attr)
            selected.append(best_attr)
            pD = best_pD
        if not groups and selected:
            groups = [list(selected)]
            bases = [self._correlation_base(X, selected)]
            selected = list(dict.fromkeys(a for b in bases for a in b))
        else:
            selected = list(dict.fromkeys(a for b in bases for a in b)) if bases else selected
        self.selected_ = np.asarray(selected, dtype=np.int64)
        self.correlation_groups_ = groups
        self.correlation_bases_ = bases

    def fit(self, X: np.ndarray) -> FDASE:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X deve ter forma (n_amostras, n_atributos).")
        e = X.shape[1]
        self.embedding_dimension_ = e
        d_full = correlation_fractal_dimension(X, n_levels=self.n_levels)
        self.intrinsic_dimension_ = d_full
        remaining = self._drop_constants(X, list(range(e)))
        if not remaining:
            self.selected_ = np.zeros(0, dtype=np.int64)
            self.correlation_groups_ = []
            self.correlation_bases_ = []
            return self
        self._fit_greedy(X, remaining, d_full)
        if self.selected_ is None or self.selected_.size == 0:
            self.selected_ = np.asarray(remaining[: max(1, int(np.ceil(d_full)))], dtype=np.int64)
        return self


@dataclass
class FractalBudget:
    """Orçamento de qubits e recorte FD-ASE. Campos None quando D2 não é estimável."""

    intrinsic_dimension: float | None = None
    embedding_dimension: int | None = None
    qubit_budget: int | None = None
    selected_columns: list[int] | None = None
    selected_feature_names: list[str] | None = None


def qubit_budget_from_d2(d2: float) -> int:
    return max(QUBIT_BUDGET_FLOOR, int(ceil(float(d2))))


def estimate_fractal_budget(
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> FractalBudget:
    """Estima D2, q* = max(2, ceil(D2)) e colunas FD-ASE truncadas a q*.

    Não roda se X não for 2D, n_samples < 32 ou n_features <= 2.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < MIN_SAMPLES_FOR_D2 or x.shape[1] < MIN_FEATURES_FOR_FDASE:
        return FractalBudget()

    embedding = int(x.shape[1])
    selector = FDASE()
    selector.fit(x)
    d2 = float(selector.intrinsic_dimension_ or 0.0)
    q_star = qubit_budget_from_d2(d2)
    selected = [int(i) for i in (selector.selected_ if selector.selected_ is not None else [])]
    if not selected:
        variances = np.var(x, axis=0)
        selected = [i for i in range(embedding) if variances[i] > 1e-15][:q_star]
    if len(selected) > q_star:
        selected = selected[:q_star]
    if not selected:
        return FractalBudget(
            intrinsic_dimension=round(d2, 4),
            embedding_dimension=embedding,
            qubit_budget=q_star,
            selected_columns=None,
            selected_feature_names=None,
        )

    names: list[str] | None = None
    if feature_names is not None and len(feature_names) >= embedding:
        names = [feature_names[i] for i in selected]

    return FractalBudget(
        intrinsic_dimension=round(d2, 4),
        embedding_dimension=embedding,
        qubit_budget=q_star,
        selected_columns=selected,
        selected_feature_names=names,
    )


def apply_column_selection(X: np.ndarray, columns: list[int] | None) -> np.ndarray:
    x = np.asarray(X, dtype=float)
    if columns is None or x.ndim != 2 or not columns:
        return x
    valid = [c for c in columns if 0 <= c < x.shape[1]]
    if not valid:
        return x
    return x[:, valid]


def apply_column_selection_row(row: np.ndarray, columns: list[int] | None) -> np.ndarray:
    arr = np.asarray(row, dtype=float).flatten()
    if columns is None or not columns:
        return arr
    valid = [c for c in columns if 0 <= c < arr.size]
    if not valid:
        return arr
    return arr[valid]
