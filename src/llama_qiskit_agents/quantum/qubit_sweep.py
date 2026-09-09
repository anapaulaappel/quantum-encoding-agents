"""Sweep de largura q: FD-ASE vs PCA vs prefixo, com kernel-alive e KTA.

O preprint (Appel, arXiv:2609.00475) compara vistas quânticas na mesma K de
fidelidade. PCA-95% não é substituto de D2: pode pedir mais qubits do que o
kernel angle-encoded aguenta. Prefixo usa as primeiras colunas originais.

Numpy only (SVD); não usa sklearn nem o pacote bbs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.kernel import (
    MAX_QUBITS_FOR_KERNEL_DIAGNOSTICS,
    KernelGeometry,
    compute_kernel_matrix,
    kernel_stats,
    score_kernel_geometry,
)
from llama_qiskit_agents.quantum.preprocessing import FeatureBounds


SWEEP_VIEWS = ("fdase", "pca", "prefix")
PCA_VARIANCE_THRESHOLD = 0.95
DEFAULT_SWEEP_Q_MAX = 8
DEFAULT_SWEEP_N_SAMPLES = 16
MIN_SAMPLES_FOR_SWEEP = 8
MIN_FEATURES_FOR_SWEEP = 3


@dataclass
class SweepPoint:
    """Um ponto (vista, q) na curva de largura."""

    view: str
    q: int
    geometry: KernelGeometry
    kta: float | None = None
    last_alive: bool = False
    columns: list[int] | None = None


@dataclass
class QubitSweepResult:
    """Curva q das três vistas + marcadores q* e PCA-95%."""

    d2: float | None
    q_star: int | None
    pca95: int
    e_original: int
    n_kernel: int
    encoding: EncodingType
    points: list[SweepPoint]


def standardize_columns(X: np.ndarray) -> np.ndarray:
    """StandardScaler numpy (média 0, desvio 1, ddof=0)."""
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("X deve ter forma (n_amostras, n_atributos).")
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-15, 1.0, sd)
    return (x - mu) / sd


def pca_variance_qubits(X: np.ndarray, threshold: float = PCA_VARIANCE_THRESHOLD) -> int:
    """Menor q cuja PCA, após standardização, acumula `threshold` da variância."""
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        return 1
    n, e = x.shape
    if n < 2 or e == 0:
        return 1
    xs = standardize_columns(x)
    k = max(1, min(n - 1, e))
    _u, s, _vt = np.linalg.svd(xs, full_matrices=False)
    var = s[:k] ** 2
    total = float(var.sum())
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(var) / total
    q = int(np.searchsorted(cumulative, threshold) + 1)
    return max(1, min(q, e))


def pca_project(X: np.ndarray, q: int) -> np.ndarray:
    """Projeta X em q componentes principais (dados padronizados)."""
    xs = standardize_columns(X)
    n, e = xs.shape
    q_use = max(1, min(int(q), e, max(1, n - 1)))
    u, s, _vt = np.linalg.svd(xs, full_matrices=False)
    return u[:, :q_use] * s[:q_use]


def project_view(
    X: np.ndarray,
    q: int,
    view: str,
    fdase_columns: list[int] | None = None,
) -> tuple[np.ndarray, list[int] | None]:
    """Devolve (Xq, índices originais ou None se PCA)."""
    x = np.asarray(X, dtype=np.float64)
    n, e = x.shape
    q_use = max(1, min(int(q), e, max(1, n - 1)))
    if view == "prefix":
        return x[:, :q_use], list(range(q_use))
    if view == "pca":
        return pca_project(x, q_use), None
    if view == "fdase":
        cols = [c for c in (fdase_columns or []) if 0 <= c < e][:q_use]
        if len(cols) < 2:
            raise ValueError("FD-ASE precisa de pelo menos 2 colunas.")
        return x[:, cols], cols
    raise ValueError(f"view deve ser um de {SWEEP_VIEWS}, recebido {view!r}.")


def last_alive_q(points: list[SweepPoint], view: str) -> int | None:
    alive = [p.q for p in points if p.view == view and p.geometry.kernel_alive]
    return max(alive) if alive else None


def q_grid_for_sweep(
    *,
    e: int,
    n_sub: int,
    pca95: int,
    q_max: int = DEFAULT_SWEEP_Q_MAX,
) -> list[int]:
    cap = max(2, min(int(q_max), e, n_sub - 1, MAX_QUBITS_FOR_KERNEL_DIAGNOSTICS))
    grid = list(range(2, cap + 1))
    extra = min(int(pca95), e, n_sub - 1, MAX_QUBITS_FOR_KERNEL_DIAGNOSTICS)
    if extra >= 2 and extra not in grid:
        grid.append(extra)
        grid.sort()
    return grid


def run_qubit_budget_sweep(
    X: np.ndarray,
    *,
    labels: list[int] | None = None,
    fdase_columns: list[int] | None = None,
    q_star: int | None = None,
    d2: float | None = None,
    encoding_type: EncodingType = EncodingType.ANGLE,
    q_max: int = DEFAULT_SWEEP_Q_MAX,
    max_samples: int = DEFAULT_SWEEP_N_SAMPLES,
    random_state: int = 42,
) -> QubitSweepResult | None:
    """
    Avalia K de fidelidade (probe = angle) em q = 2..q_max para FD-ASE, PCA e prefixo.
    Inclui o q de PCA-95% se couber no teto de qubits do compare.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        return None
    n, e = x.shape
    if n < MIN_SAMPLES_FOR_SWEEP or e < MIN_FEATURES_FOR_SWEEP:
        return None

    pca95 = pca_variance_qubits(x)
    take = min(int(max_samples), n)
    rng = np.random.default_rng(random_state)
    idx = np.arange(n) if take == n else rng.choice(n, size=take, replace=False)
    x_sub = x[idx]
    labels_sub: list[int] | None = None
    if labels is not None and len(labels) == n and len(set(labels)) >= 2:
        labels_sub = [labels[int(i)] for i in idx]

    grid = q_grid_for_sweep(e=e, n_sub=int(x_sub.shape[0]), pca95=pca95, q_max=q_max)
    fdase_ok = [c for c in (fdase_columns or []) if 0 <= c < e]
    points: list[SweepPoint] = []

    for view in SWEEP_VIEWS:
        qs = list(grid)
        if view == "fdase":
            if len(fdase_ok) < 2:
                continue
            qs = [q for q in grid if q <= len(fdase_ok)]
        for q in qs:
            try:
                xq, cols = project_view(x_sub, q, view, fdase_columns=fdase_ok)
                q_eff = int(xq.shape[1])
                if q_eff < 2 or q_eff > MAX_QUBITS_FOR_KERNEL_DIAGNOSTICS:
                    continue
                bounds = FeatureBounds.fit(xq)
                kernel = compute_kernel_matrix(
                    xq,
                    encoding_type,
                    n_qubits=q_eff,
                    feature_bounds=bounds,
                )
                geo = score_kernel_geometry(xq, kernel)
                stats = kernel_stats(kernel, labels_sub)
                points.append(
                    SweepPoint(
                        view=view,
                        q=q_eff,
                        geometry=geo,
                        kta=stats.get("kta"),
                        columns=cols,
                    )
                )
            except Exception:
                continue

    if not points:
        return None

    last_q: dict[str, int] = {}
    for p in points:
        if p.geometry.kernel_alive:
            last_q[p.view] = max(last_q.get(p.view, 0), p.q)
    marked = [
        replace(p, last_alive=last_q.get(p.view) is not None and p.q == last_q[p.view])
        for p in points
    ]
    return QubitSweepResult(
        d2=d2,
        q_star=q_star,
        pca95=pca95,
        e_original=e,
        n_kernel=int(x_sub.shape[0]),
        encoding=encoding_type,
        points=marked,
    )


def format_qubit_sweep_section(sweep: QubitSweepResult | None) -> list[str]:
    """Bloco de relatório: curva q das três vistas."""
    if sweep is None or not sweep.points:
        return []
    d2_txt = "—" if sweep.d2 is None else f"{sweep.d2:.3f}"
    qstar_txt = "—" if sweep.q_star is None else str(sweep.q_star)
    lines = [
        "=== Sweep q (FD-ASE vs PCA vs prefixo, arXiv:2609.00475) ===",
        f"  Probe: {sweep.encoding.value} encoding. Subamostra n={sweep.n_kernel}. "
        f"E={sweep.e_original}.",
        f"  D2≈{d2_txt} → q*={qstar_txt}.  PCA-95% pediria q={sweep.pca95}.",
        "  Independente do recorte aplicado ao ranking; não mistura eixos no FD-ASE/prefixo.",
        f"  {'view':<8} {'q':>3}  {'status':<7} {'near':>6} {'far':>6} {'ratio':>6} {'meanK':>6} {'KTA':>7}",
    ]
    for p in sweep.points:
        status = "ALIVE" if p.geometry.kernel_alive else "DEAD"
        if p.last_alive:
            status = "ALIVE*"
        kta_txt = "   —  " if p.kta is None else f"{p.kta:7.4f}"
        mark_bits: list[str] = []
        if sweep.q_star is not None and p.q == sweep.q_star:
            mark_bits.append("q*")
        if p.q == sweep.pca95:
            mark_bits.append("PCA-95%")
        mark = f"  ← {', '.join(mark_bits)}" if mark_bits else ""
        lines.append(
            f"  {p.view:<8} {p.q:3d}  {status:<7} "
            f"{p.geometry.fid_near:6.3f} {p.geometry.fid_far:6.3f} "
            f"{p.geometry.near_far_ratio:6.2f} {p.geometry.mean_offdiag:6.3f} "
            f"{kta_txt}{mark}"
        )

    knees = []
    for view in SWEEP_VIEWS:
        knee = last_alive_q(sweep.points, view)
        knees.append(f"{view}={knee if knee is not None else 'nenhum'}")
    lines.append(f"  Último q vivo (ALIVE*): {', '.join(knees)}.")

    pca_knee = last_alive_q(sweep.points, "pca")
    fdase_knee = last_alive_q(sweep.points, "fdase")
    if (
        sweep.q_star is not None
        and pca_knee is not None
        and sweep.pca95 > sweep.q_star
        and pca_knee < sweep.pca95
    ):
        lines.append(
            f"  Interpretação: PCA-95% pede {sweep.pca95} qubits; o kernel de fidelidade "
            f"já morreu em q={pca_knee}. q*={sweep.q_star} é o teto a priori."
        )
    elif fdase_knee is not None and sweep.q_star is not None and fdase_knee >= sweep.q_star:
        lines.append(
            f"  Interpretação: FD-ASE em q*={sweep.q_star} mantém o kernel vivo "
            "(não use a largura PCA-95% como orçamento de qubits)."
        )
    else:
        lines.append(
            "  Interpretação: compare o último q vivo de cada vista com q* e com PCA-95%."
        )
    lines.append("")
    return lines
