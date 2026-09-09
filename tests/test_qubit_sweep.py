"""Sweep q: FD-ASE vs PCA vs prefixo no probe de angle encoding."""

from __future__ import annotations

import numpy as np

from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.fractal import estimate_fractal_budget
from llama_qiskit_agents.quantum.kernel import KernelGeometry
from llama_qiskit_agents.quantum.qubit_sweep import (
    QubitSweepResult,
    SweepPoint,
    format_qubit_sweep_section,
    last_alive_q,
    pca_variance_qubits,
    project_view,
    run_qubit_budget_sweep,
)
from llama_qiskit_agents.quantum.simulate import (
    compare_embeddings,
    format_comparison_report,
)


def _rank2_plus_constants(n: int = 80, n_const: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n, 2))
    const = np.full((n, n_const), 3.14)
    return np.hstack([signal, const])


def test_pca95_on_rank2_plus_tiny_noise() -> None:
    x = _rank2_plus_constants()
    q = pca_variance_qubits(x)
    assert 2 <= q <= 4
    assert q < x.shape[1]


def test_prefix_and_pca_projection_shapes() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 5))
    xp, cols = project_view(x, 3, "prefix")
    assert xp.shape == (20, 3)
    assert cols == [0, 1, 2]
    xpca, pca_cols = project_view(x, 3, "pca")
    assert xpca.shape == (20, 3)
    assert pca_cols is None
    fdase, fd_cols = project_view(x, 2, "fdase", fdase_columns=[4, 1, 0])
    assert fdase.shape == (20, 2)
    assert fd_cols == [4, 1]


def test_last_alive_q_and_report_markers() -> None:
    geo_alive = KernelGeometry(0.5, 0.1, 5.0, 0.2, True)
    geo_dead = KernelGeometry(0.1, 0.1, 1.0, 0.01, False)
    pts = [
        SweepPoint("fdase", 2, geo_alive, last_alive=True),
        SweepPoint("pca", 2, geo_alive),
        SweepPoint("pca", 3, geo_alive, last_alive=True),
        SweepPoint("pca", 10, geo_dead),
        SweepPoint("prefix", 2, geo_dead),
    ]
    assert last_alive_q(pts, "pca") == 3
    assert last_alive_q(pts, "prefix") is None
    text = "\n".join(
        format_qubit_sweep_section(
            QubitSweepResult(
                d2=2.4,
                q_star=3,
                pca95=10,
                e_original=30,
                n_kernel=16,
                encoding=EncodingType.ANGLE,
                points=pts,
            )
        )
    )
    assert "Sweep q" in text
    assert "PCA-95%" in text
    assert "q*=3" in text
    assert "arXiv:2609.00475" in text
    assert "ALIVE*" in text


def test_small_angle_sweep_includes_pca_and_prefix() -> None:
    x = _rank2_plus_constants(n=40, n_const=3)
    budget = estimate_fractal_budget(x)
    sweep = run_qubit_budget_sweep(
        x,
        fdase_columns=budget.selected_columns,
        q_star=budget.qubit_budget,
        d2=budget.intrinsic_dimension,
        q_max=3,
        max_samples=8,
    )
    assert sweep is not None
    views = {p.view for p in sweep.points}
    assert "pca" in views
    assert "prefix" in views
    assert all(p.q <= max(3, sweep.pca95) for p in sweep.points)
    assert sweep.pca95 < x.shape[1]


def test_compare_report_includes_sweep_when_d2_exists() -> None:
    x = _rank2_plus_constants(n=40, n_const=3)
    cr = compare_embeddings(
        x,
        encoding_types=[EncodingType.ANGLE],
        shots=8,
        apply_fractal_budget=True,
        sweep_qubit_budget=True,
        sweep_q_max=3,
        sweep_n_samples=8,
    )
    assert cr.profile.intrinsic_dimension is not None
    assert cr.qubit_sweep is not None
    assert cr.qubit_sweep.points
    text = format_comparison_report(cr)
    assert "Sweep q" in text
    assert "PCA-95%" in text
    assert "fdase" in text or "pca" in text


def test_compare_can_disable_sweep() -> None:
    x = _rank2_plus_constants(n=40, n_const=3)
    cr = compare_embeddings(
        x,
        encoding_types=[EncodingType.ANGLE],
        shots=8,
        sweep_qubit_budget=False,
    )
    assert cr.qubit_sweep is None
    text = format_comparison_report(cr)
    assert "Sweep q" not in text
