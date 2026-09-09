"""Smoke tests do benchmark suite (requer scikit-learn)."""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")

from benchmarks.datasets import (  # noqa: E402
    load_breast_cancer_fdase,
    load_breast_cancer_top6,
    load_iris_binary,
    load_synthetic_order_sensitive,
)
from benchmarks.runner import run_benchmark_case  # noqa: E402


@pytest.mark.parametrize(
    "loader",
    [load_iris_binary, load_synthetic_order_sensitive],
)
def test_benchmark_case_smoke(loader) -> None:
    ds = loader(max_samples=12) if loader is load_iris_binary else loader(n_per_class=6)
    result = run_benchmark_case(
        ds,
        max_kta_samples=min(12, ds.X.shape[0]),
        shots=256,
        run_compare=False,
    )
    assert result.n_samples >= 2
    assert result.n_features >= 2
    assert result.n_features_eval == result.n_features
    assert result.n_classes == 2
    assert -1.0 <= result.baseline_best_kta <= 1.0
    assert result.baseline_best_encoding
    assert result.baseline_kernel_alive is not None


def test_breast_kta_uses_fdase_crop_not_top6() -> None:
    ds = load_breast_cancer_fdase(max_samples=32)
    top6 = load_breast_cancer_top6(max_samples=32)
    assert ds.X.shape[1] == 30
    assert top6.X.shape[1] == 6
    result = run_benchmark_case(
        ds,
        max_kta_samples=8,
        shots=8,
        run_compare=False,
        apply_fractal_budget=True,
    )
    assert result.n_features == 30
    assert result.d2 is not None
    assert result.qubit_budget is not None
    assert result.selected_columns
    assert result.n_features_eval == len(result.selected_columns)
    assert result.n_features_eval <= result.qubit_budget
    assert result.n_features_eval < 30
    assert result.baseline_kernel_alive is not None
