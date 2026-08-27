"""Smoke tests do benchmark suite (requer scikit-learn)."""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")

from benchmarks.datasets import load_iris_binary, load_synthetic_order_sensitive  # noqa: E402
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
    assert result.n_classes == 2
    assert -1.0 <= result.baseline_best_kta <= 1.0
    assert result.baseline_best_encoding
