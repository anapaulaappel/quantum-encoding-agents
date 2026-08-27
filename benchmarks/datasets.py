"""
Conjuntos de dados públicos para benchmark de encoding / KTA / otimização de features.

Requer ``scikit-learn`` (extra ``benchmark`` no pyproject.toml).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
except ImportError:  # pragma: no cover - optional dep
    load_breast_cancer = None
    load_iris = None
    load_wine = None


@dataclass(frozen=True)
class BenchmarkDataset:
    """Dataset tabular pronto para compare_embeddings / optimize_feature_encoding."""

    slug: str
    name: str
    X: np.ndarray
    y: list[int]
    feature_names: list[str]
    description: str
    tags: tuple[str, ...] = field(default_factory=tuple)


def _require_sklearn() -> None:
    if load_iris is None:
        raise ImportError(
            "scikit-learn é necessário para benchmarks. "
            "Instale com: pip install -e '.[benchmark]'"
        )


def _top_k_by_variance(
    X: np.ndarray,
    feature_names: list[str],
    k: int,
) -> tuple[np.ndarray, list[str]]:
    if X.shape[1] <= k:
        return X, feature_names
    var = X.var(axis=0)
    idx = np.argsort(var)[-k:][::-1]
    return X[:, idx], [feature_names[i] for i in idx]


def _binary_subset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_a: int,
    class_b: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    mask = (y == class_a) | (y == class_b)
    y_bin = np.where(y[mask] == class_a, 0, 1).astype(int)
    return X[mask], y_bin.tolist(), feature_names


def _subsample(
    X: np.ndarray,
    y: list[int],
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    n = X.shape[0]
    if n <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    idx.sort()
    return X[idx], [y[int(i)] for i in idx]


def load_iris_binary(*, max_samples: int = 50, seed: int = 42) -> BenchmarkDataset:
    """Iris 2 classes (setosa vs versicolor), 4 features — benchmark rápido."""
    _require_sklearn()
    data = load_iris()
    X, y, names = _binary_subset(
        data.data,
        data.target,
        list(data.feature_names),
        class_a=0,
        class_b=1,
    )
    X, y = _subsample(X, y, max_samples, seed)
    return BenchmarkDataset(
        slug="iris_binary",
        name="Iris (setosa vs versicolor)",
        X=X,
        y=y,
        feature_names=names,
        description="Clássico UCI; 4 features contínuas, 2 classes balanceadas.",
        tags=("public", "small", "binary"),
    )


def load_breast_cancer_top6(
    *,
    max_samples: int = 40,
    max_features: int = 6,
    seed: int = 42,
) -> BenchmarkDataset:
    """Wisconsin Breast Cancer — top-6 features por variância, amostragem estratificada."""
    _require_sklearn()
    data = load_breast_cancer()
    X = np.asarray(data.data, dtype=float)
    y = data.target.astype(int).tolist()
    names = list(data.feature_names)
    X, names = _top_k_by_variance(X, names, max_features)
    X, y = _subsample(X, y, max_samples, seed)
    return BenchmarkDataset(
        slug="breast_cancer_top6",
        name="Breast Cancer Wisconsin (top-6 var)",
        X=X,
        y=y,
        feature_names=names,
        description=f"UCI sklearn; {max_features} features de maior variância, binário maligno/benigno.",
        tags=("public", "medium", "binary"),
    )


def load_wine_binary(*, max_samples: int = 40, seed: int = 42) -> BenchmarkDataset:
    """Wine 2 classes (0 vs 1), 13 features — médio."""
    _require_sklearn()
    data = load_wine()
    X, y, names = _binary_subset(
        data.data,
        data.target,
        [f"f{i}" for i in range(data.data.shape[1])],
        class_a=0,
        class_b=1,
    )
    X, y = _subsample(X, y, max_samples, seed)
    return BenchmarkDataset(
        slug="wine_binary",
        name="Wine (classes 0 vs 1)",
        X=X,
        y=y,
        feature_names=names,
        description="UCI Wine; 13 features químicas, 2 classes.",
        tags=("public", "medium", "binary"),
    )


def load_synthetic_order_sensitive(*, n_per_class: int = 12, seed: int = 42) -> BenchmarkDataset:
    """
    Sintético: colunas de ruído antes do sinal — ordem de mapeamento importa para IQP.
    """
    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    for label, (a_lo, a_hi, b_lo, b_hi) in {
        0: (0.65, 1.0, 0.65, 1.0),
        1: (0.0, 0.35, 0.0, 0.35),
    }.items():
        for _ in range(n_per_class):
            rows.append([
                float(rng.uniform(0.3, 0.7)),
                float(rng.uniform(0.3, 0.7)),
                float(rng.uniform(a_lo, a_hi)),
                float(rng.uniform(b_lo, b_hi)),
            ])
            labels.append(label)
    X = np.asarray(rows, dtype=float)
    perm = rng.permutation(X.shape[0])
    return BenchmarkDataset(
        slug="synthetic_order",
        name="Sintético (ordem de colunas)",
        X=X[perm],
        y=[labels[i] for i in perm],
        feature_names=["noise_x", "noise_y", "feat_a", "feat_b"],
        description="Ruído primeiro no CSV; sinal em feat_a/feat_b — teste de permutação.",
        tags=("synthetic", "small", "binary", "order-sensitive"),
    )


BUILTIN_LOADERS: dict[str, Callable[..., BenchmarkDataset]] = {
    "iris_binary": load_iris_binary,
    "breast_cancer_top6": load_breast_cancer_top6,
    "wine_binary": load_wine_binary,
    "synthetic_order": load_synthetic_order_sensitive,
}


def list_benchmark_slugs() -> list[str]:
    return list(BUILTIN_LOADERS.keys())


def load_benchmark_dataset(slug: str, **kwargs) -> BenchmarkDataset:
    if slug not in BUILTIN_LOADERS:
        raise KeyError(f"Dataset desconhecido: {slug}. Use: {', '.join(BUILTIN_LOADERS)}")
    return BUILTIN_LOADERS[slug](**kwargs)


def load_all_benchmark_datasets(**kwargs) -> list[BenchmarkDataset]:
    return [load_benchmark_dataset(slug, **kwargs) for slug in BUILTIN_LOADERS]
