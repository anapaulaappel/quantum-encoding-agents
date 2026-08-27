"""
Execução e métricas do benchmark de encoding / KTA / otimização de features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from benchmarks.datasets import BenchmarkDataset
from llama_qiskit_agents.quantum.encoding_optimization import (
    FeatureOptimizationResult,
    optimize_feature_encoding,
    pick_encoding_for_optimization,
)
from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.kernel import compute_kta_by_encoding
from llama_qiskit_agents.quantum.simulate import compare_embeddings


@dataclass
class BenchmarkCaseResult:
    dataset: str
    n_samples: int
    n_features: int
    n_classes: int
    baseline_best_encoding: str
    baseline_best_kta: float
    optimized_target_encoding: str
    optimization_baseline_kta: float
    optimization_optimized_kta: float
    optimization_delta: float
    optimization_plan: str
    compare_best_kta_without_opt: float | None
    compare_best_kta_with_opt: float | None
    compare_delta: float | None
    notes: list[str] = field(default_factory=list)


def _best_kta(
    X: np.ndarray,
    y: list[int],
    *,
    max_samples: int,
    encoding_types: list[EncodingType] | None = None,
) -> tuple[EncodingType, float]:
    scores = compute_kta_by_encoding(
        X,
        y,
        encoding_types=encoding_types,
        max_samples=max_samples,
    )
    if not scores:
        raise RuntimeError("Nenhum encoding completou cálculo de KTA.")
    enc, score = max(scores.items(), key=lambda kv: kv[1])
    return enc, score


def run_benchmark_case(
    dataset: BenchmarkDataset,
    *,
    max_kta_samples: int = 25,
    shots: int = 512,
    run_compare: bool = True,
    optimization_encoding: EncodingType | None = None,
) -> BenchmarkCaseResult:
    X = np.asarray(dataset.X, dtype=float)
    y = dataset.y
    notes: list[str] = []

    baseline_enc, baseline_kta = _best_kta(X, y, max_samples=max_kta_samples)

    target_enc = pick_encoding_for_optimization(
        X,
        y,
        preferred=optimization_encoding,
        max_samples=max_kta_samples,
    )
    opt: FeatureOptimizationResult | None = optimize_feature_encoding(
        X,
        y,
        target_enc,
        max_samples=max_kta_samples,
        column_names=dataset.feature_names,
    )
    if opt is None:
        notes.append("Otimização de features não retornou resultado.")
        opt_kta_base = float("nan")
        opt_kta_best = float("nan")
        opt_delta = float("nan")
        plan_str = "n/a"
    else:
        opt_kta_base = opt.baseline_kta
        opt_kta_best = opt.optimized_kta
        opt_delta = opt.improvement
        plan_str = opt.plan.describe(dataset.feature_names)

    compare_off: float | None = None
    compare_on: float | None = None
    compare_delta: float | None = None

    if run_compare:
        cr_off = compare_embeddings(
            X,
            labels=y,
            shots=shots,
            optimize_features=False,
            feature_column_names=dataset.feature_names,
        )
        cr_on = compare_embeddings(
            X,
            labels=y,
            shots=shots,
            optimize_features=True,
            feature_column_names=dataset.feature_names,
        )
        if cr_off.kta_by_encoding:
            compare_off = max(cr_off.kta_by_encoding.values())
        if cr_on.kta_by_encoding:
            compare_on = max(cr_on.kta_by_encoding.values())
        if compare_off is not None and compare_on is not None:
            compare_delta = compare_on - compare_off

    return BenchmarkCaseResult(
        dataset=dataset.slug,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=len(set(y)),
        baseline_best_encoding=baseline_enc.value,
        baseline_best_kta=baseline_kta,
        optimized_target_encoding=target_enc.value,
        optimization_baseline_kta=opt_kta_base,
        optimization_optimized_kta=opt_kta_best,
        optimization_delta=opt_delta,
        optimization_plan=plan_str,
        compare_best_kta_without_opt=compare_off,
        compare_best_kta_with_opt=compare_on,
        compare_delta=compare_delta,
        notes=notes,
    )


def run_benchmark_suite(
    datasets: list[BenchmarkDataset],
    *,
    max_kta_samples: int = 25,
    shots: int = 512,
    run_compare: bool = True,
) -> list[BenchmarkCaseResult]:
    return [
        run_benchmark_case(
            ds,
            max_kta_samples=max_kta_samples,
            shots=shots,
            run_compare=run_compare,
        )
        for ds in datasets
    ]


def format_results_table(results: list[BenchmarkCaseResult]) -> str:
    header = (
        f"{'dataset':<22} {'N':>4} {'d':>3} "
        f"{'best enc':<18} {'KTA base':>9} "
        f"{'opt Δ':>8} {'compare Δ':>10} plan"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        cmp_d = f"{r.compare_delta:+.4f}" if r.compare_delta is not None else "n/a"
        opt_d = f"{r.optimization_delta:+.4f}" if r.optimization_delta == r.optimization_delta else "n/a"
        plan_short = r.optimization_plan[:40] + ("…" if len(r.optimization_plan) > 40 else "")
        lines.append(
            f"{r.dataset:<22} {r.n_samples:>4} {r.n_features:>3} "
            f"{r.baseline_best_encoding:<18} {r.baseline_best_kta:>9.4f} "
            f"{opt_d:>8} {cmp_d:>10} {plan_short}"
        )
    return "\n".join(lines)


def format_result_detail(r: BenchmarkCaseResult) -> str:
    lines = [
        f"=== {r.dataset} ===",
        f"  Amostras/features/classes: {r.n_samples}/{r.n_features}/{r.n_classes}",
        f"  Melhor KTA baseline (qualquer encoding): {r.baseline_best_encoding} = {r.baseline_best_kta:.4f}",
        f"  Otimização ({r.optimized_target_encoding}): "
        f"{r.optimization_baseline_kta:.4f} → {r.optimization_optimized_kta:.4f} "
        f"(Δ={r.optimization_delta:+.4f})",
        f"  Plano: {r.optimization_plan}",
    ]
    if r.compare_best_kta_without_opt is not None:
        lines.append(
            f"  Compare ranking KTA: {r.compare_best_kta_without_opt:.4f} → "
            f"{r.compare_best_kta_with_opt:.4f} (Δ={r.compare_delta:+.4f})"
        )
    for note in r.notes:
        lines.append(f"  Nota: {note}")
    return "\n".join(lines)
