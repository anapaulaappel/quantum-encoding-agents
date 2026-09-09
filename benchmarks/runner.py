"""
Execução e métricas do benchmark de encoding / KTA / otimização de features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from benchmarks.datasets import BenchmarkDataset
from llama_qiskit_agents.quantum.encoding_optimization import (
    FeatureOptimizationResult,
    optimize_feature_encoding,
    pick_encoding_for_optimization,
)
from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.fractal import apply_column_selection, estimate_fractal_budget
from llama_qiskit_agents.quantum.kernel import (
    KernelDiagnostics,
    compute_kernel_diagnostics_by_encoding,
)
from llama_qiskit_agents.quantum.preprocessing import FeatureBounds
from llama_qiskit_agents.quantum.simulate import compare_embeddings


@dataclass
class BenchmarkCaseResult:
    dataset: str
    n_samples: int
    n_features: int
    n_features_eval: int
    n_classes: int
    d2: float | None
    qubit_budget: int | None
    selected_columns: list[int] | None
    selected_feature_names: list[str] | None
    baseline_best_encoding: str
    baseline_best_kta: float
    baseline_kernel_alive: bool | None
    baseline_fid_near: float | None
    baseline_fid_far: float | None
    optimized_target_encoding: str
    optimization_baseline_kta: float
    optimization_optimized_kta: float
    optimization_delta: float
    optimization_plan: str
    compare_best_kta_without_opt: float | None
    compare_best_kta_with_opt: float | None
    compare_delta: float | None
    notes: list[str] = field(default_factory=list)


def _eval_view(
    dataset: BenchmarkDataset,
    *,
    apply_fractal_budget: bool,
) -> tuple[np.ndarray, list[str], float | None, int | None, list[int] | None, list[str] | None]:
    """Matriz para KTA/alive/opt: recorte FD-ASE quando D2 for estimável."""
    x = np.asarray(dataset.X, dtype=float)
    names = list(dataset.feature_names)
    if not apply_fractal_budget:
        return x, names, None, None, None, None
    budget = estimate_fractal_budget(x, feature_names=names)
    cols = budget.selected_columns
    if not cols:
        return (
            x,
            names,
            budget.intrinsic_dimension,
            budget.qubit_budget,
            None,
            None,
        )
    x_eval = apply_column_selection(x, cols)
    names_eval = [names[i] for i in cols] if len(names) >= x.shape[1] else [str(i) for i in cols]
    return (
        x_eval,
        names_eval,
        budget.intrinsic_dimension,
        budget.qubit_budget,
        list(cols),
        list(budget.selected_feature_names or names_eval),
    )


def _best_from_diagnostics(
    diagnostics: dict[EncodingType, KernelDiagnostics],
) -> tuple[EncodingType, float, KernelDiagnostics]:
    scored = {enc: diag for enc, diag in diagnostics.items() if diag.kta is not None}
    if not scored:
        raise RuntimeError("Nenhum encoding completou cálculo de KTA.")
    enc, diag = max(scored.items(), key=lambda kv: kv[1].kta or float("-inf"))
    return enc, float(diag.kta), diag


def run_benchmark_case(
    dataset: BenchmarkDataset,
    *,
    max_kta_samples: int = 25,
    shots: int = 512,
    run_compare: bool = True,
    optimization_encoding: EncodingType | None = None,
    apply_fractal_budget: bool = True,
) -> BenchmarkCaseResult:
    x_raw = np.asarray(dataset.X, dtype=float)
    y = dataset.y
    notes: list[str] = []
    x_eval, names_eval, d2, q_star, cols, col_names = _eval_view(
        dataset,
        apply_fractal_budget=apply_fractal_budget,
    )
    if cols:
        notes.append(
            f"FD-ASE: E={x_raw.shape[1]} → {x_eval.shape[1]} colunas "
            f"(D2={d2}, q*={q_star}): {cols}"
        )
    elif apply_fractal_budget:
        notes.append("FD-ASE não aplicado (N<32, E≤2, binário, ou D2 indisponível).")

    bounds = FeatureBounds.fit(x_eval) if x_eval.shape[0] >= 2 else None
    diagnostics = compute_kernel_diagnostics_by_encoding(
        x_eval,
        labels=y,
        max_samples=max_kta_samples,
        feature_bounds=bounds,
    )
    baseline_enc, baseline_kta, baseline_diag = _best_from_diagnostics(diagnostics)
    alive = baseline_diag.geometry.kernel_alive
    fid_near = baseline_diag.geometry.fid_near
    fid_far = baseline_diag.geometry.fid_far

    target_enc = pick_encoding_for_optimization(
        x_eval,
        y,
        preferred=optimization_encoding,
        max_samples=max_kta_samples,
    )
    opt: FeatureOptimizationResult | None = optimize_feature_encoding(
        x_eval,
        y,
        target_enc,
        max_samples=max_kta_samples,
        column_names=names_eval,
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
        plan_str = opt.plan.describe(names_eval)

    compare_off: float | None = None
    compare_on: float | None = None
    compare_delta: float | None = None

    if run_compare:
        cr_off = compare_embeddings(
            x_raw,
            labels=y,
            shots=shots,
            optimize_features=False,
            feature_column_names=dataset.feature_names,
            apply_fractal_budget=apply_fractal_budget,
            sweep_qubit_budget=False,
        )
        cr_on = compare_embeddings(
            x_raw,
            labels=y,
            shots=shots,
            optimize_features=True,
            feature_column_names=dataset.feature_names,
            apply_fractal_budget=apply_fractal_budget,
            sweep_qubit_budget=False,
        )
        if cr_off.kta_by_encoding:
            compare_off = max(cr_off.kta_by_encoding.values())
        if cr_on.kta_by_encoding:
            compare_on = max(cr_on.kta_by_encoding.values())
        if compare_off is not None and compare_on is not None:
            compare_delta = compare_on - compare_off

    return BenchmarkCaseResult(
        dataset=dataset.slug,
        n_samples=int(x_raw.shape[0]),
        n_features=int(x_raw.shape[1]),
        n_features_eval=int(x_eval.shape[1]),
        n_classes=len(set(y)),
        d2=d2,
        qubit_budget=q_star,
        selected_columns=cols,
        selected_feature_names=col_names,
        baseline_best_encoding=baseline_enc.value,
        baseline_best_kta=baseline_kta,
        baseline_kernel_alive=alive,
        baseline_fid_near=fid_near,
        baseline_fid_far=fid_far,
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
    apply_fractal_budget: bool = True,
) -> list[BenchmarkCaseResult]:
    return [
        run_benchmark_case(
            ds,
            max_kta_samples=max_kta_samples,
            shots=shots,
            run_compare=run_compare,
            apply_fractal_budget=apply_fractal_budget,
        )
        for ds in datasets
    ]


def format_results_table(results: list[BenchmarkCaseResult]) -> str:
    header = (
        f"{'dataset':<22} {'N':>4} {'E':>3} {'q':>3} "
        f"{'alive':<6} {'best enc':<18} {'KTA base':>9} "
        f"{'opt Δ':>8} {'compare Δ':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        cmp_d = f"{r.compare_delta:+.4f}" if r.compare_delta is not None else "n/a"
        opt_d = f"{r.optimization_delta:+.4f}" if r.optimization_delta == r.optimization_delta else "n/a"
        alive = (
            "ALIVE"
            if r.baseline_kernel_alive is True
            else "DEAD"
            if r.baseline_kernel_alive is False
            else "n/a"
        )
        q_txt = r.n_features_eval
        lines.append(
            f"{r.dataset:<22} {r.n_samples:>4} {r.n_features:>3} {q_txt:>3} "
            f"{alive:<6} {r.baseline_best_encoding:<18} {r.baseline_best_kta:>9.4f} "
            f"{opt_d:>8} {cmp_d:>10}"
        )
    return "\n".join(lines)


def format_result_detail(r: BenchmarkCaseResult) -> str:
    alive = (
        "ALIVE"
        if r.baseline_kernel_alive is True
        else "DEAD"
        if r.baseline_kernel_alive is False
        else "n/a"
    )
    d2_txt = "—" if r.d2 is None else f"{r.d2:.3f}"
    qstar_txt = "—" if r.qubit_budget is None else str(r.qubit_budget)
    cols_txt = "—" if not r.selected_columns else str(r.selected_columns)
    names_txt = ""
    if r.selected_feature_names:
        names_txt = " (" + ", ".join(r.selected_feature_names) + ")"
    lines = [
        f"=== {r.dataset} ===",
        f"  Amostras/E original/classes: {r.n_samples}/{r.n_features}/{r.n_classes}",
        f"  FD-ASE: D2={d2_txt}, q*={qstar_txt}, largura eval={r.n_features_eval}, colunas={cols_txt}{names_txt}",
        f"  Melhor KTA baseline: {r.baseline_best_encoding} = {r.baseline_best_kta:.4f} "
        f"({alive}; near={r.baseline_fid_near}, far={r.baseline_fid_far})",
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
