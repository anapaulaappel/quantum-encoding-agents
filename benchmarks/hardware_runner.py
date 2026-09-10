"""
Benchmark em hardware IBM — preset enxuto para ~10 min/mês de QPU.

Fluxo:
  1. Simulador: otimiza plano de features (KTA) — benchmarks/runner.py
  2. Transpile-only: depth/2q no backend alvo (grátis)
  3. Execute (opcional): um Sampler job com 4 pubs — 1 amostra/classe × baseline vs otimizado

Preset --monthly: iris_binary, angle encoding, 1 job Sampler (4 pubs), 512 shots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from benchmarks.datasets import BenchmarkDataset
from llama_qiskit_agents.quantum.encoding_optimization import (
    FeatureEncodingPlan,
    optimize_feature_encoding,
    pick_encoding_for_optimization,
)
from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.hardware_run import (
    IBMHardwareError,
    HardwareRunResult,
    TranspileStats,
    build_sample_encoding_circuit,
    resolve_backend_name,
    run_encoding_circuits,
    transpile_encoding_circuit,
)
from llama_qiskit_agents.quantum.kernel import compute_kta_by_encoding
from llama_qiskit_agents.quantum.preprocessing import FeatureBounds


@dataclass
class HardwareJobSpec:
    label: str
    plan: str  # baseline | optimized
    sample_index: int
    class_label: int
    row: list[float]


@dataclass
class HardwareJobOutcome:
    spec: HardwareJobSpec
    executed: bool
    transpile: TranspileStats | None = None
    run: HardwareRunResult | None = None
    error: str | None = None


@dataclass
class HardwareBenchmarkResult:
    dataset: str
    backend: str
    encoding: str
    sim_kta_baseline_plan: float
    sim_kta_optimized_plan: float
    sim_kta_delta: float
    baseline_plan_desc: str
    optimized_plan_desc: str
    jobs: list[HardwareJobOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def estimated_qpu_seconds(self) -> float:
        return sum(j.run.elapsed_sec for j in self.jobs if j.run is not None)


def _pick_one_sample_per_class(y: list[int]) -> list[tuple[int, int]]:
    """Retorna [(sample_idx, class_label), ...] com uma amostra por classe."""
    by_class: dict[int, int] = {}
    for idx, label in enumerate(y):
        if label not in by_class:
            by_class[label] = idx
    return [(idx, lbl) for lbl, idx in sorted(by_class.items())]


def _subsample_stratified(
    X: np.ndarray,
    y: list[int],
    max_samples: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """Primeiras N linhas do Iris vêm de uma só classe; o hardware precisa das duas."""
    n = X.shape[0]
    if n <= max_samples:
        return X, y
    y_arr = np.asarray(y)
    classes = sorted(set(y))
    rng = np.random.default_rng(seed)
    per = max(1, max_samples // max(1, len(classes)))
    idx: list[int] = []
    for lab in classes:
        lab_idx = np.where(y_arr == lab)[0]
        take = min(per, len(lab_idx))
        idx.extend(int(i) for i in rng.choice(lab_idx, size=take, replace=False))
    leftover = max_samples - len(idx)
    if leftover > 0:
        rest = np.setdiff1d(np.arange(n), np.asarray(idx, dtype=int))
        if len(rest) > 0:
            extra = rng.choice(rest, size=min(leftover, len(rest)), replace=False)
            idx.extend(int(i) for i in np.atleast_1d(extra))
    chosen = np.asarray(sorted(idx), dtype=int)
    return X[chosen], [y[int(i)] for i in chosen]


def _apply_plan(X: np.ndarray, plan: FeatureEncodingPlan) -> np.ndarray:
    return plan.apply(X)


def _kta(plan: FeatureEncodingPlan, X: np.ndarray, y: list[int], enc: EncodingType) -> float:
    mapped = _apply_plan(X, plan)
    bounds = FeatureBounds.fit(mapped) if mapped.shape[0] >= 2 else None
    scores = compute_kta_by_encoding(
        mapped,
        y,
        encoding_types=[enc],
        feature_bounds=bounds,
        max_samples=min(25, mapped.shape[0]),
    )
    return scores.get(enc, float("nan"))


def build_hardware_job_specs(
    dataset: BenchmarkDataset,
    *,
    encoding: EncodingType,
    max_samples: int = 12,
) -> tuple[HardwareBenchmarkResult, list[HardwareJobSpec], FeatureEncodingPlan, FeatureEncodingPlan]:
    X = np.asarray(dataset.X, dtype=float)
    y = dataset.y
    if len(set(y)) < 2:
        raise ValueError("Hardware benchmark requer labels binários (≥2 classes).")

    if X.shape[0] > max_samples:
        X, y = _subsample_stratified(X, y, max_samples)

    enc = encoding
    target = pick_encoding_for_optimization(X, y, preferred=enc, max_samples=min(25, X.shape[0]))
    opt = optimize_feature_encoding(
        X,
        y,
        target,
        max_samples=min(25, X.shape[0]),
        column_names=dataset.feature_names,
    )
    if opt is None:
        raise RuntimeError("Otimização de features falhou — necessária para comparar planos.")

    baseline_plan = FeatureEncodingPlan.identity(X.shape[1])
    optimized_plan = opt.plan

    sim_base = _kta(baseline_plan, X, y, target)
    sim_opt = _kta(optimized_plan, X, y, target)

    samples = _pick_one_sample_per_class(y)
    specs: list[HardwareJobSpec] = []
    for sample_idx, class_label in samples:
        row_base = baseline_plan.apply_vector(X[sample_idx]).tolist()
        row_opt = optimized_plan.apply_vector(X[sample_idx]).tolist()
        specs.append(
            HardwareJobSpec(
                label=f"class{class_label}_baseline",
                plan="baseline",
                sample_index=sample_idx,
                class_label=class_label,
                row=row_base,
            )
        )
        specs.append(
            HardwareJobSpec(
                label=f"class{class_label}_optimized",
                plan="optimized",
                sample_index=sample_idx,
                class_label=class_label,
                row=row_opt,
            )
        )

    result = HardwareBenchmarkResult(
        dataset=dataset.slug,
        backend=resolve_backend_name(),
        encoding=target.value,
        sim_kta_baseline_plan=sim_base,
        sim_kta_optimized_plan=sim_opt,
        sim_kta_delta=sim_opt - sim_base,
        baseline_plan_desc=baseline_plan.describe(dataset.feature_names),
        optimized_plan_desc=optimized_plan.describe(dataset.feature_names),
        notes=[
            "Simulador escolhe plano; hardware valida execução e histogramas.",
            f"Jobs planejados: {len(specs)} (1 amostra/classe × baseline/otimizado).",
        ],
    )
    return result, specs, baseline_plan, optimized_plan


def _bounds_for_plan(plan: FeatureEncodingPlan, X: np.ndarray) -> FeatureBounds | None:
    mapped = plan.apply(X)
    if mapped.shape[0] < 2:
        return None
    return FeatureBounds.fit(mapped)


def run_hardware_benchmark(
    dataset: BenchmarkDataset,
    *,
    backend_name: str | None = None,
    encoding: EncodingType = EncodingType.ANGLE,
    shots: int = 512,
    execute: bool = False,
    max_jobs: int | None = None,
    max_samples: int = 12,
) -> HardwareBenchmarkResult:
    """
    execute=False → só transpile (não gasta QPU).
    execute=True  → submete jobs (respeita max_jobs).
    """
    backend = resolve_backend_name(backend_name)
    result, specs, baseline_plan, optimized_plan = build_hardware_job_specs(
        dataset,
        encoding=encoding,
        max_samples=max_samples,
    )

    if max_jobs is not None:
        specs = specs[:max_jobs]
        result.notes.append(f"Limitado a max_jobs={max_jobs}.")

    X_fit = np.asarray(dataset.X, dtype=float)
    y_fit = dataset.y
    if X_fit.shape[0] > max_samples:
        X_fit, y_fit = _subsample_stratified(X_fit, y_fit, max_samples)
    del y_fit
    bounds_by_plan = {
        "baseline": _bounds_for_plan(baseline_plan, X_fit),
        "optimized": _bounds_for_plan(optimized_plan, X_fit),
    }

    circuits: list = []
    for spec in specs:
        outcome = HardwareJobOutcome(spec=spec, executed=execute)
        try:
            qc = build_sample_encoding_circuit(
                spec.row,
                encoding_type=EncodingType(result.encoding),
                feature_bounds=bounds_by_plan[spec.plan],
            )
            _, tstats = transpile_encoding_circuit(qc, backend_name=backend)
            outcome.transpile = tstats
            circuits.append(qc)
        except IBMHardwareError as exc:
            outcome.error = str(exc)
            circuits.append(None)
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            circuits.append(None)
        result.jobs.append(outcome)

    if execute:
        ready = [(i, qc) for i, qc in enumerate(circuits) if qc is not None]
        if ready:
            try:
                runs = run_encoding_circuits(
                    [qc for _, qc in ready],
                    backend_name=backend,
                    shots=shots,
                )
                for (idx, _), run in zip(ready, runs):
                    result.jobs[idx].run = run
            except IBMHardwareError as exc:
                for idx, _ in ready:
                    if result.jobs[idx].error is None:
                        result.jobs[idx].error = str(exc)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                for idx, _ in ready:
                    if result.jobs[idx].error is None:
                        result.jobs[idx].error = msg

    result.backend = backend
    return result


def format_hardware_benchmark_report(result: HardwareBenchmarkResult) -> str:
    lines = [
        "=== Benchmark hardware IBM ===",
        f"  Dataset: {result.dataset}",
        f"  Backend: {result.backend}",
        f"  Encoding: {result.encoding}",
        "",
        "  KTA (simulador, plano CSV vs otimizado):",
        f"    baseline:  {result.sim_kta_baseline_plan:.4f}",
        f"    otimizado: {result.sim_kta_optimized_plan:.4f}  (Δ={result.sim_kta_delta:+.4f})",
        "",
        f"  Plano CSV:       {result.baseline_plan_desc}",
        f"  Plano otimizado: {result.optimized_plan_desc}",
        "",
        "  Jobs:",
    ]
    for job in result.jobs:
        s = job.spec
        lines.append(
            f"    • {s.label}  amostra={s.sample_index} classe={s.class_label} plano={s.plan}"
        )
        if job.transpile:
            lines.append(f"      transpile: {job.transpile.summary()}")
        if job.run:
            top = sorted(job.run.counts.items(), key=lambda kv: -kv[1])[:3]
            lines.append(
                f"      exec: job_id={job.run.job_id}, {job.run.elapsed_sec:.1f}s, "
                f"top counts={dict(top)}"
            )
        if job.error:
            lines.append(f"      ERRO: {job.error}")
    if result.estimated_qpu_seconds() > 0:
        lines.append(f"\n  Tempo execução (wall, {len(result.jobs)} jobs): "
                     f"{result.estimated_qpu_seconds():.1f}s")
    for note in result.notes:
        lines.append(f"  Nota: {note}")
    lines.append("")
    return "\n".join(lines)


MONTHLY_PRESETS: dict[str, dict] = {
    "week1_iris": {
        "dataset": "iris_binary",
        "encoding": EncodingType.ANGLE,
        "max_jobs": 4,
        "shots": 512,
        "description": "Iris 2 classes — 4 jobs (~2–4 min QPU + fila). Recomendado para 1ª corrida do mês.",
    },
    "week2_synthetic": {
        "dataset": "synthetic_order",
        "encoding": EncodingType.DENSE_ANGLE,
        "max_jobs": 4,
        "shots": 512,
        "description": "Sintético ordem de colunas — mostra Δ KTA grande também no hardware.",
    },
    "week3_breast": {
        "dataset": "breast_cancer_top6",
        "encoding": EncodingType.ANGLE,
        "max_jobs": 4,
        "shots": 512,
        "max_samples": 20,
        "description": "Breast cancer top-6 — mais lento no simulador; reserve ~10 min totais.",
    },
}
