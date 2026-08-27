"""Testes dos bugs do hardware_runner (encoding_type, bounds, max_samples)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

sklearn = pytest.importorskip("sklearn")

from benchmarks.datasets import load_breast_cancer_top6, load_iris_binary  # noqa: E402
from benchmarks.hardware_runner import (  # noqa: E402
    HardwareBenchmarkResult,
    HardwareJobSpec,
    build_hardware_job_specs,
    run_hardware_benchmark,
)
from llama_qiskit_agents.quantum.encoding_optimization import (  # noqa: E402
    FeatureEncodingPlan,
    FeatureOptimizationResult,
)
from llama_qiskit_agents.quantum.encodings import EncodingType  # noqa: E402


def _stub_optimization_result(n_features: int) -> FeatureOptimizationResult:
    plan = FeatureEncodingPlan(column_indices=[0, min(1, n_features - 1)], weights=[1.0, 2.0])
    return FeatureOptimizationResult(
        plan=plan,
        baseline_plan=FeatureEncodingPlan.identity(n_features),
        encoding_type=EncodingType.ANGLE,
        baseline_kta=0.1,
        optimized_kta=0.2,
    )


def test_build_hardware_job_specs_honors_max_samples(monkeypatch) -> None:
    seen: dict[str, int] = {}

    def fake_opt(X, y, *args, **kwargs):
        seen["n"] = int(X.shape[0])
        return _stub_optimization_result(X.shape[1])

    monkeypatch.setattr(
        "benchmarks.hardware_runner.pick_encoding_for_optimization",
        lambda *a, **k: EncodingType.ANGLE,
    )
    monkeypatch.setattr("benchmarks.hardware_runner.optimize_feature_encoding", fake_opt)
    monkeypatch.setattr("benchmarks.hardware_runner._kta", lambda *a, **k: 0.1)

    ds = load_breast_cancer_top6(max_samples=40)
    build_hardware_job_specs(ds, encoding=EncodingType.ANGLE, max_samples=20)
    assert seen["n"] == 20


def test_run_hardware_benchmark_forwards_max_samples(monkeypatch) -> None:
    ds = load_iris_binary(max_samples=8)
    seen: dict[str, int] = {}
    plan = FeatureEncodingPlan.identity(ds.X.shape[1])
    empty = HardwareBenchmarkResult(
        dataset=ds.slug,
        backend="fake",
        encoding=EncodingType.ANGLE.value,
        sim_kta_baseline_plan=0.0,
        sim_kta_optimized_plan=0.0,
        sim_kta_delta=0.0,
        baseline_plan_desc="id",
        optimized_plan_desc="id",
    )

    def fake_specs(dataset, *, encoding, max_samples=12):
        seen["max_samples"] = max_samples
        return empty, [], plan, plan

    monkeypatch.setattr("benchmarks.hardware_runner.build_hardware_job_specs", fake_specs)
    monkeypatch.setattr("benchmarks.hardware_runner.resolve_backend_name", lambda name=None: "fake")

    run_hardware_benchmark(ds, execute=False, max_samples=20)
    assert seen["max_samples"] == 20


def test_run_hardware_benchmark_encoding_type_and_plan_bounds(monkeypatch) -> None:
    ds = load_iris_binary(max_samples=8)
    baseline = FeatureEncodingPlan.identity(ds.X.shape[1])
    optimized = FeatureEncodingPlan(column_indices=[2, 0], weights=[1.0, 2.0])
    specs = [
        HardwareJobSpec(
            label="class0_baseline",
            plan="baseline",
            sample_index=0,
            class_label=0,
            row=baseline.apply_vector(ds.X[0]).tolist(),
        ),
        HardwareJobSpec(
            label="class0_optimized",
            plan="optimized",
            sample_index=0,
            class_label=0,
            row=optimized.apply_vector(ds.X[0]).tolist(),
        ),
    ]
    result_obj = HardwareBenchmarkResult(
        dataset=ds.slug,
        backend="fake",
        encoding=EncodingType.ANGLE.value,
        sim_kta_baseline_plan=0.1,
        sim_kta_optimized_plan=0.2,
        sim_kta_delta=0.1,
        baseline_plan_desc="id",
        optimized_plan_desc="drop",
    )
    captured: list[tuple[EncodingType, int, int | None]] = []

    def fake_specs(dataset, *, encoding, max_samples=12):
        return result_obj, specs, baseline, optimized

    def fake_circuit(row, encoding_type, *, feature_bounds=None):
        captured.append(
            (
                encoding_type,
                len(list(row)),
                None if feature_bounds is None else len(feature_bounds.col_min),
            )
        )
        return MagicMock()

    monkeypatch.setattr("benchmarks.hardware_runner.build_hardware_job_specs", fake_specs)
    monkeypatch.setattr("benchmarks.hardware_runner.build_sample_encoding_circuit", fake_circuit)
    monkeypatch.setattr("benchmarks.hardware_runner.resolve_backend_name", lambda name=None: "fake")
    monkeypatch.setattr(
        "benchmarks.hardware_runner.transpile_encoding_circuit",
        lambda qc, backend_name=None: (qc, MagicMock()),
    )

    run_hardware_benchmark(ds, execute=False)

    assert len(captured) == 2
    assert all(enc is EncodingType.ANGLE for enc, _, _ in captured)
    for _, row_len, bounds_len in captured:
        assert bounds_len == row_len
    assert captured[0][1] == ds.X.shape[1]
    assert captured[1][1] == 2
