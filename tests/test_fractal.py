"""Orçamento de qubits por D2 e seleção FD-ASE no pipeline de encoding."""

from __future__ import annotations

import numpy as np
import pytest

from llama_qiskit_agents.api.schemas import profile_to_response
from llama_qiskit_agents.quantum.data_analysis import (
    DataProfile,
    effective_n_features,
    infer_data_profile,
    recommend_encoding,
)
from llama_qiskit_agents.quantum.encodings import EncodingType, build_encoding_circuit
from llama_qiskit_agents.quantum.explanation import build_natural_explanation, generate_qiskit_code
from llama_qiskit_agents.quantum.fractal import (
    FDASE,
    apply_column_selection,
    estimate_fractal_budget,
)
from llama_qiskit_agents.quantum.problem_context import ProblemContext
from llama_qiskit_agents.quantum.simulate import compare_embeddings


def _gaussian_r2_with_constants(n: int = 200, n_const: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cloud = rng.normal(size=(n, 2))
    const = np.full((n, n_const), 3.14)
    return np.hstack([cloud, const])


def _wide_signal_padding(n: int = 40, e: int = 18, n_signal: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n, n_signal))
    pad = np.zeros((n, e - n_signal))
    return np.hstack([signal, pad])


def test_d2_of_r2_cloud_is_near_two_not_embedding() -> None:
    x = _gaussian_r2_with_constants()
    budget = estimate_fractal_budget(x)
    assert budget.embedding_dimension == 5
    assert budget.intrinsic_dimension is not None
    assert 1.2 <= budget.intrinsic_dimension <= 2.6
    assert budget.intrinsic_dimension < budget.embedding_dimension - 0.5
    assert budget.qubit_budget == max(2, int(np.ceil(budget.intrinsic_dimension)))


def test_fdase_does_not_select_constant_columns() -> None:
    x = _gaussian_r2_with_constants()
    selector = FDASE()
    selector.fit(x)
    selected = [int(i) for i in selector.selected_]
    assert selected
    assert set(selected).issubset({0, 1})
    budget = estimate_fractal_budget(x)
    assert budget.selected_columns
    assert set(budget.selected_columns).issubset({0, 1})


def test_skip_when_too_few_samples() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 18))
    budget = estimate_fractal_budget(x)
    assert budget.intrinsic_dimension is None
    assert budget.selected_columns is None
    assert budget.qubit_budget is None


def test_wide_synthetic_budget_and_short_selection() -> None:
    x = _wide_signal_padding(n=40, e=18)
    budget = estimate_fractal_budget(x)
    assert budget.embedding_dimension == 18
    assert budget.intrinsic_dimension is not None
    assert budget.qubit_budget in {2, 3, 4}
    assert budget.selected_columns is not None
    assert 1 <= len(budget.selected_columns) <= budget.qubit_budget
    assert max(budget.selected_columns) < 18


def test_recommend_encoding_does_not_angle_encode_18_qubits() -> None:
    profile = DataProfile(
        n_samples=64,
        n_features=18,
        is_binary=False,
        is_categorical=False,
        is_continuous=True,
        has_negative=True,
        description="18 colunas sintéticas",
        intrinsic_dimension=2.7,
        embedding_dimension=18,
        qubit_budget=3,
        selected_columns=None,
    )
    enc, reason, _ctx = recommend_encoding(profile)
    assert enc != EncodingType.ANGLE
    assert enc != EncodingType.IQP
    assert effective_n_features(profile) == 18
    assert "q*" in reason or "orçamento" in reason

    cropped = DataProfile(
        n_samples=64,
        n_features=18,
        is_binary=False,
        is_categorical=False,
        is_continuous=True,
        has_negative=True,
        description="18 colunas, recorte FD-ASE",
        intrinsic_dimension=2.7,
        embedding_dimension=18,
        qubit_budget=3,
        selected_columns=[0, 1, 2],
    )
    enc_c, _reason_c, _ = recommend_encoding(cropped)
    if enc_c == EncodingType.ANGLE:
        assert effective_n_features(cropped) <= 3
    sample = apply_column_selection(np.arange(18, dtype=float).reshape(1, -1), cropped.selected_columns)
    qc = build_encoding_circuit(enc_c, sample[0])
    assert qc.num_qubits <= 3


def test_description_only_keeps_fractal_fields_none() -> None:
    profile = infer_data_profile("dados contínuos de classificação de pacientes")
    assert profile.n_features == 0
    assert profile.intrinsic_dimension is None
    assert profile.embedding_dimension is None
    assert profile.qubit_budget is None
    assert profile.selected_columns is None
    assert profile.selected_feature_names is None
    resp = profile_to_response(profile)
    assert resp.intrinsic_dimension is None
    enc, _reason, _ = recommend_encoding(profile)
    assert enc in EncodingType


def test_apply_fractal_budget_false_keeps_selection_advisory() -> None:
    x = _wide_signal_padding(n=40, e=18)
    on = infer_data_profile(x, apply_fractal_budget=True)
    off = infer_data_profile(x, apply_fractal_budget=False)
    assert on.selected_columns is not None
    assert on.fractal_selection_applied is True
    assert off.selected_columns is not None
    assert off.intrinsic_dimension is not None
    assert off.fractal_selection_applied is False
    assert effective_n_features(off) == 18
    assert effective_n_features(on) == len(on.selected_columns)
    assert off.n_features == 18

    enc_off, reason_off, _ = recommend_encoding(off)
    assert enc_off != EncodingType.ANGLE or effective_n_features(off) == 18
    assert "não aplicada" in reason_off or "sugeriria" in reason_off

    cr_on = compare_embeddings(
        x,
        encoding_types=[EncodingType.ANGLE],
        shots=8,
        apply_fractal_budget=True,
        sweep_qubit_budget=False,
    )
    cr_off = compare_embeddings(
        x,
        encoding_types=[EncodingType.ANGLE],
        shots=8,
        apply_fractal_budget=False,
        sweep_qubit_budget=False,
    )
    assert cr_off.profile.selected_columns is not None
    assert cr_off.profile.fractal_selection_applied is False
    assert cr_off.results[0].num_qubits == 18
    assert cr_on.profile.selected_columns is not None
    assert cr_on.results[0].num_qubits == len(cr_on.profile.selected_columns)
    assert cr_on.results[0].num_qubits <= (cr_on.profile.qubit_budget or 18)


def test_explanation_cites_d2_in_pt_and_en() -> None:
    profile = DataProfile(
        n_samples=64,
        n_features=18,
        is_binary=False,
        is_categorical=False,
        is_continuous=True,
        has_negative=False,
        description="wide",
        intrinsic_dimension=3.2,
        embedding_dimension=18,
        qubit_budget=4,
        selected_columns=[0, 2, 5],
        selected_feature_names=["a", "c", "f"],
    )
    ctx = ProblemContext()
    pt = build_natural_explanation(profile, EncodingType.DENSE_ANGLE, "", None, ctx, lang="pt")
    en = build_natural_explanation(profile, EncodingType.DENSE_ANGLE, "", None, ctx, lang="en")
    assert "D2" in pt and "orçamento" in pt and "FD-ASE" in pt
    assert "D2" in en and "qubit budget" in en and "FD-ASE" in en
    code = generate_qiskit_code(
        EncodingType.DENSE_ANGLE,
        [0.1, 0.2, 0.3],
        profile,
        lang="pt",
    )
    assert "FD-ASE" in code
    assert "selected_columns" not in code or "[0, 2, 5]" in code or "0, 2, 5" in code


def test_binary_and_narrow_arrays_skip_fdase() -> None:
    binary = np.array([[0, 1, 0], [1, 0, 1]] * 20, dtype=float)
    profile_bin = infer_data_profile(binary)
    assert profile_bin.is_binary
    assert profile_bin.intrinsic_dimension is None

    narrow = np.random.default_rng(0).normal(size=(40, 2))
    budget = estimate_fractal_budget(narrow)
    assert budget.intrinsic_dimension is None

    row = infer_data_profile([0.1, 0.2, 0.3, 0.4])
    assert row.n_samples == 1
    assert row.intrinsic_dimension is None
    assert row.embedding_dimension == 4


def test_qiskit_circuit_uses_selected_columns_only() -> None:
    x = _wide_signal_padding(n=40, e=18)
    profile = infer_data_profile(x)
    assert profile.selected_columns
    sample = apply_column_selection(x[:1], profile.selected_columns)[0]
    assert sample.shape[0] == len(profile.selected_columns)
    enc, _reason, _ = recommend_encoding(profile)
    qc = build_encoding_circuit(enc, sample)
    assert qc.num_qubits <= (profile.qubit_budget or qc.num_qubits)
    if enc == EncodingType.ANGLE:
        assert qc.num_qubits == len(profile.selected_columns)


def test_breast_cancer_demo_fields() -> None:
    pytest.importorskip("sklearn")
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    x = np.asarray(data.data, dtype=float)
    budget = estimate_fractal_budget(x, feature_names=list(data.feature_names))
    assert budget.embedding_dimension == 30
    assert budget.intrinsic_dimension is not None
    assert budget.qubit_budget is not None
    assert budget.selected_columns
    assert len(budget.selected_columns) <= budget.qubit_budget
    profile = infer_data_profile(x, feature_names=list(data.feature_names))
    enc, _reason, _ = recommend_encoding(profile)
    assert enc != EncodingType.ANGLE or effective_n_features(profile) <= (profile.qubit_budget or 30)
