"""Simulação de circuitos de encoding e comparação entre estratégias."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from llama_qiskit_agents.quantum.data_analysis import (
    DataProfile,
    effective_n_features,
    get_encoding_tradeoffs,
    infer_data_profile,
    load_csv,
    recommend_encoding,
)
from llama_qiskit_agents.quantum.encoding_ranking import (
    format_encoding_ranking_section,
    format_measurements_note_section,
)
from llama_qiskit_agents.quantum.encodings import (
    EncodingType,
    build_encoding_circuit,
)
from llama_qiskit_agents.quantum.encoding_optimization import (
    FeatureOptimizationResult,
    format_feature_optimization_section,
    optimize_feature_encoding,
    pick_encoding_for_optimization,
)
from llama_qiskit_agents.quantum.fractal import apply_column_selection, apply_column_selection_row
from llama_qiskit_agents.quantum.kernel import (
    MAX_QUBITS_FOR_KERNEL_DIAGNOSTICS,
    KernelGeometry,
    compute_kernel_diagnostics_by_encoding,
    format_kernel_alive_section,
)
from llama_qiskit_agents.quantum.qubit_sweep import (
    DEFAULT_SWEEP_N_SAMPLES,
    DEFAULT_SWEEP_Q_MAX,
    QubitSweepResult,
    format_qubit_sweep_section,
    run_qubit_budget_sweep,
)
from llama_qiskit_agents.quantum.preprocessing import (
    FeatureBounds,
    format_preprocess_section,
    preprocess_for_encoding,
)
from llama_qiskit_agents.quantum.simulability import format_simulability_section
from llama_qiskit_agents.quantum.problem_context import ProblemContext


@dataclass
class SimulationResult:
    """Resultado da simulação de um circuito de encoding."""

    encoding_type: EncodingType
    circuit: QuantumCircuit
    depth: int
    num_qubits: int
    counts: dict[str, int]
    shots: int
    statevector: np.ndarray | None = field(default=None, repr=False)


@dataclass
class CompareEmbeddingsResult:
    """Saída agregada de compare_embeddings."""

    results: list[SimulationResult]
    profile: DataProfile
    recommended: EncodingType
    reason: str
    context: ProblemContext
    kta_by_encoding: dict[EncodingType, float] | None = None
    kernel_geometry_by_encoding: dict[EncodingType, KernelGeometry] | None = None
    labels: list[int] | None = None
    feature_bounds: FeatureBounds | None = None
    sample_row: np.ndarray | None = None
    feature_optimization: FeatureOptimizationResult | None = None
    feature_column_names: list[str] | None = None
    qubit_sweep: QubitSweepResult | None = None


def simulate_encoding_circuit(
    circuit: QuantumCircuit,
    encoding_type: EncodingType,
    shots: int = 1024,
    save_statevector: bool = False,
) -> SimulationResult:
    """
    Simula um circuito de encoding: adiciona medição, transpila e executa no AerSimulator.

    Args:
        save_statevector: se True, captura também o statevector puro antes da medição
                          via StatevectorSimulator (para visualização da esfera de Bloch).
    """
    qc = circuit.copy()
    if qc.num_clbits == 0:
        qc.measure_all()
    sim = AerSimulator()
    transpiled = transpile(qc, sim)
    job = sim.run(transpiled, shots=shots)
    counts = dict(job.result().get_counts())

    sv = None
    if save_statevector:
        from llama_qiskit_agents.quantum.visualization import simulate_statevector
        sv = simulate_statevector(circuit)

    return SimulationResult(
        encoding_type=encoding_type,
        circuit=circuit,
        depth=transpiled.depth(),
        num_qubits=qc.num_qubits,
        counts=counts,
        shots=shots,
        statevector=sv,
    )


def compare_embeddings(
    data: np.ndarray | list[float] | str | Path,
    encoding_types: list[EncodingType] | None = None,
    n_qubits: int | None = None,
    shots: int = 1024,
    *,
    task: str | None = None,
    algorithm: str | None = None,
    problem_description: str | None = None,
    labels: list[int] | None = None,
    optimize_features: bool = False,
    optimization_encoding: EncodingType | None = None,
    feature_column_names: list[str] | None = None,
    apply_fractal_budget: bool = True,
    sweep_qubit_budget: bool = True,
    sweep_q_max: int = DEFAULT_SWEEP_Q_MAX,
    sweep_n_samples: int = DEFAULT_SWEEP_N_SAMPLES,
) -> CompareEmbeddingsResult:
    """
    Compara múltiplos encodings no mesmo dado: simula cada um e retorna
    resultados, perfil do dado, encoding recomendado e justificativa.
    Com labels (≥2 classes), calcula KTA por encoding para reordenar o ranking.
    Com N≥4, avalia também kernel-alive (geometria de K, independente de rótulo).
    Com D2 estimado, varre q em FD-ASE vs PCA vs prefixo (probe = angle).
    """
    if encoding_types is None:
        encoding_types = list(EncodingType)

    full_x: np.ndarray | None = None

    if isinstance(data, (str, Path)):
        p = Path(data) if isinstance(data, str) else data
        if str(p).lower().endswith(".csv") and p.exists():
            full_x = load_csv(p)
            profile = infer_data_profile(
                full_x,
                feature_names=feature_column_names,
                apply_fractal_budget=apply_fractal_budget,
            )
            data_arr = full_x[0] if full_x.ndim > 1 else full_x.flatten()
        else:
            profile = infer_data_profile(data, apply_fractal_budget=apply_fractal_budget)
            data_arr = np.array([0.1, 0.2, 0.3])
    else:
        full_x = np.asarray(data)
        profile = infer_data_profile(
            full_x,
            feature_names=feature_column_names,
            apply_fractal_budget=apply_fractal_budget,
        )
        if full_x.ndim > 1 and full_x.shape[0] > 0:
            data_arr = full_x[0]
        else:
            data_arr = full_x.flatten() if hasattr(full_x, "__len__") else np.array([0.0])

    original_n_features = int(profile.n_features)
    original_x: np.ndarray | None = None
    if full_x is not None and full_x.ndim == 2:
        original_x = np.asarray(full_x, dtype=float)
    if apply_fractal_budget and full_x is not None and full_x.ndim == 2 and profile.selected_columns:
        full_x = apply_column_selection(full_x, profile.selected_columns)
        data_arr = apply_column_selection_row(data_arr, profile.selected_columns)
        if feature_column_names and len(feature_column_names) >= original_n_features:
            feature_column_names = [feature_column_names[i] for i in profile.selected_columns]

    if len(data_arr) == 0:
        data_arr = np.array([0.1, 0.2, 0.3])

    bounds: FeatureBounds | None = None
    if full_x is not None and full_x.ndim == 2 and full_x.shape[0] >= 2:
        bounds = FeatureBounds.fit(full_x)

    recommended, reason, ctx = recommend_encoding(
        profile,
        task=task,
        algorithm=algorithm,
        problem_description=problem_description,
    )

    feature_optimization: FeatureOptimizationResult | None = None
    if (
        optimize_features
        and labels is not None
        and full_x is not None
        and full_x.ndim == 2
        and full_x.shape[0] >= 2
        and len(labels) == full_x.shape[0]
        and len(set(labels)) >= 2
    ):
        enc_for_opt = pick_encoding_for_optimization(
            full_x,
            labels,
            preferred=optimization_encoding,
            n_qubits=n_qubits,
        )
        feature_optimization = optimize_feature_encoding(
            full_x,
            labels,
            enc_for_opt,
            n_qubits=n_qubits,
            column_names=feature_column_names,
        )
        if feature_optimization is not None:
            full_x = feature_optimization.plan.apply(full_x)
            fractal_fields = (
                profile.intrinsic_dimension,
                profile.embedding_dimension or original_n_features,
                profile.qubit_budget,
                profile.selected_columns,
                profile.selected_feature_names,
                profile.fractal_selection_applied,
            )
            profile = infer_data_profile(full_x, apply_fractal_budget=False)
            profile.intrinsic_dimension = fractal_fields[0]
            profile.embedding_dimension = fractal_fields[1]
            profile.qubit_budget = fractal_fields[2]
            profile.selected_columns = fractal_fields[3]
            profile.selected_feature_names = fractal_fields[4]
            profile.fractal_selection_applied = fractal_fields[5]
            bounds = FeatureBounds.fit(full_x) if full_x.shape[0] >= 2 else bounds
            data_arr = full_x[0]

    results: list[SimulationResult] = []
    for enc in encoding_types:
        try:
            qc = build_encoding_circuit(
                enc,
                data_arr,
                n_qubits=n_qubits,
                feature_bounds=bounds,
            )
            res = simulate_encoding_circuit(qc, enc, shots=shots)
            results.append(res)
        except Exception:
            continue

    kta_scores: dict[EncodingType, float] | None = None
    geometry: dict[EncodingType, KernelGeometry] | None = None
    if full_x is not None and full_x.ndim == 2 and full_x.shape[0] >= 2:
        labels_for_kta: list[int] | None = None
        if (
            labels is not None
            and len(labels) == full_x.shape[0]
            and len(set(labels)) >= 2
        ):
            labels_for_kta = labels
        encodings_for_diag = [
            r.encoding_type
            for r in results
            if r.num_qubits <= MAX_QUBITS_FOR_KERNEL_DIAGNOSTICS
        ]
        if encodings_for_diag:
            diagnostics = compute_kernel_diagnostics_by_encoding(
                full_x,
                labels=labels_for_kta,
                encoding_types=encodings_for_diag,
                n_qubits=n_qubits,
                feature_bounds=bounds,
            )
            if diagnostics:
                kta_scores = {
                    enc: diag.kta
                    for enc, diag in diagnostics.items()
                    if diag.kta is not None
                } or None
                if full_x.shape[0] >= 4:
                    geometry = {enc: diag.geometry for enc, diag in diagnostics.items()}

    qubit_sweep: QubitSweepResult | None = None
    if (
        sweep_qubit_budget
        and original_x is not None
        and profile.intrinsic_dimension is not None
    ):
        labels_for_sweep: list[int] | None = None
        if (
            labels is not None
            and len(labels) == original_x.shape[0]
            and len(set(labels)) >= 2
        ):
            labels_for_sweep = labels
        qubit_sweep = run_qubit_budget_sweep(
            original_x,
            labels=labels_for_sweep,
            fdase_columns=profile.selected_columns,
            q_star=profile.qubit_budget,
            d2=profile.intrinsic_dimension,
            q_max=sweep_q_max,
            max_samples=sweep_n_samples,
        )

    return CompareEmbeddingsResult(
        results=results,
        profile=profile,
        recommended=recommended,
        reason=reason,
        context=ctx,
        kta_by_encoding=kta_scores,
        kernel_geometry_by_encoding=geometry,
        labels=labels,
        feature_bounds=bounds,
        sample_row=np.asarray(data_arr, dtype=float),
        feature_optimization=feature_optimization,
        feature_column_names=feature_column_names,
        qubit_sweep=qubit_sweep,
    )


def format_comparison_report(
    compare_result: CompareEmbeddingsResult,
) -> str:
    """Formata relatório de comparação."""
    cr = compare_result
    lines = [
        "=== Perfil do dado ===",
        f"  Amostras: {cr.profile.n_samples}, Features: {cr.profile.n_features}",
        f"  Binário: {cr.profile.is_binary}, Categórico: {cr.profile.is_categorical}, "
        f"Contínuo: {cr.profile.is_continuous}",
        f"  Descrição: {cr.profile.description}",
        "",
    ]

    if cr.profile.intrinsic_dimension is not None:
        n_used = effective_n_features(cr.profile)
        e_orig = cr.profile.embedding_dimension or cr.profile.n_features
        cols = cr.profile.selected_columns or []
        names = cr.profile.selected_feature_names
        if names:
            col_txt = ", ".join(names)
        else:
            col_txt = ", ".join(str(c) for c in cols)
        lines.extend([
            "=== Orçamento fractal (D2 / FD-ASE) ===",
            f"  E original: {e_orig} colunas",
            f"  D2 ≈ {cr.profile.intrinsic_dimension:.3f}",
            f"  q* = max(2, ceil(D2)) = {cr.profile.qubit_budget}",
            f"  Colunas FD-ASE ({len(cols)}): [{col_txt}]",
            f"  Largura efetiva: {n_used} (recorte {'aplicado' if cr.profile.fractal_selection_applied else 'não aplicado'})",
            (
                "  Recorte aplicado: KTA e kernel-alive abaixo são neste subconjunto."
                if cr.profile.fractal_selection_applied
                else "  Recorte NÃO aplicado: KTA e kernel-alive abaixo são nas colunas originais; a seleção acima é aviso."
            ),
            "",
        ])

    if cr.labels is not None:
        n_cls = len(set(cr.labels))
        lines.extend([
            "=== Labels detectados ===",
            f"  {len(cr.labels)} amostras com rótulo ({n_cls} classes distintas). "
            "Ranking abaixo usa KTA (Kernel-Target Alignment) quando disponível.",
            "",
        ])

    if cr.feature_optimization is not None:
        n_orig = len(cr.feature_column_names) if cr.feature_column_names else cr.profile.n_features
        lines.extend(
            format_feature_optimization_section(
                cr.feature_optimization,
                column_names=cr.feature_column_names,
                n_original_features=n_orig,
            )
        )

    if cr.sample_row is not None and cr.results:
        lines.append("=== Pré-processamento matemático (amostra simulada) ===")
        for res in cr.results:
            pp = preprocess_for_encoding(
                cr.sample_row,
                res.encoding_type,
                feature_bounds=cr.feature_bounds,
            )
            lines.extend(format_preprocess_section(res.encoding_type, pp))
        lines.append("")

    lines.extend([
        "=== Recomendação detalhada (dado + problema QML) ===",
        f"  Encoding sugerido (heurística): {cr.recommended.value}",
        f"  {cr.reason}",
        "",
    ])

    if cr.kta_by_encoding:
        best_kta = max(cr.kta_by_encoding.items(), key=lambda kv: kv[1])
        lines.extend([
            f"  Melhor KTA observado: {best_kta[0].value} (KTA={best_kta[1]:.4f})",
            "",
        ])

    lines.extend(
        format_encoding_ranking_section(
            cr.profile,
            cr.recommended,
            cr.reason,
            cr.results,
            cr.context,
            kta_by_encoding=cr.kta_by_encoding,
            kernel_alive_by_encoding=(
                {enc: geo.kernel_alive for enc, geo in cr.kernel_geometry_by_encoding.items()}
                if cr.kernel_geometry_by_encoding
                else None
            ),
        )
    )
    lines.extend(format_kernel_alive_section(cr.kernel_geometry_by_encoding))
    lines.extend(format_qubit_sweep_section(cr.qubit_sweep))
    lines.extend(format_measurements_note_section(cr.results))
    lines.extend(format_simulability_section(cr.results, iqp_pairwise="all"))
    lines.append("=== Trade-offs por tipo de encoding ===")
    for enc, text in get_encoding_tradeoffs().items():
        lines.append(f"  {enc.value}: {text}")
    return "\n".join(lines)


def compare_embeddings_report(
    data: np.ndarray | list[float] | list[list[float]] | str | Path,
    n_qubits: int | None = None,
    shots: int = 1024,
    *,
    task: str | None = None,
    algorithm: str | None = None,
    problem_description: str | None = None,
    labels: list[int] | None = None,
    optimize_features: bool = False,
    optimization_encoding: EncodingType | None = None,
    feature_column_names: list[str] | None = None,
    apply_fractal_budget: bool = True,
    sweep_qubit_budget: bool = True,
) -> str:
    """
    Compara todos os encodings no dado: simula cada um e retorna relatório
    com perfil, recomendação, KTA (se labels), kernel-alive (N≥4), sweep q e trade-offs.
    """
    if isinstance(data, (str, Path)):
        data_input: np.ndarray | list[float] | str | Path = data
    else:
        data_input = np.asarray(data)
        if data_input.size == 0:
            data_input = np.array([0.1, 0.2, 0.3])
    cr = compare_embeddings(
        data_input,
        n_qubits=n_qubits,
        shots=shots,
        task=task,
        algorithm=algorithm,
        problem_description=problem_description,
        labels=labels,
        optimize_features=optimize_features,
        optimization_encoding=optimization_encoding,
        feature_column_names=feature_column_names,
        apply_fractal_budget=apply_fractal_budget,
        sweep_qubit_budget=sweep_qubit_budget,
    )
    return format_comparison_report(cr)


def explain_tradeoffs() -> str:
    """Explica os trade-offs entre amplitude, angle, basis, data re-uploading e feature maps customizados."""
    return "\n".join(
        f"- {enc.value}: {text}" for enc, text in get_encoding_tradeoffs().items()
    )
