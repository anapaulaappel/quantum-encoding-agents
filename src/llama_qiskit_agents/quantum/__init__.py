"""Módulo de computação quântica com Qiskit."""

from llama_qiskit_agents.quantum.circuits import run_bell_circuit, run_simple_circuit
from llama_qiskit_agents.quantum.encodings import (
    EncodingType,
    build_encoding_circuit,
)
from llama_qiskit_agents.quantum.data_analysis import (
    DataProfile,
    infer_data_profile,
    load_csv,
    load_csv_from_string,
    load_csv_from_string_with_labels,
    feature_names_from_csv_text,
    recommend_encoding,
    get_encoding_tradeoffs,
    effective_n_features,
)
from llama_qiskit_agents.quantum.fractal import (
    FDASE,
    FractalBudget,
    estimate_fractal_budget,
)
from llama_qiskit_agents.quantum.qubit_sweep import (
    QubitSweepResult,
    pca_variance_qubits,
    run_qubit_budget_sweep,
)
from llama_qiskit_agents.quantum.hardware_profile import (
    HardwareProfile,
    NISQ_GATE_ERROR_THRESHOLD,
)
from llama_qiskit_agents.quantum.problem_context import (
    MLTask,
    ProblemContext,
    infer_problem_context,
    scenario_guide_when_unspecified,
)
from llama_qiskit_agents.quantum.explanation import (
    detect_language,
    build_natural_explanation,
    generate_qiskit_code,
)
from llama_qiskit_agents.quantum.visualization import (
    simulate_statevector,
    render_bloch_sphere,
    bloch_caption,
)
from llama_qiskit_agents.quantum.kernel import (
    compute_kernel,
    compute_kernel_diagnostics_by_encoding,
    compute_kernel_matrix,
    compute_kta_by_encoding,
    kernel_is_alive,
    kernel_stats,
    render_kernel_heatmap,
    kernel_caption,
    score_kernel_geometry,
    KernelGeometry,
    KernelResult,
)
from llama_qiskit_agents.quantum.preprocessing import (
    FeatureBounds,
    PreprocessResult,
    preprocess_for_encoding,
)
from llama_qiskit_agents.quantum.encoding_optimization import (
    FeatureEncodingPlan,
    FeatureOptimizationResult,
    optimize_feature_encoding,
    pick_encoding_for_optimization,
    format_feature_optimization_section,
)
from llama_qiskit_agents.quantum.simulate import (
    CompareEmbeddingsResult,
    SimulationResult,
    compare_embeddings,
    compare_embeddings_report,
    explain_tradeoffs,
    format_comparison_report,
    simulate_encoding_circuit,
)

__all__ = [
    # circuits
    "run_bell_circuit",
    "run_simple_circuit",
    # encodings
    "EncodingType",
    "build_encoding_circuit",
    # data analysis
    "DataProfile",
    "infer_data_profile",
    "load_csv",
    "load_csv_from_string",
    "load_csv_from_string_with_labels",
    "feature_names_from_csv_text",
    "recommend_encoding",
    "get_encoding_tradeoffs",
    "effective_n_features",
    "FDASE",
    "FractalBudget",
    "estimate_fractal_budget",
    "QubitSweepResult",
    "pca_variance_qubits",
    "run_qubit_budget_sweep",
    # hardware profile
    "HardwareProfile",
    "NISQ_GATE_ERROR_THRESHOLD",
    # problem context
    "MLTask",
    "ProblemContext",
    "infer_problem_context",
    "scenario_guide_when_unspecified",
    # explanation & code generation
    "detect_language",
    "build_natural_explanation",
    "generate_qiskit_code",
    # visualization
    "simulate_statevector",
    "render_bloch_sphere",
    "bloch_caption",
    # kernel
    "compute_kernel",
    "compute_kernel_matrix",
    "compute_kernel_diagnostics_by_encoding",
    "compute_kta_by_encoding",
    "kernel_is_alive",
    "kernel_stats",
    "render_kernel_heatmap",
    "kernel_caption",
    "score_kernel_geometry",
    "KernelGeometry",
    "KernelResult",
    "FeatureBounds",
    "PreprocessResult",
    "preprocess_for_encoding",
    "FeatureEncodingPlan",
    "FeatureOptimizationResult",
    "optimize_feature_encoding",
    "pick_encoding_for_optimization",
    "format_feature_optimization_section",
    # simulation
    "CompareEmbeddingsResult",
    "SimulationResult",
    "compare_embeddings",
    "compare_embeddings_report",
    "explain_tradeoffs",
    "format_comparison_report",
    "simulate_encoding_circuit",
]
