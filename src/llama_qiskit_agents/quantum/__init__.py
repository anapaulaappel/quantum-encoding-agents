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
    recommend_encoding,
    get_encoding_tradeoffs,
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
    compute_kernel_matrix,
    kernel_stats,
    render_kernel_heatmap,
    kernel_caption,
    KernelResult,
)
from llama_qiskit_agents.quantum.simulate import (
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
    "recommend_encoding",
    "get_encoding_tradeoffs",
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
    "kernel_stats",
    "render_kernel_heatmap",
    "kernel_caption",
    "KernelResult",
    # simulation
    "SimulationResult",
    "compare_embeddings",
    "compare_embeddings_report",
    "explain_tradeoffs",
    "format_comparison_report",
    "simulate_encoding_circuit",
]
