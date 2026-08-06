"""Módulo de computação quântica com Qiskit."""

from llama_qiskit_agents.quantum.circuits import run_bell_circuit, run_simple_circuit
from llama_qiskit_agents.quantum.encodings import (
    EncodingType,
    build_encoding_circuit,
)
from llama_qiskit_agents.quantum.data_analysis import (
    infer_data_profile,
    load_csv,
    recommend_encoding,
)
from llama_qiskit_agents.quantum.problem_context import (
    MLTask,
    ProblemContext,
    infer_problem_context,
    scenario_guide_when_unspecified,
)
from llama_qiskit_agents.quantum.simulate import (
    compare_embeddings,
    compare_embeddings_report,
    explain_tradeoffs,
    format_comparison_report,
    simulate_encoding_circuit,
)

__all__ = [
    "run_bell_circuit",
    "run_simple_circuit",
    "EncodingType",
    "build_encoding_circuit",
    "infer_data_profile",
    "load_csv",
    "recommend_encoding",
    "MLTask",
    "ProblemContext",
    "infer_problem_context",
    "scenario_guide_when_unspecified",
    "compare_embeddings",
    "compare_embeddings_report",
    "explain_tradeoffs",
    "format_comparison_report",
    "simulate_encoding_circuit",
]
