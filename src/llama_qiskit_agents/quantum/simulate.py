"""Simulação de circuitos de encoding e comparação entre estratégias."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from llama_qiskit_agents.quantum.data_analysis import (
    DataProfile,
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
    # Statevector puro antes da medição (apenas quando save_statevector=True)
    statevector: np.ndarray | None = field(default=None, repr=False)


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

    # Capturar statevector separadamente se solicitado
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
) -> tuple[list[SimulationResult], DataProfile, EncodingType, str, ProblemContext]:
    """
    Compara múltiplos encodings no mesmo dado: simula cada um e retorna
    resultados, perfil do dado, encoding recomendado e justificativa.
    Aceita CSV (path): carrega, analisa e usa a primeira linha como amostra para simulação.
    """
    if encoding_types is None:
        encoding_types = list(EncodingType)
    # CSV: carregar e usar primeira linha como amostra para simulação
    if isinstance(data, (str, Path)):
        p = Path(data) if isinstance(data, str) else data
        if str(p).lower().endswith(".csv") and p.exists():
            x = load_csv(p)
            profile = infer_data_profile(x)
            data_arr = x[0] if x.ndim > 1 else x.flatten()
        else:
            profile = infer_data_profile(data)
            data_arr = np.array([0.1, 0.2, 0.3])
    else:
        x = np.asarray(data)
        profile = infer_data_profile(x)
        if x.ndim > 1 and x.shape[0] > 0:
            data_arr = x[0]
        else:
            data_arr = x.flatten() if hasattr(x, "__len__") else np.array([0.0])
    if len(data_arr) == 0:
        data_arr = np.array([0.1, 0.2, 0.3])
    recommended, reason, ctx = recommend_encoding(
        profile,
        task=task,
        algorithm=algorithm,
        problem_description=problem_description,
    )
    results: list[SimulationResult] = []
    for enc in encoding_types:
        try:
            qc = build_encoding_circuit(enc, data_arr, n_qubits=n_qubits)
            res = simulate_encoding_circuit(qc, enc, shots=shots)
            results.append(res)
        except Exception:
            continue
    return results, profile, recommended, reason, ctx


def format_comparison_report(
    results: list[SimulationResult],
    profile: DataProfile,
    recommended: EncodingType,
    reason: str,
    problem_context: ProblemContext | None = None,
) -> str:
    """Formata um relatório em texto para o agente/usuário."""
    lines = [
        "=== Perfil do dado ===",
        f"  Amostras: {profile.n_samples}, Features: {profile.n_features}",
        f"  Binário: {profile.is_binary}, Categórico: {profile.is_categorical}, Contínuo: {profile.is_continuous}",
        f"  Descrição: {profile.description}",
        "",
        "=== Recomendação detalhada (dado + problema QML) ===",
        f"  Encoding sugerido: {recommended.value}",
        f"  {reason}",
        "",
    ]
    lines.extend(
        format_encoding_ranking_section(
            profile, recommended, reason, results, problem_context
        )
    )
    lines.extend(format_measurements_note_section(results))
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
) -> str:
    """
    Compara todos os encodings no dado: simula cada um e retorna relatório
    com perfil, recomendação, resultados e trade-offs. Aceita path de CSV.
    """
    if isinstance(data, (str, Path)):
        data_input: np.ndarray | list[float] | str | Path = data
    else:
        data_input = np.asarray(data)
        if data_input.size == 0:
            data_input = np.array([0.1, 0.2, 0.3])
    results, profile, recommended, reason, ctx = compare_embeddings(
        data_input,
        n_qubits=n_qubits,
        shots=shots,
        task=task,
        algorithm=algorithm,
        problem_description=problem_description,
    )
    return format_comparison_report(results, profile, recommended, reason, ctx)


def explain_tradeoffs() -> str:
    """Explica os trade-offs entre amplitude, angle, basis, data re-uploading e feature maps customizados."""
    return "\n".join(
        f"- {enc.value}: {text}" for enc, text in get_encoding_tradeoffs().items()
    )
