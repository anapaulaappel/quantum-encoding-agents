"""
Visualização de estados quânticos — esfera de Bloch por qubit.

Fluxo:
  1. simulate_statevector() roda o circuito no StatevectorSimulator (sem medição)
  2. render_bloch_sphere() converte o statevector em PNG base64 via matplotlib
  3. O PNG é retornado no campo bloch_sphere_b64 de ExplainResponse

Por que StatevectorSimulator e não AerSimulator?
  - Medição colapsaria o estado — precisamos do estado puro antes de medir.
  - StatevectorSimulator retorna o vetor de amplitudes complexas diretamente.
  - Não exige cópia de circuito nem add_measure_all().
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import StatevectorSimulator

if TYPE_CHECKING:
    pass


def simulate_statevector(circuit: QuantumCircuit) -> np.ndarray | None:
    """
    Executa o circuito no StatevectorSimulator e retorna o vetor de estado complexo.
    Retorna None se a simulação falhar (circuito muito grande, gates não suportados etc.).
    """
    try:
        # StatevectorSimulator não aceita medições clássicas
        qc = circuit.copy()
        # Remove medições se existirem
        qc.remove_final_measurements(inplace=True)

        sim = StatevectorSimulator()
        transpiled = transpile(qc, sim)
        job = sim.run(transpiled)
        result = job.result()
        sv = np.array(result.get_statevector(), dtype=complex)
        return sv
    except Exception:
        return None


def render_bloch_sphere(
    statevector: np.ndarray,
    encoding_name: str,
    lang: str = "pt",
) -> str | None:
    """
    Renderiza a esfera de Bloch para cada qubit do statevector.
    Retorna a imagem como string base64 PNG, ou None se falhar.

    Para n qubits, gera n esferas lado a lado em uma única figura.
    Limita a 6 qubits para manter a imagem legível.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend sem display — compatível com containers
        import matplotlib.pyplot as plt
        from qiskit.visualization import plot_bloch_multivector
        from qiskit.quantum_info import Statevector

        sv = Statevector(statevector)
        n_qubits = int(np.log2(len(statevector)))

        # Limita visualização a 6 qubits (acima fica ilegível)
        if n_qubits > 6:
            return None

        title = (
            f"{encoding_name} — estado quântico por qubit"
            if lang == "pt"
            else f"{encoding_name} — quantum state per qubit"
        )

        fig = plot_bloch_multivector(sv, title=title, figsize=(3 * n_qubits, 3.5))

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception:
        return None


def bloch_caption(n_qubits: int, encoding_name: str, lang: str = "pt") -> str:
    """Legenda explicativa para acompanhar a imagem da esfera de Bloch."""
    if lang == "en":
        return (
            f"Bloch sphere representation of the {encoding_name} encoding state "
            f"({n_qubits} qubit{'s' if n_qubits != 1 else ''}). "
            "Each sphere shows where the qubit state 'points' on the unit sphere: "
            "north pole = |0⟩, south pole = |1⟩, equator = superposition. "
            "This is the quantum state before measurement — measuring collapses "
            "each qubit to |0⟩ or |1⟩ with probability determined by the shown angle."
        )
    return (
        f"Representação na esfera de Bloch do estado codificado por {encoding_name} "
        f"({n_qubits} qubit{'s' if n_qubits != 1 else ''}). "
        "Cada esfera mostra para onde o estado do qubit 'aponta' na esfera unitária: "
        "polo norte = |0⟩, polo sul = |1⟩, equador = superposição. "
        "Este é o estado quântico antes da medição — medir colapsa cada qubit para "
        "|0⟩ ou |1⟩ com probabilidade determinada pelo ângulo mostrado."
    )
