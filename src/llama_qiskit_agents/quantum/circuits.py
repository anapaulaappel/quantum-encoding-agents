"""Circuitos quânticos de exemplo com Qiskit."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def run_simple_circuit(n_qubits: int = 2, shots: int = 1024) -> dict[str, int]:
    """Executa um circuito simples (H no primeiro qubit) no simulador."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.measure_all()
    sim = AerSimulator()
    circ = transpile(qc, sim)
    job = sim.run(circ, shots=shots)
    return dict(job.result().get_counts())


def run_bell_circuit(shots: int = 1024) -> dict[str, int]:
    """Executa o circuito de Bell (estado emaranhado) no simulador."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    sim = AerSimulator()
    circ = transpile(qc, sim)
    job = sim.run(circ, shots=shots)
    return dict(job.result().get_counts())
