"""
Execução de circuitos de encoding em hardware IBM Quantum (Qiskit Runtime).

Opcional: pip install -e ".[ibm]"
Autenticação: QISKIT_IBM_TOKEN ou `ibm-quantum-login` (salvo localmente).

Uso típico:
  stats = transpile_encoding_circuit(qc, backend_name="ibm_fez")
  result = run_encoding_circuit(qc, backend_name="ibm_fez", shots=512)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile

from llama_qiskit_agents.quantum.encodings import EncodingType, build_encoding_circuit


class IBMHardwareError(RuntimeError):
    """Erro de configuração ou execução no IBM Quantum."""


@dataclass
class TranspileStats:
    backend_name: str
    depth: int
    size: int
    num_qubits: int
    two_qubit_gates: int
    layout: list[int] = field(default_factory=list)

    def summary(self) -> str:
        lay = f", layout={self.layout}" if self.layout else ""
        return (
            f"{self.backend_name}: depth={self.depth}, size={self.size}, "
            f"qubits={self.num_qubits}, 2q={self.two_qubit_gates}{lay}"
        )


@dataclass
class HardwareRunResult:
    backend_name: str
    job_id: str
    shots: int
    counts: dict[str, int]
    elapsed_sec: float
    metadata: dict[str, str | int | float] = field(default_factory=dict)


def _require_ibm_runtime():
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService  # noqa: F401
    except ImportError as exc:
        raise IBMHardwareError(
            "qiskit-ibm-runtime não instalado. Use: pip install -e '.[ibm]'"
        ) from exc


def resolve_backend_name(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("IBM_QUANTUM_BACKEND", "").strip()
    if env:
        return env
    return "ibm_fez"


def get_ibm_backend(backend_name: str | None = None):
    """Retorna backend IBM (real device ou simulator se nome contiver 'sim')."""
    _require_ibm_runtime()
    from qiskit_ibm_runtime import QiskitRuntimeService

    name = resolve_backend_name(backend_name)
    token = os.environ.get("QISKIT_IBM_TOKEN", "").strip() or None
    service = QiskitRuntimeService(token=token) if token else QiskitRuntimeService()
    try:
        return service.backend(name)
    except Exception as exc:
        available = []
        try:
            for b in service.backends(operational=True, simulator=False):
                pending = getattr(b.status(), "pending_jobs", "?")
                available.append(f"{b.name}(pending={pending})")
        except Exception:
            pass
        hint = f" Disponíveis agora: {', '.join(available)}." if available else ""
        raise IBMHardwareError(
            f"Backend '{name}' indisponível neste plano.{hint} "
            "Passe --backend ou IBM_QUANTUM_BACKEND."
        ) from exc


def _two_qubit_count(circuit: QuantumCircuit) -> int:
    total = 0
    for _, count in circuit.count_ops().items():
        if len(_) == 2 or (isinstance(_, str) and _.lower() in {"cx", "cz", "ecr", "rxx", "ryy", "rzz"}):
            total += count
    # count_ops keys are Instruction names as str in recent Qiskit
    ops = circuit.count_ops()
    two_q = sum(v for k, v in ops.items() if k in {"cx", "cy", "cz", "ecr", "rxx", "ryy", "rzz", "iswap"})
    return two_q


def transpile_encoding_circuit(
    circuit: QuantumCircuit,
    *,
    backend_name: str | None = None,
    optimization_level: int = 1,
) -> tuple[QuantumCircuit, TranspileStats]:
    """Transpila para o backend alvo (não consome tempo de QPU)."""
    backend = get_ibm_backend(backend_name)
    name = backend.name
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
        isa = pm.run(circuit)
    except Exception:
        isa = transpile(circuit, backend=backend, optimization_level=optimization_level)

    layout: list[int] = []
    if hasattr(isa, "layout") and isa.layout is not None:
        try:
            layout = list(isa.layout.initial_index_layout())  # type: ignore[attr-defined]
        except Exception:
            pass

    return isa, TranspileStats(
        backend_name=name,
        depth=isa.depth(),
        size=isa.size(),
        num_qubits=isa.num_qubits,
        two_qubit_gates=_two_qubit_count(isa),
        layout=layout,
    )


def _counts_from_pub(pub) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        if hasattr(pub, "join_data"):
            data = pub.join_data()
            if hasattr(data, "get_counts"):
                return dict(data.get_counts())
        if hasattr(pub, "data"):
            for item in pub.data:
                if hasattr(item, "get_counts"):
                    return dict(item.get_counts())
    except Exception:
        return {}
    return counts


def run_encoding_circuits(
    circuits: list[QuantumCircuit],
    *,
    backend_name: str | None = None,
    shots: int = 512,
    optimization_level: int = 1,
) -> list[HardwareRunResult]:
    """Submete vários circuitos no mesmo job SamplerV2 (uma fila)."""
    if not circuits:
        return []
    _require_ibm_runtime()
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    backend = get_ibm_backend(backend_name)
    isas: list[QuantumCircuit] = []
    for qc in circuits:
        isa, _ = transpile_encoding_circuit(
            qc,
            backend_name=backend.name,
            optimization_level=optimization_level,
        )
        isas.append(isa)

    t0 = time.perf_counter()
    sampler = Sampler(mode=backend)
    job = sampler.run(isas, shots=shots)
    result = job.result()
    elapsed = time.perf_counter() - t0

    raw_id = getattr(job, "job_id", None)
    if callable(raw_id):
        raw_id = raw_id()
    job_id = str(raw_id if raw_id is not None else job)

    outcomes: list[HardwareRunResult] = []
    for i, isa in enumerate(isas):
        counts = _counts_from_pub(result[i])
        outcomes.append(
            HardwareRunResult(
                backend_name=backend.name,
                job_id=job_id,
                shots=shots,
                counts=counts,
                elapsed_sec=elapsed if i == 0 else 0.0,
                metadata={"depth": isa.depth(), "size": isa.size(), "pub": i},
            )
        )
    return outcomes


def run_encoding_circuit(
    circuit: QuantumCircuit,
    *,
    backend_name: str | None = None,
    shots: int = 512,
    optimization_level: int = 1,
) -> HardwareRunResult:
    """Executa circuito no backend IBM (consome cota / fila)."""
    return run_encoding_circuits(
        [circuit],
        backend_name=backend_name,
        shots=shots,
        optimization_level=optimization_level,
    )[0]


def build_sample_encoding_circuit(
    row: list[float] | np.ndarray,
    encoding_type: EncodingType,
    *,
    feature_bounds=None,
) -> QuantumCircuit:
    """Circuito de encoding + measure_all para hardware."""
    qc = build_encoding_circuit(
        encoding_type,
        np.asarray(row, dtype=float),
        feature_bounds=feature_bounds,
    )
    if qc.num_clbits == 0:
        qc.measure_all()
    return qc
