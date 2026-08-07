"""Encodings quânticos para embedding de dados clássicos em circuitos Qiskit."""

from enum import Enum
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation


class EncodingType(str, Enum):
    """Tipos de encoding suportados."""

    AMPLITUDE = "amplitude"
    ANGLE = "angle"
    DENSE_ANGLE = "dense_angle"
    BASIS = "basis"
    DATA_REUPLOADING = "data_reuploading"
    CUSTOM_FEATURE_MAP = "custom_feature_map"


def _ensure_1d_normalized(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """Garante vetor 1D normalizado com tamanho 2^n_qubits."""
    x = np.asarray(x, dtype=complex).flatten()
    n_amplitudes = 2**n_qubits
    if len(x) > n_amplitudes:
        x = x[:n_amplitudes]
    elif len(x) < n_amplitudes:
        x = np.pad(x, (0, n_amplitudes - len(x)))
    norm = np.linalg.norm(x)
    if norm > 1e-10:
        x = x / norm
    return x


def amplitude_encoding(
    data: np.ndarray,
    n_qubits: int,
    name: str = "amplitude",
) -> QuantumCircuit:
    """
    Amplitude encoding: dados normalizados nas amplitudes do estado quântico.
    Requer 2^n amplitudes para n qubits; poucos qubits, muitas portas.
    """
    x = _ensure_1d_normalized(data, n_qubits)
    qc = QuantumCircuit(n_qubits, name=name)
    # StatePreparation prepara o estado a partir do vetor de amplitudes
    qc.append(StatePreparation(x), range(n_qubits))
    return qc


def angle_encoding(
    data: np.ndarray,
    n_qubits: int | None = None,
    name: str = "angle",
) -> QuantumCircuit:
    """
    Angle encoding: cada feature vira ângulo de rotação (Ry).
    n features → n qubits; poucas portas, mais qubits.
    """
    x = np.asarray(data, dtype=float).flatten()
    if n_qubits is None:
        n_qubits = max(1, len(x))
    n_qubits = max(n_qubits, 1)
    if len(x) < n_qubits:
        x = np.pad(x, (0, n_qubits - len(x)))
    else:
        x = x[:n_qubits]
    qc = QuantumCircuit(n_qubits, name=name)
    for i, val in enumerate(x):
        qc.ry(val, i)
    return qc


def dense_angle_encoding(
    data: np.ndarray,
    n_qubits: int | None = None,
    name: str = "dense_angle",
) -> QuantumCircuit:
    """
    Dense angle encoding: empacota 2 features por qubit usando Ry(x[2i]) · Rz(x[2i+1]).
    n features → ⌈n/2⌉ qubits; profundidade 2 — mais eficiente que angle quando features > qubits.
    Identificado como encoding faltante no survey Sammartino (arXiv:2606.05387, 2026).
    """
    x = np.asarray(data, dtype=float).flatten()
    if n_qubits is None:
        n_qubits = max(1, int(np.ceil(len(x) / 2)))
    n_qubits = max(n_qubits, 1)
    # Pad para ter exatamente 2 * n_qubits features (pares completos)
    n_needed = 2 * n_qubits
    if len(x) < n_needed:
        x = np.pad(x, (0, n_needed - len(x)))
    else:
        x = x[:n_needed]
    qc = QuantumCircuit(n_qubits, name=name)
    for i in range(n_qubits):
        qc.ry(x[2 * i],     i)  # primeira feature do par → rotação Y
        qc.rz(x[2 * i + 1], i)  # segunda feature do par → rotação Z
    return qc


def basis_encoding(
    data: np.ndarray,
    n_qubits: int | None = None,
    name: str = "basis",
) -> QuantumCircuit:
    """
    Basis encoding: cada amostra é um bit string; cada bit → |0⟩ ou |1⟩.
    Adequado para dados binários/categóricos.
    """
    x = np.asarray(data).flatten()
    # Binarizar: 0 ou não-zero → 0 ou 1
    binary = (np.asarray(x, dtype=float) != 0).astype(int)
    if n_qubits is None:
        n_qubits = max(1, len(binary))
    if len(binary) < n_qubits:
        binary = np.pad(binary, (0, n_qubits - len(binary)))
    else:
        binary = binary[:n_qubits]
    qc = QuantumCircuit(n_qubits, name=name)
    for i, b in enumerate(binary):
        if b:
            qc.x(i)
    return qc


def data_reuploading_encoding(
    data: np.ndarray,
    n_layers: int = 2,
    n_qubits: int | None = None,
    name: str = "reupload",
) -> QuantumCircuit:
    """
    Data re-uploading: múltiplas camadas de angle encoding (repetir dados).
    Aumenta expressibilidade sem aumentar qubits.
    """
    x = np.asarray(data, dtype=float).flatten()
    if n_qubits is None:
        n_qubits = max(1, len(x))
    n_qubits = max(n_qubits, 1)
    if len(x) < n_qubits:
        x = np.pad(x, (0, n_qubits - len(x)))
    else:
        x = x[:n_qubits]
    qc = QuantumCircuit(n_qubits, name=name)
    for _ in range(n_layers):
        for i, val in enumerate(x):
            qc.ry(val, i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    return qc


def custom_feature_map_encoding(
    data: np.ndarray,
    n_qubits: int | None = None,
    n_layers: int = 1,
    name: str = "custom_fm",
) -> QuantumCircuit:
    """
    Feature map customizado: camadas de rotações (Ry/Rz) + entrelaçamento ZZ.
    Boa expressibilidade e controle de entrelaçamento.
    """
    x = np.asarray(data, dtype=float).flatten()
    if n_qubits is None:
        n_qubits = max(1, len(x))
    n_qubits = max(n_qubits, 1)
    if len(x) < n_qubits:
        x = np.pad(x, (0, n_qubits - len(x)))
    else:
        x = x[:n_qubits]
    qc = QuantumCircuit(n_qubits, name=name)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.h(i)
            qc.rz(x[i] if i < len(x) else 0.0, i)
            qc.ry(x[i] if i < len(x) else 0.0, i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.cz(i, j)
    return qc


def build_encoding_circuit(
    encoding_type: EncodingType,
    data: np.ndarray,
    n_qubits: int | None = None,
    **kwargs: Any,
) -> QuantumCircuit:
    """
    Constrói o circuito Qiskit para o tipo de encoding escolhido.
    """
    data = np.asarray(data)
    if n_qubits is None and encoding_type == EncodingType.AMPLITUDE:
        n_qubits = max(1, int(np.ceil(np.log2(max(1, data.size)))))
    elif n_qubits is None and encoding_type == EncodingType.DENSE_ANGLE:
        n_qubits = max(1, int(np.ceil(data.size / 2)))
    elif n_qubits is None:
        n_qubits = max(1, data.size) if data.size else 1

    if encoding_type == EncodingType.AMPLITUDE:
        return amplitude_encoding(data, n_qubits, **kwargs)
    if encoding_type == EncodingType.ANGLE:
        return angle_encoding(data, n_qubits=n_qubits, **kwargs)
    if encoding_type == EncodingType.DENSE_ANGLE:
        return dense_angle_encoding(data, n_qubits=n_qubits, **kwargs)
    if encoding_type == EncodingType.BASIS:
        return basis_encoding(data, n_qubits=n_qubits, **kwargs)
    if encoding_type == EncodingType.DATA_REUPLOADING:
        return data_reuploading_encoding(data, n_qubits=n_qubits, **kwargs)
    if encoding_type == EncodingType.CUSTOM_FEATURE_MAP:
        return custom_feature_map_encoding(data, n_qubits=n_qubits, **kwargs)
    raise ValueError(f"Encoding não suportado: {encoding_type}")
