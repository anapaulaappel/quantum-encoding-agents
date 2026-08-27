"""
Pré-processamento matemático antes de encodings quânticos.

Normalização / scaling evita degenerescência por periodicidade de Ry/Rz e
alinha com boas práticas (LaRose 2020; preprints PQFM; revisão José Victor).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from llama_qiskit_agents.quantum.encodings import EncodingType

# Intervalo seguro para rotações Ry/Rz (evita colapso por 2π-periodicidade em [0, π])
ANGLE_SCALE_LOW: float = 0.0
ANGLE_SCALE_HIGH: float = float(np.pi)

# Dois ângulos mais próximos abaixo deste limiar → aviso de degenerescência
DEGENERACY_ANGLE_TOLERANCE: float = 1e-3

# Resolução angular típica em hardware NISQ (ordem de grandeza)
DEFAULT_HARDWARE_ANGLE_RESOLUTION: float = 1e-4


@dataclass
class FeatureBounds:
    """Min/max por feature (coluna), ajustados no conjunto de treino / CSV."""

    col_min: np.ndarray
    col_max: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> FeatureBounds:
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return cls(col_min=x.min(axis=0), col_max=x.max(axis=0))

    def transform_column(self, col_idx: int, value: float) -> float:
        lo = float(self.col_min[col_idx])
        hi = float(self.col_max[col_idx])
        if hi - lo < 1e-12:
            return (ANGLE_SCALE_LOW + ANGLE_SCALE_HIGH) / 2.0
        t = (value - lo) / (hi - lo)
        return ANGLE_SCALE_LOW + t * (ANGLE_SCALE_HIGH - ANGLE_SCALE_LOW)


@dataclass
class PreprocessResult:
    """Dado transformado + metadados para relatório / agente."""

    data: np.ndarray
    transform_description: str
    warnings: list[str] = field(default_factory=list)
    scaled_angles: np.ndarray | None = None


def _needs_angle_scaling(encoding_type: EncodingType) -> bool:
    return encoding_type in {
        EncodingType.ANGLE,
        EncodingType.DENSE_ANGLE,
        EncodingType.IQP,
        EncodingType.DATA_REUPLOADING,
        EncodingType.CUSTOM_FEATURE_MAP,
    }


def _scale_vector_to_angles(
    x: np.ndarray,
    bounds: FeatureBounds | None,
) -> tuple[np.ndarray, str]:
    x = np.asarray(x, dtype=float).flatten()
    if bounds is not None and len(bounds.col_min) >= len(x):
        scaled = np.array(
            [bounds.transform_column(i, v) for i, v in enumerate(x)],
            dtype=float,
        )
        desc = (
            f"Min-max por feature no dataset → [{ANGLE_SCALE_LOW:.4f}, {ANGLE_SCALE_HIGH:.4f}] rad"
        )
    else:
        xmin, xmax = float(x.min()), float(x.max())
        if xmax - xmin < 1e-12:
            scaled = np.full_like(x, (ANGLE_SCALE_LOW + ANGLE_SCALE_HIGH) / 2.0)
            desc = f"Features constantes → ângulo médio {(ANGLE_SCALE_LOW + ANGLE_SCALE_HIGH) / 2:.4f} rad"
        else:
            scaled = ANGLE_SCALE_LOW + (x - xmin) / (xmax - xmin) * (
                ANGLE_SCALE_HIGH - ANGLE_SCALE_LOW
            )
            desc = (
                f"Min-max na amostra → [{ANGLE_SCALE_LOW:.4f}, {ANGLE_SCALE_HIGH:.4f}] rad"
            )
    return scaled, desc


def _collect_angle_warnings(
    angles: np.ndarray,
    *,
    angle_resolution: float = DEFAULT_HARDWARE_ANGLE_RESOLUTION,
) -> list[str]:
    warnings: list[str] = []
    n = len(angles)
    if n < 2:
        return warnings

    sorted_idx = np.argsort(angles)
    sorted_angles = angles[sorted_idx]

    for k in range(n - 1):
        diff = float(sorted_angles[k + 1] - sorted_angles[k])
        if diff < DEGENERACY_ANGLE_TOLERANCE:
            i, j = int(sorted_idx[k]), int(sorted_idx[k + 1])
            warnings.append(
                f"Features {i} e {j} mapeiam para ângulos muito próximos "
                f"({diff:.2e} rad) — risco de degenerescência (periodicidade Ry/Rz)."
            )

    unique_diffs = [
        float(sorted_angles[k + 1] - sorted_angles[k])
        for k in range(n - 1)
        if sorted_angles[k + 1] - sorted_angles[k] > 1e-12
    ]
    if unique_diffs:
        min_sep = min(unique_diffs)
        if min_sep < angle_resolution:
            warnings.append(
                f"Menor separação angular entre features distintas ({min_sep:.2e} rad) "
                f"pode ficar abaixo da resolução típica de hardware (~{angle_resolution:.0e} rad)."
            )
    return warnings


def preprocess_for_encoding(
    data: np.ndarray,
    encoding_type: EncodingType,
    *,
    feature_bounds: FeatureBounds | None = None,
    angle_resolution: float = DEFAULT_HARDWARE_ANGLE_RESOLUTION,
) -> PreprocessResult:
    """
    Aplica transformação clássica antes do circuito de encoding.
    """
    x = np.asarray(data, dtype=float).flatten()
    warnings: list[str] = []

    if encoding_type == EncodingType.AMPLITUDE:
        n = len(x)
        norm = np.linalg.norm(x)
        if norm > 1e-10:
            out = x / norm
            desc = "Normalização L2 (estado quântico unitário)"
        else:
            out = np.zeros(n, dtype=float)
            out[0] = 1.0
            desc = "Vetor nulo → |0⟩"
            warnings.append("Vetor de entrada com norma zero; amplitude encoding usa |0⟩.")
        return PreprocessResult(data=out, transform_description=desc, warnings=warnings)

    if encoding_type == EncodingType.BASIS:
        return PreprocessResult(
            data=x,
            transform_description="Binarização implícita (0 vs não-zero) no circuito",
            warnings=warnings,
        )

    if _needs_angle_scaling(encoding_type):
        scaled, desc = _scale_vector_to_angles(x, feature_bounds)
        warnings.extend(_collect_angle_warnings(scaled, angle_resolution=angle_resolution))
        return PreprocessResult(
            data=scaled,
            transform_description=desc,
            warnings=warnings,
            scaled_angles=scaled,
        )

    return PreprocessResult(data=x, transform_description="Sem transformação adicional", warnings=warnings)


def format_preprocess_section(
    encoding_type: EncodingType,
    result: PreprocessResult,
) -> list[str]:
    """Linhas para incluir no relatório de comparação."""
    lines = [
        f"  • {encoding_type.value}: {result.transform_description}",
    ]
    for w in result.warnings:
        lines.append(f"    ⚠ {w}")
    return lines
