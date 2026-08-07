"""Análise de dados e recomendação de estratégia de embedding quântico."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llama_qiskit_agents.quantum.encodings import EncodingType


def _rows_from_csv_reader(reader: csv.reader) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in reader:
        vals: list[float] = []
        for cell in row:
            try:
                vals.append(float(cell.strip()))
            except (ValueError, AttributeError):
                pass
        if vals:
            rows.append(vals)
    return rows


def _array_from_numeric_rows(rows: list[list[float]], source: str) -> np.ndarray:
    if not rows:
        raise ValueError(f"CSV vazio ou sem colunas numéricas: {source}")
    n_cols = max(len(r) for r in rows)
    arr = np.zeros((len(rows), n_cols))
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return arr


def load_csv_from_string(text: str) -> np.ndarray:
    """
    Parse de CSV em memória (upload HTTP). Mesmas regras que load_csv (só células numéricas).
    """
    rows = _rows_from_csv_reader(csv.reader(io.StringIO(text.strip())))
    return _array_from_numeric_rows(rows, "upload")


def load_csv(path: str | Path) -> np.ndarray:
    """
    Carrega um arquivo CSV e retorna array 2D numérico.
    Colunas não numéricas são ignoradas; usa a primeira linha como header se parecer texto.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = _rows_from_csv_reader(csv.reader(f))
    return _array_from_numeric_rows(rows, str(path))


@dataclass
class DataProfile:
    """Perfil inferido do conjunto de dados."""

    n_samples: int
    n_features: int
    is_binary: bool
    is_categorical: bool
    is_continuous: bool
    has_negative: bool
    description: str


def infer_data_profile(
    data: np.ndarray | list[Any] | str | Path,
) -> DataProfile:
    """
    Analisa o dado (array, CSV, ou descrição) e infere tipo, dimensões e características.
    - Se `data` for path de arquivo .csv: carrega e analisa.
    - Se `data` for string (não path): perfil genérico baseado em palavras-chave.
    - Se `data` for array 2D: n_samples=linhas, n_features=colunas.
    """
    # CSV: path como string ou Path
    if isinstance(data, (str, Path)):
        p = Path(data) if isinstance(data, str) else data
        if str(p).lower().endswith(".csv") and p.exists():
            x = load_csv(p)
            return _infer_profile_from_array(x, f"CSV: {p.name} ({x.shape[0]} linhas, {x.shape[1]} colunas)")
        if isinstance(data, str):
            d = data.lower()
            # Só tratar como descrição se não parecer path
            if ".csv" not in d or not Path(data).exists():
                is_binary = "binár" in d or "binary" in d or "bit" in d or "0 e 1" in d or "0 e 0" in d
                is_categorical = "categór" in d or "categorical" in d or "classe" in d or "label" in d
                is_continuous = "contínu" in d or "continuous" in d or "real" in d or "numér" in d
                return DataProfile(
                    n_samples=0,
                    n_features=0,
                    is_binary=is_binary or (not is_continuous and not is_categorical),
                    is_categorical=is_categorical,
                    is_continuous=is_continuous or (not is_binary and not is_categorical),
                    has_negative="negat" in d or "negative" in d or "real" in d,
                    description=data,
                )

    x = np.asarray(data)
    return _infer_profile_from_array(x, "Array de dados")


def _infer_profile_from_array(x: np.ndarray, description: str) -> DataProfile:
    """Infere perfil a partir de array 1D ou 2D."""
    if x.ndim == 0:
        x = np.array([[float(x)]])
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    n_samples, n_features = x.shape
    x_flat = np.asarray(x, dtype=float).flatten()
    try:
        finite = np.isfinite(x_flat)
        vals = x_flat[finite] if np.any(finite) else x_flat
        unique = np.unique(vals)
        is_binary = len(unique) <= 2 and all(
            np.isclose(v, 0) or np.isclose(v, 1) for v in unique
        )
        is_continuous = not is_binary and (
            len(unique) > 10 or np.issubdtype(x_flat.dtype, np.floating)
        )
        is_categorical = not is_continuous and len(unique) <= 20
        has_negative = np.any(x_flat < -1e-9)
    except Exception:
        is_binary = is_categorical = has_negative = False
        is_continuous = True
    desc = description if description else f"{n_samples} amostras × {n_features} features"
    return DataProfile(
        n_samples=int(n_samples),
        n_features=int(n_features),
        is_binary=bool(is_binary),
        is_categorical=bool(is_categorical),
        is_continuous=bool(is_continuous),
        has_negative=bool(has_negative),
        description=desc,
    )


def _recommend_encoding_from_data(profile: DataProfile) -> tuple[EncodingType, str]:
    """Recomendação baseada só no perfil dos dados."""
    if profile.is_binary and profile.n_features <= 16:
        return (
            EncodingType.BASIS,
            "Dados binários/categóricos com poucos bits: basis encoding usa 1 qubit por bit e é natural para classificação.",
        )
    if profile.n_features <= 4 and profile.is_continuous:
        return (
            EncodingType.ANGLE,
            "Poucos features contínuos: angle encoding é simples, poucas portas e boa para protótipos.",
        )
    if 4 < profile.n_features <= 12 and profile.is_continuous and not profile.has_negative:
        return (
            EncodingType.DENSE_ANGLE,
            f"Dense angle encoding empacota 2 features por qubit via Ry·Rz — "
            f"{profile.n_features} features em ⌈{profile.n_features}/2⌉={int(np.ceil(profile.n_features/2))} qubits "
            f"com profundidade 2. Mais eficiente que angle sem o custo do re-uploading.",
        )
    if profile.n_samples <= 1 and profile.n_features >= 4 and profile.is_continuous:
        return (
            EncodingType.AMPLITUDE,
            "Vetor único com muitos componentes: amplitude encoding compacta em log2(n) qubits (estado puro).",
        )
    if profile.is_continuous and profile.n_features <= 8:
        return (
            EncodingType.DATA_REUPLOADING,
            "Dados contínuos com dimensão moderada: data re-uploading aumenta expressibilidade sem mais qubits.",
        )
    if profile.is_continuous and 8 < profile.n_features <= 16:
        return (
            EncodingType.IQP,
            f"IQP encoding com {profile.n_features} features: H + Rz(x²) + Rzz(x·x') diagonais — "
            "base teórica forte para kernels quânticos em dimensão moderada (Havlíček et al., 2019).",
        )
    if profile.is_continuous:
        return (
            EncodingType.CUSTOM_FEATURE_MAP,
            "Feature map customizado com rotações e entrelaçamento: flexível e bom para kernels quânticos.",
        )
    return (
        EncodingType.ANGLE,
        "Caso genérico: angle encoding é um bom padrão (simples e interpretável).",
    )


def recommend_encoding(
    profile: DataProfile,
    *,
    task: str | None = None,
    algorithm: str | None = None,
    problem_description: str | None = None,
    problem_context: ProblemContext | None = None,
) -> tuple[EncodingType, str, "ProblemContext"]:
    """
    Recomenda encoding considerando dados e, opcionalmente, tarefa QML e algoritmo.
    Se nenhum contexto for dado, anexa um guia do que usar em cada tipo de problema.
    Retorna (encoding, texto_justificativa, contexto_inferido).
    """
    from llama_qiskit_agents.quantum.problem_context import (
        infer_problem_context,
        refine_recommendation,
    )

    ctx = problem_context or infer_problem_context(task, algorithm, problem_description)
    base_enc, base_reason = _recommend_encoding_from_data(profile)
    enc, reason = refine_recommendation(profile, base_enc, base_reason, ctx)
    return enc, reason, ctx


def get_encoding_tradeoffs() -> dict[EncodingType, str]:
    """Retorna texto resumindo trade-offs de cada encoding."""
    return {
        EncodingType.AMPLITUDE: (
            "Amplitude: poucos qubits (2^n amplitudes), mas circuito de preparação profundo e difícil em hardware real. "
            "Melhor para vetores únicos ou quando qubits são o gargalo."
        ),
        EncodingType.ANGLE: (
            "Angle: 1 qubit por feature, poucas portas, fácil de implementar. "
            "Limitado em expressibilidade; bom para começar e dados de baixa dimensão."
        ),
        EncodingType.DENSE_ANGLE: (
            "Dense angle: 2 features por qubit (Ry·Rz), profundidade 2. "
            "Melhor eficiência de qubits que angle sem o custo de profundidade do re-uploading. "
            "Ideal para dados contínuos com 5–12 features sem valores negativos."
        ),
        EncodingType.IQP: (
            "IQP: H + Rz(xᵢ²) diagonal + Rzz(xᵢ·xⱼ) entre pares. "
            "Separável em teoria de complexidade dos encodings de ângulo — "
            "base teórica forte para kernels quânticos em dimensão moderada. "
            "Mais profundo que dense_angle, mas mais expressivo sem entrelaçamento arbitrário."
        ),
        EncodingType.BASIS: (
            "Basis: 1 qubit por bit, ideal para dados binários/categóricos. "
            "Não aproveita superposição; eficiente para problemas de decisão."
        ),
        EncodingType.DATA_REUPLOADING: (
            "Data re-uploading: repete o dado em várias camadas, aumentando expressibilidade sem mais qubits. "
            "Circuito mais profundo; bom compromisso para QML."
        ),
        EncodingType.CUSTOM_FEATURE_MAP: (
            "Feature map customizado: rotações + entrelaçamento (ex.: ZZ). "
            "Controla expressibilidade e entrelaçamento; típico em quantum kernels."
        ),
    }
