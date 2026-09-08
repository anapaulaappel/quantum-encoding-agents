"""Análise de dados e recomendação de estratégia de embedding quântico."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.fractal import (
    FractalBudget,
    estimate_fractal_budget,
)


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
    features, _ = load_csv_from_string_with_labels(text)
    return features


def _row_all_numeric(cells: list[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        try:
            float(cell.strip())
        except (ValueError, AttributeError):
            return False
    return True


_LABEL_COLUMN_ALIASES = frozenset(
    {"label", "class", "target", "y", "classe", "rotulo", "rótulo", "category"}
)


def _parse_label_value(raw: str) -> int:
    s = raw.strip()
    try:
        return int(float(s))
    except ValueError:
        return hash(s) % 1000


def _detect_label_column_index(header: list[str], rows: list[list[str]]) -> int | None:
    for i, name in enumerate(header):
        if name.strip().lower() in _LABEL_COLUMN_ALIASES:
            return i
    if len(header) < 2 or len(rows) < 2:
        return None
    last = len(header) - 1
    values: list[int] = []
    for row in rows:
        if len(row) <= last:
            continue
        try:
            values.append(_parse_label_value(row[last]))
        except Exception:
            return None
    uniq = set(values)
    if 2 <= len(uniq) <= 20:
        return last
    return None


def load_csv_from_string_with_labels(
    text: str,
    label_column: str | None = None,
) -> tuple[np.ndarray, list[int] | None]:
    """
    Carrega CSV com features numéricas e coluna de rótulo opcional.
    Detecta header; label por nome (label/class/target/y) ou última coluna categórica.
    """
    rows_raw = [row for row in csv.reader(io.StringIO(text.strip())) if row and any(c.strip() for c in row)]
    if not rows_raw:
        raise ValueError("CSV vazio ou sem colunas numéricas: upload")

    has_header = not _row_all_numeric(rows_raw[0])
    header = [c.strip() for c in rows_raw[0]] if has_header else None
    data_rows = rows_raw[1:] if has_header else rows_raw
    if not data_rows:
        raise ValueError("CSV sem linhas de dados: upload")

    label_idx: int | None = None
    if header is not None:
        if label_column:
            key = label_column.strip().lower()
            for i, name in enumerate(header):
                if name.strip().lower() == key:
                    label_idx = i
                    break
            if label_idx is None:
                raise ValueError(f"Coluna de label '{label_column}' não encontrada no CSV.")
        else:
            label_idx = _detect_label_column_index(header, data_rows)
    else:
        last_vals: list[int] = []
        for row in data_rows:
            if not row:
                continue
            try:
                last_vals.append(_parse_label_value(row[-1]))
            except Exception:
                last_vals = []
                break
        if last_vals and 2 <= len(set(last_vals)) <= 20:
            label_idx = len(data_rows[0]) - 1

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    n_cols = max(len(r) for r in data_rows)

    for row in data_rows:
        cells = row + [""] * (n_cols - len(row))
        li = label_idx
        if li is not None and li >= len(cells):
            li = None
        if li is not None:
            labels.append(_parse_label_value(cells[li]))
        feats: list[float] = []
        for i, cell in enumerate(cells):
            if li is not None and i == li:
                continue
            try:
                feats.append(float(cell.strip()))
            except (ValueError, AttributeError):
                pass
        if feats:
            feature_rows.append(feats)

    arr = _array_from_numeric_rows(feature_rows, "upload")
    label_list: list[int] | None = None
    if labels and len(labels) == arr.shape[0]:
        label_list = labels
    return arr, label_list


def feature_names_from_csv_text(
    text: str,
    label_column: str | None = None,
) -> list[str] | None:
    """Nomes das colunas de feature (exclui label), se o CSV tiver header."""
    rows_raw = [row for row in csv.reader(io.StringIO(text.strip())) if row and any(c.strip() for c in row)]
    if not rows_raw:
        return None
    if _row_all_numeric(rows_raw[0]):
        return None
    header = [c.strip() for c in rows_raw[0]]

    label_idx: int | None = None
    if label_column:
        key = label_column.strip().lower()
        for i, name in enumerate(header):
            if name.strip().lower() == key:
                label_idx = i
                break
    else:
        label_idx = _detect_label_column_index(header, rows_raw[1:])

    names: list[str] = []
    for i, name in enumerate(header):
        if label_idx is not None and i == label_idx:
            continue
        names.append(name or f"col_{i}")
    return names if names else None


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
    intrinsic_dimension: float | None = None
    embedding_dimension: int | None = None
    qubit_budget: int | None = None
    selected_columns: list[int] | None = None
    selected_feature_names: list[str] | None = None
    fractal_selection_applied: bool = True


def effective_n_features(profile: DataProfile) -> int:
    """Largura efetiva após FD-ASE (senão n_features original)."""
    if profile.fractal_selection_applied and profile.selected_columns:
        return len(profile.selected_columns)
    return profile.n_features


def attach_fractal_budget(
    profile: DataProfile,
    budget: FractalBudget,
) -> DataProfile:
    profile.intrinsic_dimension = budget.intrinsic_dimension
    profile.embedding_dimension = budget.embedding_dimension
    profile.qubit_budget = budget.qubit_budget
    profile.selected_columns = budget.selected_columns
    profile.selected_feature_names = budget.selected_feature_names
    return profile


def infer_data_profile(
    data: np.ndarray | list[Any] | str | Path,
    *,
    feature_names: list[str] | None = None,
    apply_fractal_budget: bool = True,
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
            return _infer_profile_from_array(
                x,
                f"CSV: {p.name} ({x.shape[0]} linhas, {x.shape[1]} colunas)",
                feature_names=feature_names,
                apply_fractal_budget=apply_fractal_budget,
            )
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
    return _infer_profile_from_array(
        x,
        "Array de dados",
        feature_names=feature_names,
        apply_fractal_budget=apply_fractal_budget,
    )


def _infer_profile_from_array(
    x: np.ndarray,
    description: str,
    *,
    feature_names: list[str] | None = None,
    apply_fractal_budget: bool = True,
) -> DataProfile:
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
    profile = DataProfile(
        n_samples=int(n_samples),
        n_features=int(n_features),
        is_binary=bool(is_binary),
        is_categorical=bool(is_categorical),
        is_continuous=bool(is_continuous),
        has_negative=bool(has_negative),
        description=desc,
        embedding_dimension=int(n_features),
    )
    if not profile.is_binary:
        budget = estimate_fractal_budget(x, feature_names=feature_names)
        if budget.intrinsic_dimension is not None:
            attach_fractal_budget(profile, budget)
            profile.fractal_selection_applied = bool(apply_fractal_budget)
    return profile


def _recommend_encoding_from_data(profile: DataProfile) -> tuple[EncodingType, str]:
    """Recomendação baseada só no perfil dos dados (largura efetiva após FD-ASE)."""
    n_used = effective_n_features(profile)
    q_star = profile.qubit_budget if profile.fractal_selection_applied else None
    fractal_note = _fractal_reason_suffix(profile, lang="pt")

    if profile.is_binary and n_used <= 16:
        return (
            EncodingType.BASIS,
            "Dados binários/categóricos com poucos bits: basis encoding usa 1 qubit por bit e é natural para classificação."
            + fractal_note,
        )

    angle_qubits = n_used
    dense_qubits = max(1, int(np.ceil(n_used / 2)))
    angle_fits = q_star is None or angle_qubits <= q_star
    dense_fits = q_star is None or dense_qubits <= q_star

    if n_used <= 4 and profile.is_continuous and angle_fits:
        return (
            EncodingType.ANGLE,
            "Poucos features contínuos: angle encoding é simples, poucas portas e boa para protótipos."
            + fractal_note,
        )
    if 4 < n_used <= 12 and profile.is_continuous and not profile.has_negative and dense_fits:
        return (
            EncodingType.DENSE_ANGLE,
            f"Dense angle encoding empacota 2 features por qubit via Ry·Rz — "
            f"{n_used} features em ⌈{n_used}/2⌉={dense_qubits} qubits "
            f"com profundidade 2. Mais eficiente que angle sem o custo do re-uploading."
            + fractal_note,
        )
    if profile.n_samples <= 1 and n_used >= 4 and profile.is_continuous:
        return (
            EncodingType.AMPLITUDE,
            "Vetor único com muitos componentes: amplitude encoding compacta em log2(n) qubits (estado puro)."
            + fractal_note,
        )
    if profile.is_continuous and n_used <= 8:
        if not angle_fits and dense_fits:
            return (
                EncodingType.DENSE_ANGLE,
                f"Orçamento fractal q*={q_star}: angle/IQP precisariam de {n_used} qubits. "
                f"Dense-angle usa {dense_qubits} qubits para {n_used} features."
                + fractal_note,
            )
        return (
            EncodingType.DATA_REUPLOADING,
            "Dados contínuos com dimensão moderada: data re-uploading aumenta expressibilidade sem mais qubits."
            + fractal_note,
        )
    if profile.is_continuous and 8 < n_used <= 16 and angle_fits:
        return (
            EncodingType.IQP,
            f"IQP encoding com {n_used} features: H + Rz(x²) + Rzz(x·x') diagonais — "
            "base teórica forte para kernels quânticos em dimensão moderada (Havlíček et al., 2019)."
            + fractal_note,
        )
    if profile.is_continuous and not angle_fits and dense_fits:
        return (
            EncodingType.DENSE_ANGLE,
            f"E={profile.n_features} colunas, n_used={n_used} > q*={q_star}: "
            f"não encodar em angle/IQP nessa largura. Dense-angle usa {dense_qubits} qubits."
            + fractal_note,
        )
    if profile.is_continuous:
        return (
            EncodingType.CUSTOM_FEATURE_MAP,
            "Feature map customizado com rotações e entrelaçamento: flexível e bom para kernels quânticos."
            + fractal_note,
        )
    if not angle_fits:
        if dense_fits:
            return (
                EncodingType.DENSE_ANGLE,
                f"Orçamento fractal q*={q_star}: angle pediria {n_used} qubits. "
                f"Dense-angle usa {dense_qubits} qubits."
                + fractal_note,
            )
        return (
            EncodingType.CUSTOM_FEATURE_MAP,
            f"E={profile.n_features}, n_used={n_used} acima de q*={q_star}: "
            "não encodar em angle/IQP nessa largura."
            + fractal_note,
        )
    return (
        EncodingType.ANGLE,
        "Caso genérico: angle encoding é um bom padrão (simples e interpretável)."
        + fractal_note,
    )


def _fractal_reason_suffix(profile: DataProfile, *, lang: str) -> str:
    d2 = profile.intrinsic_dimension
    q_star = profile.qubit_budget
    if d2 is None or q_star is None:
        return ""
    cols = profile.selected_columns or []
    names = profile.selected_feature_names
    if names:
        col_txt = ", ".join(names)
    else:
        col_txt = ", ".join(str(c) for c in cols)
    n_sel = len(cols) if cols else 0
    dense_sel = max(1, int(np.ceil(max(n_sel, 1) / 2)))
    if not profile.fractal_selection_applied:
        if lang == "en":
            return (
                f" D2 ≈ {d2:.2f} → qubit budget q*={q_star}. "
                f"FD-ASE would keep [{col_txt}] ({n_sel} of {profile.n_features} columns). "
                f"Selection was not applied: recommendation uses all {profile.n_features} columns. "
                f"Angle-encoding the full width is likely to collapse; dense-angle on the subset would use {dense_sel} qubit(s)."
            )
        return (
            f" D2 ≈ {d2:.2f} → orçamento {q_star} qubits. "
            f"FD-ASE sugeriria [{col_txt}] ({n_sel} de {profile.n_features} colunas). "
            f"Seleção não aplicada: a recomendação usa as {profile.n_features} colunas. "
            f"Angle na largura cheia tende a colapsar; dense-angle no recorte usaria {dense_sel} qubit(s)."
        )
    if lang == "en":
        return (
            f" D2 ≈ {d2:.2f} → qubit budget q*={q_star}. "
            f"Do not angle-encode all {profile.n_features} columns. "
            f"FD-ASE kept [{col_txt}]. Dense-angle would use {dense_sel} qubit(s)."
        )
    return (
        f" D2 ≈ {d2:.2f} → orçamento {q_star} qubits. "
        f"Não encodar as {profile.n_features} colunas em angle. "
        f"FD-ASE manteve [{col_txt}]. Dense-angle usa {dense_sel} qubit(s)."
    )


def recommend_encoding(
    profile: DataProfile,
    *,
    task: str | None = None,
    algorithm: str | None = None,
    problem_description: str | None = None,
    problem_context: ProblemContext | None = None,
    hardware_profile: "HardwareProfile | None" = None,
) -> tuple[EncodingType, str, "ProblemContext"]:
    """
    Recomenda encoding considerando dados, tarefa QML, algoritmo e hardware alvo (opcional).
    Se nenhum contexto for dado, anexa um guia do que usar em cada tipo de problema.
    Retorna (encoding, texto_justificativa, contexto_inferido).
    """
    from llama_qiskit_agents.quantum.problem_context import (
        infer_problem_context,
        refine_recommendation,
    )

    ctx = problem_context or infer_problem_context(task, algorithm, problem_description)
    base_enc, base_reason = _recommend_encoding_from_data(profile)
    enc, reason = refine_recommendation(profile, base_enc, base_reason, ctx, hardware_profile=hardware_profile)
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
            "IQP: H + Rz(xᵢ²) diagonal + Rzz(xᵢ·xⱼ) em todos os pares (Hamiltoniano ZZ/diagonal, Havlíček). "
            "Separável em teoria de complexidade dos encodings de ângulo — "
            "base teórica forte para kernels quânticos em dimensão moderada. "
            "Use pairwise='adjacent' em topologias lineares/NISQ."
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
