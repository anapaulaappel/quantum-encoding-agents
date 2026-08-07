"""Contexto de problema QML: tarefa, algoritmo e refinamento da recomendação de encoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llama_qiskit_agents.quantum.data_analysis import DataProfile
from llama_qiskit_agents.quantum.encodings import EncodingType


class MLTask(str, Enum):
    """Tipo de problema em quantum machine learning."""

    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    ENCODING_ONLY = "encoding_only"
    KERNEL_METHOD = "kernel_method"
    VARIATIONAL = "variational"
    UNKNOWN = "unknown"


@dataclass
class ProblemContext:
    """O que o usuário quer resolver (opcional)."""

    task: MLTask = MLTask.UNKNOWN
    algorithm: str | None = None
    raw_hints: str = ""
    inferred_note: str = ""

    def has_explicit_info(self) -> bool:
        return (
            self.task != MLTask.UNKNOWN
            or (self.algorithm is not None and self.algorithm.strip() != "")
            or (self.raw_hints.strip() != "")
        )


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def parse_task_label(task: str | None) -> MLTask | None:
    """Mapeia string explícita do usuário para MLTask."""
    t = _norm(task)
    if not t:
        return None
    mapping = {
        "classification": MLTask.CLASSIFICATION,
        "classificacao": MLTask.CLASSIFICATION,
        "classificação": MLTask.CLASSIFICATION,
        "clustering": MLTask.CLUSTERING,
        "clusterização": MLTask.CLUSTERING,
        "clusterizacao": MLTask.CLUSTERING,
        "encoding_only": MLTask.ENCODING_ONLY,
        "encoding": MLTask.ENCODING_ONLY,
        "state_preparation": MLTask.ENCODING_ONLY,
        "preparacao": MLTask.ENCODING_ONLY,
        "preparação": MLTask.ENCODING_ONLY,
        "kernel": MLTask.KERNEL_METHOD,
        "kernel_method": MLTask.KERNEL_METHOD,
        "variational": MLTask.VARIATIONAL,
        "vqc": MLTask.VARIATIONAL,
    }
    if t in mapping:
        return mapping[t]
    for key, val in mapping.items():
        if key in t:
            return val
    return None


# (substring no nome do algoritmo, task sugerida, encoding preferido ou None, nota curta)
_ALGORITHM_HINTS: list[tuple[str, MLTask, EncodingType | None, str]] = [
    ("qsvm", MLTask.KERNEL_METHOD, EncodingType.CUSTOM_FEATURE_MAP, "QSVM costuma usar feature map com entrelaçamento."),
    ("quantum svm", MLTask.KERNEL_METHOD, EncodingType.CUSTOM_FEATURE_MAP, "Quantum kernel SVM: feature map rico em expressibilidade."),
    ("quantum kernel", MLTask.KERNEL_METHOD, EncodingType.CUSTOM_FEATURE_MAP, "Métodos de kernel quântico combinam bem com feature maps (ZZ/Pauli-like)."),
    ("qkernel", MLTask.KERNEL_METHOD, EncodingType.CUSTOM_FEATURE_MAP, ""),
    ("vqc", MLTask.VARIATIONAL, EncodingType.DATA_REUPLOADING, "VQC: ângulos + re-upload ajudam a treinar fronteiras não lineares."),
    ("variational classifier", MLTask.VARIATIONAL, EncodingType.DATA_REUPLOADING, ""),
    ("qnn", MLTask.VARIATIONAL, EncodingType.DATA_REUPLOADING, "QNN / ansatz variacional: angle ou data re-uploading na camada de entrada."),
    ("vqe", MLTask.VARIATIONAL, None, "VQE: o encoding depende do Hamiltoniano; amplitude se o estado alvo for vetor denso."),
    ("qaoa", MLTask.VARIATIONAL, None, "QAOA: encoding do problema costuma ser específico do grafo (não só dos dados brutos)."),
    ("qgan", MLTask.VARIATIONAL, EncodingType.ANGLE, "QGAN: frequentemente angle encoding na entrada do gerador."),
    ("qpca", MLTask.CLUSTERING, EncodingType.AMPLITUDE, "QPCA-like: às vezes amplitude para carregar covariâncias em estado."),
]


def _infer_from_keywords(combined: str) -> tuple[MLTask, str]:
    """Inferência por palavras no texto livre."""
    if not combined:
        return MLTask.UNKNOWN, ""
    if any(
        w in combined
        for w in ("classific", "classification", "rotular", "label", "supervisionad")
    ):
        return MLTask.CLASSIFICATION, "Inferido: classificação (palavras-chave no texto)."
    if any(w in combined for w in ("cluster", "agrupar", "agrupamento", "não supervisionad", "nao supervisionad")):
        return MLTask.CLUSTERING, "Inferido: clusterização (palavras-chave no texto)."
    if any(
        w in combined
        for w in (
            "só encoding",
            "so encoding",
            "apenas encoding",
            "state prep",
            "preparação de estado",
            "preparacao de estado",
            "carregar o vetor",
        )
    ):
        return MLTask.ENCODING_ONLY, "Inferido: apenas preparação de estado / encoding."
    if any(
        w in combined
        for w in ("kernel", "qsvm", "quantum svm", "feature map", "hilbert", "overlap")
    ):
        return MLTask.KERNEL_METHOD, "Inferido: método de kernel / feature map (palavras-chave)."
    if any(
        w in combined
        for w in ("vqe", "qaoa", "variacional", "ansatz", "qnn", "vqc", "parametriz")
    ):
        return MLTask.VARIATIONAL, "Inferido: circuito variacional (palavras-chave)."
    return MLTask.UNKNOWN, ""


def infer_problem_context(
    task: str | None = None,
    algorithm: str | None = None,
    problem_description: str | None = None,
) -> ProblemContext:
    """
    Combina campos explícitos + texto livre + nome do algoritmo para montar o contexto.
    """
    parts = [_norm(task), _norm(algorithm), _norm(problem_description)]
    combined = " ".join(p for p in parts if p)

    ctx_task = MLTask.UNKNOWN
    inferred_note = ""

    parsed = parse_task_label(task)
    if parsed is not None:
        ctx_task = parsed

    algo = _norm(algorithm)
    if algo:
        for sub, tsk, _, _ in _ALGORITHM_HINTS:
            if sub in algo:
                if ctx_task == MLTask.UNKNOWN:
                    ctx_task = tsk
                break

    kw_task, kw_note = _infer_from_keywords(combined)
    if ctx_task == MLTask.UNKNOWN and kw_task != MLTask.UNKNOWN:
        ctx_task = kw_task
        inferred_note = kw_note

    return ProblemContext(
        task=ctx_task,
        algorithm=algorithm.strip() if algorithm else None,
        raw_hints=combined,
        inferred_note=inferred_note,
    )


def scenario_guide_when_unspecified() -> str:
    """Texto explicando qual encoding tende a ser melhor para cada tipo de problema."""
    return (
        "Guia rápido — qual encoding costuma servir para o quê:\n"
        "• Classificação binária / rótulos discretos com poucos bits: **basis** (direto) ou **angle** (features contínuas baixa dimensão).\n"
        "• Classificação variacional (VQC, QNN): **angle** ou **data_reuploading** na entrada + seu ansatz variacional.\n"
        "• QSVM / quantum kernel / similaridade em espaço quântico: **custom_feature_map** (rotações + entrelaçamento) ou **data_reuploading**.\n"
        "• Clusterização em feature space quântico: muitas vezes **custom_feature_map** ou **amplitude** se o protocolo carrega o vetor de features inteiro.\n"
        "• Só preparar estado a partir de um vetor normalizado: **amplitude** (poucos qubits, circuito mais profundo) ou **angle** (mais qubits, circuito raso).\n"
        "• Hardware raso / poucas portas: prefira **angle**; se qubits são o limite, considere **amplitude**.\n"
        "Quando você informar a tarefa (classificação, clusterização, só encoding, kernel, variacional) ou o nome do algoritmo, a recomendação acima é ajustada a esse contexto."
    )


def refine_recommendation(
    profile: DataProfile,
    data_encoding: EncodingType,
    data_reason: str,
    context: ProblemContext,
    hardware_profile: "HardwareProfile | None" = None,
) -> tuple[EncodingType, str]:
    """
    Ajusta encoding e justificativa com base na tarefa QML, algoritmo citado
    e restrições de hardware (opcional).

    Ordem de prioridade:
      1. Heurísticas por tarefa QML
      2. Override pelo nome do algoritmo
      3. Filtro de hardware (gate_error_rate, max_depth_budget, max_qubits)
      4. Guia genérico se nenhum contexto foi dado
    """
    from llama_qiskit_agents.quantum.hardware_profile import (
        HardwareProfile,
        NISQ_GATE_ERROR_THRESHOLD,
    )

    enc = data_encoding
    segments: list[str] = [data_reason]
    if context.inferred_note:
        segments.append(context.inferred_note)

    task = context.task
    if task == MLTask.KERNEL_METHOD and profile.is_continuous:
        # Encodings sem entrelaçamento são inadequados para kernels — elevar para o melhor disponível
        _NON_KERNEL_ENCODINGS = {EncodingType.BASIS, EncodingType.ANGLE, EncodingType.DENSE_ANGLE, EncodingType.AMPLITUDE}
        if enc in _NON_KERNEL_ENCODINGS:
            enc = EncodingType.CUSTOM_FEATURE_MAP
        segments.append(
            "Para kernel / QSVM em dados contínuos, feature maps com entrelaçamento (custom_feature_map) "
            "ampliam o espaço de features no espaço de Hilbert. IQP também é adequado para kernels "
            "com base teórica sólida (Havlíček et al. 2019)."
        )
    elif task == MLTask.VARIATIONAL and profile.is_continuous:
        if enc in (EncodingType.BASIS, EncodingType.AMPLITUDE) and profile.n_features <= 16:
            enc = EncodingType.DATA_REUPLOADING
        segments.append(
            "Em modelos variacionais (VQC/QNN), angle ou data re-uploading na entrada costuma acoplar melhor a um ansatz treinável."
        )
    elif task == MLTask.CLASSIFICATION:
        if profile.is_binary and profile.n_features <= 16:
            enc = EncodingType.BASIS
            segments.append("Classificação com dados binários/categóricos compactos: basis encoding alinha rótulos com estados da base.")
        elif profile.is_continuous:
            segments.append(
                "Classificação com features contínuas: angle (simples) ou data re-uploading / custom feature map (mais expressivo)."
            )
    elif task == MLTask.CLUSTERING:
        segments.append(
            "Clusterização: em QML costuma-se usar feature maps que aumentem separação em Hilbert; "
            "custom_feature_map ou amplitude (vetor inteiro) são candidatos fortes conforme o protocolo."
        )
        if profile.is_continuous and enc == EncodingType.BASIS:
            enc = EncodingType.CUSTOM_FEATURE_MAP
    elif task == MLTask.ENCODING_ONLY:
        if profile.n_samples <= 1 and profile.n_features >= 4 and profile.is_continuous:
            enc = EncodingType.AMPLITUDE
            segments.append("Apenas encoding de um vetor: amplitude minimiza qubits se o vetor já está normalizado.")
        else:
            segments.append("Apenas preparação de estado: angle é o mais simples; amplitude se precisar compactar muitas amplitudes em poucos qubits.")

    algo = _norm(context.algorithm)
    if algo:
        for sub, tsk, enc_pref, blurb in _ALGORITHM_HINTS:
            if sub in algo:
                if enc_pref is not None:
                    enc = enc_pref
                    segments.append(f"Ajuste pelo algoritmo ({sub}): preferir {enc_pref.value}. {blurb}".strip())
                elif blurb:
                    segments.append(f"Nota sobre {sub}: {blurb}")
                break

    # ── Filtro de hardware ────────────────────────────────────────────────
    if hardware_profile is not None:
        hw = hardware_profile

        # Encodings profundos penalizados acima do limiar p* de gate error
        _DEEP_ENCODINGS = {EncodingType.AMPLITUDE, EncodingType.CUSTOM_FEATURE_MAP, EncodingType.IQP}
        _SHALLOW_FALLBACK = EncodingType.DENSE_ANGLE if profile.n_features > 4 else EncodingType.ANGLE

        if hw.is_nisq_constrained() and enc in _DEEP_ENCODINGS:
            original = enc
            enc = _SHALLOW_FALLBACK
            segments.append(
                f"Ajuste de hardware: gate_error_rate={hw.gate_error_rate:.1e} ≥ p*={NISQ_GATE_ERROR_THRESHOLD:.0e} "
                f"(Sammartino 2026). {original.value} seria degradado por ruído acumulado — "
                f"substituído por {enc.value} (circuito mais raso, mais robusto em NISQ)."
            )

        # Overhead de SWAP em topologias restritas penaliza gates all-to-all
        if hw.has_swap_overhead() and enc == EncodingType.CUSTOM_FEATURE_MAP:
            segments.append(
                f"Atenção: custom_feature_map usa CZ full-pairwise — em topologia '{hw.connectivity}' "
                f"isso gera overhead de SWAP. Considere IQP (Rzz apenas entre pares adjacentes) "
                f"ou data_reuploading (CX em cadeia linear) para reduzir profundidade transpilada."
            )

        # Aviso de budget de profundidade (não muda o encoding — informa apenas)
        if hw.max_depth_budget is not None:
            _DEPTH_ESTIMATES = {
                EncodingType.ANGLE: 1,
                EncodingType.DENSE_ANGLE: 2,
                EncodingType.BASIS: 1,
                EncodingType.DATA_REUPLOADING: 4 * profile.n_features,
                EncodingType.IQP: 3 + 3 * max(0, profile.n_features - 1),
                EncodingType.CUSTOM_FEATURE_MAP: 3 * profile.n_features,
                EncodingType.AMPLITUDE: 4 * profile.n_features,
            }
            est = _DEPTH_ESTIMATES.get(enc, 0)
            if est > hw.max_depth_budget:
                segments.append(
                    f"Aviso de profundidade: {enc.value} estimado em ~{est} portas de profundidade — "
                    f"acima do budget de {hw.max_depth_budget}. O circuito real depende da transpilação "
                    f"e da topologia do hardware ({hw.connectivity})."
                )

        # Aviso de qubits insuficientes
        if hw.max_qubits is not None:
            import math
            _QUBIT_ESTIMATES = {
                EncodingType.AMPLITUDE: max(1, math.ceil(math.log2(max(1, profile.n_features)))),
                EncodingType.DENSE_ANGLE: max(1, math.ceil(profile.n_features / 2)),
            }
            est_qubits = _QUBIT_ESTIMATES.get(enc, profile.n_features)
            if est_qubits > hw.max_qubits:
                segments.append(
                    f"Aviso de qubits: {enc.value} requer ~{est_qubits} qubits para {profile.n_features} features — "
                    f"acima do limite de {hw.max_qubits} qubits do hardware informado."
                )

    if not context.has_explicit_info():
        segments.append("")
        segments.append(scenario_guide_when_unspecified())

    return enc, "\n".join(segments)
