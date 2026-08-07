"""Ranking de encodings com justificativa (dado + problema QML), separado do histograma de medições."""

from __future__ import annotations

from llama_qiskit_agents.quantum.data_analysis import DataProfile
from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.problem_context import MLTask, ProblemContext

# Frases curtas: quando considerar cada encoding como alternativa
_WHEN: dict[EncodingType, str] = {
    EncodingType.AMPLITUDE: (
        "Bom quando você precisa poucos qubits para muitas amplitudes; o preço é um circuito de preparação mais profundo."
    ),
    EncodingType.ANGLE: (
        "Bom ponto de partida: poucas portas e fácil de rodar em hardware; 1 qubit por feature contínua."
    ),
    EncodingType.DENSE_ANGLE: (
        "Metade dos qubits do angle encoding com profundidade 2 (Ry·Rz por qubit): "
        "melhor escolha quando features > qubits disponíveis e dados não têm valores negativos."
    ),
    EncodingType.BASIS: (
        "Ideal para dados já binários/categóricos compactos; pouco uso de superposição no encoding."
    ),
    EncodingType.DATA_REUPLOADING: (
        "Útil em QML variacional: reintroduz o dado em camadas e aumenta expressibilidade sem mais qubits."
    ),
    EncodingType.CUSTOM_FEATURE_MAP: (
        "Forte para quantum kernels / QSVM: entrelaçamento + rotações enriquecem o feature space em Hilbert."
    ),
}


def _first_substantive_line(text: str, max_len: int = 400) -> str:
    for ln in text.split("\n"):
        s = ln.strip()
        if s and not s.startswith("---"):
            return s[:max_len] + ("…" if len(s) > max_len else "")
    return (text.strip()[:max_len] + "…") if len(text.strip()) > max_len else text.strip()


def _alternativa_rationale(
    enc: EncodingType,
    recommended: EncodingType,
    profile: DataProfile,
    sim_enc,
    sim_rec,
    ctx: ProblemContext | None,
) -> str:
    parts: list[str] = []
    if sim_enc and sim_rec and enc != recommended:
        if sim_enc.depth < sim_rec.depth:
            parts.append(
                f"Circuito mais raso que a escolha principal (profundidade {sim_enc.depth} vs {sim_rec.depth}), "
                "favorecendo hardware NISQ com limite de profundidade."
            )
        elif sim_enc.depth > sim_rec.depth:
            parts.append(
                f"Mais profundo ({sim_enc.depth} vs {sim_rec.depth}); pode trazer mais expressibilidade se o hardware aguentar."
            )
    if enc != recommended:
        parts.append(_WHEN[enc])
    if ctx and ctx.task != MLTask.UNKNOWN:
        if ctx.task == MLTask.KERNEL_METHOD and enc == EncodingType.CUSTOM_FEATURE_MAP and recommended != enc:
            parts.append("Combina bem com tarefas de kernel mesmo não sendo a primeira opção heurística para o seu vetor.")
        elif ctx.task == MLTask.CLASSIFICATION and enc == EncodingType.BASIS and profile.is_binary:
            parts.append("Adequado a classificação com entradas binárias discretas.")
    return " ".join(parts) if parts else _WHEN[enc]


def format_encoding_ranking_section(
    profile: DataProfile,
    recommended: EncodingType,
    full_recommendation_reason: str,
    results: list,
    problem_context: ProblemContext | None = None,
) -> list[str]:
    """
    Blocos de texto para o relatório: ranking ordenado com 'por quê'.
    `results` é lista de SimulationResult (mesmo tipo que em simulate).
    """
    by_type = {r.encoding_type: r for r in results}
    if not by_type:
        return [
            "=== Ranking de encodings ===",
            "  (Nenhuma simulação concluída; não foi possível ordenar.)",
            "",
        ]

    header = [
        "=== Ranking de encodings (do mais adequado às alternativas) ===",
        "Critério: encaixe com o perfil do dado e com a tarefa/algoritmo QML que você informou — "
        "não pela frequência das medições no simulador.",
        "",
    ]

    if recommended not in by_type:
        ordered = sorted(
            [e for e in EncodingType if e in by_type],
            key=lambda e: (by_type[e].depth, by_type[e].num_qubits),
        )
        lines = header + [
            f"  Nota: a recomendação teórica é '{recommended.value}', mas a simulação desse encoding falhou. "
            "Abaixo, ordenação pragmática (profundidade → qubits) entre os circuitos que rodaram.",
            "",
        ]
        for i, enc in enumerate(ordered, start=1):
            r = by_type[enc]
            lines.append(f"  {i}. {enc.value}  |  qubits={r.num_qubits}, profundidade≈{r.depth}")
            lines.append(f"     Notas: {_WHEN[enc]}")
            lines.append("")
        return lines

    ordered: list[EncodingType] = [recommended]
    rest = [e for e in EncodingType if e in by_type and e != recommended]
    rest_sorted = sorted(rest, key=lambda e: (by_type[e].depth, by_type[e].num_qubits))
    ordered.extend(rest_sorted)

    summary = _first_substantive_line(full_recommendation_reason)
    lines = header.copy()

    sim_rec = by_type[recommended]
    for i, enc in enumerate(ordered, start=1):
        r = by_type[enc]
        if i == 1:
            lines.append(f"  {i}. {enc.value}  |  qubits={r.num_qubits}, profundidade≈{r.depth}")
            lines.append(f"     Por quê (escolha principal): {summary}")
        else:
            why = _alternativa_rationale(
                enc, recommended, profile, r, sim_rec, problem_context
            )
            lines.append(f"  {i}. {enc.value}  |  qubits={r.num_qubits}, profundidade≈{r.depth}")
            lines.append(f"     Por quê (alternativa): {why}")
        lines.append("")

    return lines


def format_measurements_note_section(results: list) -> list[str]:
    """Explica o histograma e mostra um resumo curto por encoding."""
    lines = [
        "=== O que são as medições (histograma) ===",
        "Ao medir, o simulador conta quantas vezes cada string de bits (ex.: '00', '11') apareceu. "
        "Isso descreve o estado preparado por aquele encoding, não diz qual encoding é 'melhor' para o seu problema de ML — "
        "para isso use o ranking acima.",
        "",
        "Resumo (até 3 outcomes mais frequentes por encoding):",
    ]
    for r in results:
        top = sorted(r.counts.items(), key=lambda x: -x[1])[:3]
        lines.append(f"  • {r.encoding_type.value}: {dict(top)}")
    lines.append("")
    return lines
