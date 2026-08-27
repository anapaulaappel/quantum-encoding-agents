"""
Heurística de simulabilidade clássica de feature maps quânticos.

Motivação (comentário C6 / José Victor): kernels quânticos só trazem vantagem
potencial quando o feature map não é trivialmente simulável em hardware clássico.
Esta avaliação é conservadora e orientada a relatório — não substitui análise formal.
"""

from __future__ import annotations

from enum import Enum

from llama_qiskit_agents.quantum.encodings import EncodingType


class ClassicalSimulabilityHint(str, Enum):
    """Classificação qualitativa para o relatório de comparação."""

    LIKELY_EASY = "likely_classically_easy"
    MODERATE = "moderate"
    LIKELY_HARD = "likely_classically_hard"


def assess_encoding_simulability(
    encoding_type: EncodingType,
    n_qubits: int,
    *,
    iqp_pairwise: str = "all",
) -> tuple[ClassicalSimulabilityHint, str]:
    """
    Retorna (hint, explicação curta em português) para incluir no relatório.
    """
    n = max(1, n_qubits)

    if encoding_type in (EncodingType.ANGLE, EncodingType.BASIS):
        return (
            ClassicalSimulabilityHint.LIKELY_EASY,
            "Estados produto (sem entrelaçamento estrutural) — kernel implícito "
            "costuma ser reproduzível clássicamente com baixo custo.",
        )

    if encoding_type == EncodingType.DENSE_ANGLE:
        return (
            ClassicalSimulabilityHint.LIKELY_EASY,
            "Rotações locais Ry·Rz por qubit, sem gates de 2 qubits — "
            "feature map eficientemente simulável clássicamente.",
        )

    if encoding_type == EncodingType.AMPLITUDE:
        return (
            ClassicalSimulabilityHint.MODERATE,
            "Preparação de estado profunda, mas ainda simulável exatamente via "
            "statevector para n modesto; vantagem quântica depende do pós-processamento.",
        )

    if encoding_type == EncodingType.DATA_REUPLOADING:
        return (
            ClassicalSimulabilityHint.MODERATE,
            "Entrelaçamento leve (CX em cadeia) + re-uploading — mais expressivo que angle, "
            "mas ainda simulável clássicamente em escalas pequenas.",
        )

    if encoding_type == EncodingType.IQP:
        if iqp_pairwise == "adjacent" and n <= 6:
            return (
                ClassicalSimulabilityHint.MODERATE,
                "Hamiltoniano ZZ restrito (pares adjacentes) + rotações não-Clifford — "
                "candidato intermediário; simulável em n pequeno, menos trivial que angle.",
            )
        return (
            ClassicalSimulabilityHint.LIKELY_HARD,
            "Hamiltoniano ZZ/diagonal estilo Havlíček com termos xᵢ·xⱼ em múltiplos pares + H — "
            "feature map não trivialmente simulável; alinhado à motivação de kernels quânticos.",
        )

    if encoding_type == EncodingType.CUSTOM_FEATURE_MAP:
        return (
            ClassicalSimulabilityHint.LIKELY_HARD,
            "Entrelaçamento CZ/ZZ em muitos pares — estrutura rica em Hilbert, "
            "tipicamente difícil de simular clássicamente em escalas úteis.",
        )

    return (
        ClassicalSimulabilityHint.MODERATE,
        "Simulabilidade clássica não classificada — tratar como caso intermediário.",
    )


def format_simulability_section(
    results: list,
    *,
    iqp_pairwise: str = "all",
) -> list[str]:
    """Bloco de texto para o relatório de comparação."""
    lines = [
        "=== Simulabilidade clássica (heurística) ===",
        "Indica se o feature map tende a ser trivialmente simulável em hardware clássico "
        "(comentário C6 / motivação QML kernel). Não é prova formal de quantum advantage.",
        "",
    ]
    for r in results:
        hint, explanation = assess_encoding_simulability(
            r.encoding_type,
            r.num_qubits,
            iqp_pairwise=iqp_pairwise if r.encoding_type == EncodingType.IQP else "all",
        )
        label = {
            ClassicalSimulabilityHint.LIKELY_EASY: "provável simulação clássica fácil",
            ClassicalSimulabilityHint.MODERATE: "intermediário",
            ClassicalSimulabilityHint.LIKELY_HARD: "provável simulação clássica difícil",
        }[hint]
        lines.append(f"  • {r.encoding_type.value}: {label}")
        lines.append(f"    {explanation}")
    lines.append("")
    return lines
