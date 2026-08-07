"""
Perfil de hardware quântico para recomendação de encoding ciente de ruído.

O limiar crítico p* ≈ 10⁻³ (Sammartino, arXiv:2606.05387, 2026):
  - gate_error_rate >= p* → encodings profundos (amplitude, custom_feature_map, IQP)
    degradam mais rápido que o ganho expressivo justifica.
  - gate_error_rate < p* → todos os encodings são viáveis; critério principal é n_qubits.

Connectivity options e impacto:
  - "all-to-all": qualquer par de qubits pode aplicar gate de 2 qubits (simulador, IQP ideal).
  - "heavy-hex": topologia IBM Eagle/Heron — CNOT entre qubits não-adjacentes exige SWAPs.
  - "linear": cadeia 1D — IQP e custom_feature_map sofrem overhead de SWAP alto.
  - "grid": grade 2D — compromisso entre linear e all-to-all.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Limiar crítico de error rate de porta (Sammartino 2026, derivado formalmente)
NISQ_GATE_ERROR_THRESHOLD: float = 1e-3


@dataclass
class HardwareProfile:
    """
    Restrições de hardware alvo para ajuste da recomendação de encoding.
    Todos os campos são opcionais — qualquer subconjunto pode ser informado.
    """

    # Taxa de erro de porta de 2 qubits (ex: CNOT, CZ).
    # Valores típicos: IBM Eagle ~5e-3, IonQ ~3e-3, simulador ~0.
    # Limiar crítico p* ≈ 1e-3: acima disso encodings profundos são penalizados.
    gate_error_rate: float | None = None

    # Profundidade máxima de circuito tolerada após transpilação.
    # Encodings cujo depth simulado exceder este valor são rebaixados no ranking.
    max_depth_budget: int | None = None

    # Número máximo de qubits físicos disponíveis.
    # Encodings que requerem mais qubits são descartados.
    max_qubits: int | None = None

    # Topologia de conectividade do chip.
    # Impacta overhead de SWAP para gates de 2 qubits não-adjacentes.
    connectivity: str = "all-to-all"  # "all-to-all" | "heavy-hex" | "linear" | "grid"

    # Nome do backend (informativo — não altera lógica de recomendação diretamente).
    backend_name: str | None = None

    def is_nisq_constrained(self) -> bool:
        """True se gate_error_rate >= p* (Sammartino threshold)."""
        if self.gate_error_rate is None:
            return False
        return self.gate_error_rate >= NISQ_GATE_ERROR_THRESHOLD

    def exceeds_depth(self, depth: int) -> bool:
        """True se o circuito com 'depth' ultrapassa o budget de profundidade."""
        if self.max_depth_budget is None:
            return False
        return depth > self.max_depth_budget

    def exceeds_qubits(self, n_qubits: int) -> bool:
        """True se o circuito exige mais qubits que o hardware suporta."""
        if self.max_qubits is None:
            return False
        return n_qubits > self.max_qubits

    def has_swap_overhead(self) -> bool:
        """True se a conectividade implica overhead de SWAP para gates all-to-all."""
        return self.connectivity in ("heavy-hex", "linear", "grid")

    def describe(self, lang: str = "pt") -> str:
        """Resumo legível do perfil de hardware."""
        parts = []
        if self.backend_name:
            parts.append(self.backend_name)
        if self.gate_error_rate is not None:
            status = (
                ("acima" if lang == "pt" else "above") + " p*"
                if self.is_nisq_constrained()
                else ("abaixo" if lang == "pt" else "below") + " p*"
            )
            parts.append(f"gate_error={self.gate_error_rate:.1e} ({status})")
        if self.max_depth_budget is not None:
            lbl = "depth_max" if lang == "en" else "profundidade_max"
            parts.append(f"{lbl}={self.max_depth_budget}")
        if self.max_qubits is not None:
            parts.append(f"max_qubits={self.max_qubits}")
        if self.connectivity != "all-to-all":
            parts.append(f"topology={self.connectivity}")
        if not parts:
            return "simulador (sem restrições)" if lang == "pt" else "simulator (no constraints)"
        return ", ".join(parts)
