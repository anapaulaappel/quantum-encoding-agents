"""Modelos Pydantic para a API HTTP."""

from dataclasses import asdict

from pydantic import BaseModel, Field

from llama_qiskit_agents.quantum.data_analysis import DataProfile


class HardwareProfileInput(BaseModel):
    """
    Restrições de hardware quântico alvo — todos os campos são opcionais.

    Exemplos reais:
      IBM Eagle:      gate_error_rate=5e-3, connectivity="heavy-hex", max_qubits=127
      IonQ Aria:      gate_error_rate=3e-3, connectivity="all-to-all", max_qubits=25
      AerSimulator:   gate_error_rate=0,    connectivity="all-to-all"  (padrão)

    Limiar crítico p* ≈ 1e-3 (Sammartino, arXiv:2606.05387):
      - gate_error_rate >= 1e-3 → encodings profundos são penalizados na recomendação.
    """

    gate_error_rate: float | None = Field(
        default=None,
        description="Taxa de erro de porta de 2 qubits (ex: 5e-3 para IBM Eagle). Limiar crítico p*=1e-3.",
    )
    max_depth_budget: int | None = Field(
        default=None,
        description="Profundidade máxima de circuito tolerada. Encodings que a excedem recebem aviso.",
    )
    max_qubits: int | None = Field(
        default=None,
        description="Número máximo de qubits físicos disponíveis.",
    )
    connectivity: str = Field(
        default="all-to-all",
        description="Topologia do chip: 'all-to-all' | 'heavy-hex' | 'linear' | 'grid'.",
    )
    backend_name: str | None = Field(
        default=None,
        description="Nome do backend (informativo, ex: 'ibm_torino', 'ionq_aria').",
    )


class DataInput(BaseModel):
    """Entrada: descrição em texto ou lista numérica (uma amostra ou features)."""

    description: str | None = Field(default=None, description="Texto descrevendo o dataset")
    data: list[float] | None = Field(default=None, description="Lista de números (uma linha de features)")
    task: str | None = Field(
        default=None,
        description="Tarefa QML: classification | clustering | encoding | kernel | variational",
    )
    algorithm: str | None = Field(default=None, description="Nome do algoritmo (ex.: QSVM, VQC, QAOA)")
    problem_description: str | None = Field(
        default=None,
        description="Texto livre sobre o problema (classificação, kernel, etc.)",
    )
    hardware_profile: HardwareProfileInput | None = Field(
        default=None,
        description="Restrições de hardware alvo (gate_error_rate, max_depth_budget, max_qubits, connectivity).",
    )


class CompareRequest(DataInput):
    n_qubits: int | None = None
    shots: int = 1024


class CircuitRequest(BaseModel):
    encoding_name: str = Field(
        ...,
        description="amplitude | angle | dense_angle | iqp | basis | data_reuploading | custom_feature_map",
    )
    data: list[float] = Field(default_factory=list)
    n_qubits: int | None = None


class SimulateRequest(CircuitRequest):
    shots: int = 1024


class ProfileResponse(BaseModel):
    n_samples: int
    n_features: int
    is_binary: bool
    is_categorical: bool
    is_continuous: bool
    has_negative: bool
    description: str


class RecommendResponse(BaseModel):
    profile: ProfileResponse
    recommended_encoding: str
    reason: str
    task_interpreted: str = Field(description="Tarefa QML inferida ou informada")
    algorithm: str | None = Field(default=None, description="Algoritmo informado (se houver)")
    scenario_guide_included: bool = Field(
        default=False,
        description="True se nenhum contexto de problema foi dado e o guia geral foi anexado",
    )


class ExplainRequest(DataInput):
    """Entrada para /v1/recommend/explain — tudo em um campo só para facilitar o chat."""

    n_qubits: int | None = Field(default=None, description="Número de qubits (opcional)")
    shots: int = Field(default=512, description="Shots para simulação do circuito recomendado")
    lang: str | None = Field(
        default=None,
        description="Idioma da resposta: 'pt' ou 'en'. Se omitido, detectado automaticamente.",
    )


class ExplainResponse(BaseModel):
    """Resposta rica: recomendação + justificativa narrativa + código Qiskit."""

    lang: str = Field(description="Idioma detectado ou informado: 'pt' ou 'en'")
    recommended_encoding: str
    explanation: str = Field(description="Justificativa em linguagem natural, no idioma do input")
    qiskit_code: str = Field(description="Código Python/Qiskit completo e copiável")
    profile: ProfileResponse
    task_interpreted: str
    circuit_depth: int | None = Field(default=None, description="Profundidade do circuito simulado")
    circuit_qubits: int | None = Field(default=None, description="Número de qubits do circuito simulado")
    hardware_constraints_applied: bool = Field(
        default=False,
        description="True se hardware_profile foi informado e influenciou a recomendação.",
    )


class HealthResponse(BaseModel):
    status: str
    service: str


class ErrorResponse(BaseModel):
    detail: str


def profile_to_response(p: DataProfile) -> ProfileResponse:
    d = asdict(p)
    return ProfileResponse(**{k: d[k] for k in ProfileResponse.model_fields})
