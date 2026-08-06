"""
Geração de explicações em linguagem natural e código Qiskit para o encoding recomendado.

Responsabilidades:
  - Detectar idioma do input (PT-BR vs EN)
  - Gerar justificativa narrativa que cita DataProfile + SimulationResult explicitamente
  - Gerar código Python/Qiskit completo e copiável para o encoding escolhido
"""

from __future__ import annotations

from llama_qiskit_agents.quantum.data_analysis import DataProfile
from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.problem_context import MLTask, ProblemContext
from llama_qiskit_agents.quantum.simulate import SimulationResult


# ---------------------------------------------------------------------------
# Detecção de idioma
# ---------------------------------------------------------------------------

_PT_MARKERS = {
    "com", "para", "dados", "que", "uma", "quero", "usar", "meu", "minha",
    "tenho", "features", "amostras", "classificação", "classificacao",
    "contínuo", "continuo", "binário", "binario", "categórico", "categorico",
    "algoritmo", "modelo", "problema", "quântico", "quantico", "circuito",
    "treinamento", "predição", "predicao", "resultado", "entrada", "saída",
    "rede", "neural", "kernel", "vetor", "matriz",
}

_EN_MARKERS = {
    "with", "for", "data", "that", "want", "use", "my", "have",
    "features", "samples", "classification", "continuous", "binary",
    "categorical", "algorithm", "model", "problem", "quantum", "circuit",
    "training", "prediction", "result", "input", "output", "network",
    "neural", "kernel", "vector", "matrix", "encoding", "dataset",
}


def detect_language(text: str) -> str:
    """
    Detecta se o texto é PT-BR ou EN.
    Retorna 'pt' ou 'en'. Default: 'pt' em caso de empate.
    """
    if not text or not text.strip():
        return "pt"
    words = set(text.lower().split())
    pt_score = len(words & _PT_MARKERS)
    en_score = len(words & _EN_MARKERS)
    return "en" if en_score > pt_score else "pt"


# ---------------------------------------------------------------------------
# Frases narrativas dinâmicas — citam números concretos do perfil e simulação
# ---------------------------------------------------------------------------

def _fmt_features(n: int, lang: str) -> str:
    if lang == "en":
        return f"{n} feature{'s' if n != 1 else ''}"
    return f"{n} feature{'s' if n != 1 else ''}"


def build_natural_explanation(
    profile: DataProfile,
    recommended: EncodingType,
    reason: str,
    sim_result: SimulationResult | None,
    context: ProblemContext,
    lang: str = "pt",
) -> str:
    """
    Gera uma explicação narrativa completa em PT-BR ou EN citando explicitamente:
      - Características estruturais do dado (DataProfile)
      - Métricas de hardware do circuito (SimulationResult: depth, num_qubits)
      - Por que essas características favorecem o encoding escolhido
    """
    parts: list[str] = []

    # --- Parágrafo 1: perfil dos dados ---
    if lang == "en":
        data_desc = _describe_data_en(profile)
        parts.append(data_desc)
    else:
        data_desc = _describe_data_pt(profile)
        parts.append(data_desc)

    # --- Parágrafo 2: recomendação com justificativa estrutural ---
    if lang == "en":
        parts.append(_explain_recommendation_en(profile, recommended, context))
    else:
        parts.append(_explain_recommendation_pt(profile, recommended, context))

    # --- Parágrafo 3: métricas do circuito simulado ---
    if sim_result is not None:
        if lang == "en":
            parts.append(_explain_circuit_en(sim_result, recommended, profile))
        else:
            parts.append(_explain_circuit_pt(sim_result, recommended, profile))

    # --- Parágrafo 4: por que é melhor que as alternativas ---
    if lang == "en":
        parts.append(_explain_vs_alternatives_en(recommended, profile, context))
    else:
        parts.append(_explain_vs_alternatives_pt(recommended, profile, context))

    return "\n\n".join(p for p in parts if p.strip())


# --- Descrição dos dados ---

def _describe_data_pt(p: DataProfile) -> str:
    tipo = (
        "binários (apenas 0 e 1)" if p.is_binary
        else "categóricos (valores discretos)" if p.is_categorical
        else "contínuos (valores reais)" if p.is_continuous
        else "de tipo misto"
    )
    neg = " com valores negativos presentes" if p.has_negative else ""
    if p.n_samples > 0:
        return (
            f"Seu dataset tem {p.n_samples} amostra{'s' if p.n_samples != 1 else ''} "
            f"e {p.n_features} feature{'s' if p.n_features != 1 else ''}, "
            f"com valores {tipo}{neg}."
        )
    return (
        f"A descrição indica dados {tipo}{neg}."
    )


def _describe_data_en(p: DataProfile) -> str:
    tipo = (
        "binary (only 0s and 1s)" if p.is_binary
        else "categorical (discrete values)" if p.is_categorical
        else "continuous (real-valued)" if p.is_continuous
        else "mixed-type"
    )
    neg = " with negative values present" if p.has_negative else ""
    if p.n_samples > 0:
        return (
            f"Your dataset has {p.n_samples} sample{'s' if p.n_samples != 1 else ''} "
            f"and {p.n_features} feature{'s' if p.n_features != 1 else ''}, "
            f"with {tipo} values{neg}."
        )
    return f"The description indicates {tipo} data{neg}."


# --- Justificativa da recomendação ---

_RECOMMENDATION_REASONS_PT: dict[EncodingType, str] = {
    EncodingType.AMPLITUDE: (
        "**Amplitude encoding** é o mais indicado aqui porque seu vetor de dados pode ser "
        "codificado como amplitudes de um estado quântico, reduzindo drasticamente o número "
        "de qubits necessários: em vez de 1 qubit por feature, usa-se apenas ⌈log₂(n)⌉ qubits. "
        "A desvantagem é um circuito de preparação de estado mais profundo — compensado quando "
        "o número de qubits disponíveis é limitado."
    ),
    EncodingType.ANGLE: (
        "**Angle encoding** é o mais indicado aqui: cada feature vira o ângulo de uma rotação Ry "
        "em um qubit dedicado. Com poucos features e valores contínuos, o circuito fica muito raso "
        "(profundidade ≈ 1), fácil de rodar até em hardware NISQ real. É o melhor ponto de partida "
        "para prototipagem e dados de baixa dimensão."
    ),
    EncodingType.BASIS: (
        "**Basis encoding** é o mais indicado aqui porque seus dados já são binários ou categóricos: "
        "cada valor 0/1 mapeia diretamente para um estado da base computacional |0⟩/|1⟩. "
        "O circuito usa apenas portas X (bit-flip), tornando-o extremamente raso e eficiente — "
        "ideal para problemas de decisão e classificação com entradas discretas."
    ),
    EncodingType.DATA_REUPLOADING: (
        "**Data re-uploading** é o mais indicado aqui: o dado é inserido no circuito em múltiplas "
        "camadas (re-upload), o que aumenta a expressibilidade do modelo sem precisar de mais qubits. "
        "Cada camada aplica rotações Ry seguidas de CNOTs para entrelaçamento, criando um espaço de "
        "features implicitamente não linear — fundamental para redes neurais quânticas (QNN) e VQC."
    ),
    EncodingType.CUSTOM_FEATURE_MAP: (
        "**Custom feature map** é o mais indicado aqui: combina rotações (H, Rz, Ry) com "
        "entrelaçamento (CZ) para mapear os dados a um espaço de Hilbert de alta dimensão. "
        "Essa riqueza estrutural é essencial para métodos de kernel quântico (QSVM) onde "
        "a similaridade entre amostras é calculada nesse espaço — quanto mais expressivo o "
        "feature map, maior o potencial de separabilidade quântica."
    ),
}

_RECOMMENDATION_REASONS_EN: dict[EncodingType, str] = {
    EncodingType.AMPLITUDE: (
        "**Amplitude encoding** is the best fit here because your data vector can be encoded "
        "as quantum state amplitudes, dramatically reducing the qubit count: instead of 1 qubit "
        "per feature, only ⌈log₂(n)⌉ qubits are needed. The trade-off is a deeper state "
        "preparation circuit — worth it when qubit count is the bottleneck."
    ),
    EncodingType.ANGLE: (
        "**Angle encoding** is the best fit here: each feature becomes the angle of an Ry "
        "rotation on a dedicated qubit. With few continuous features, the circuit stays very "
        "shallow (depth ≈ 1), making it easy to run even on real NISQ hardware. It's the best "
        "starting point for prototyping and low-dimensional data."
    ),
    EncodingType.BASIS: (
        "**Basis encoding** is the best fit here because your data is already binary or "
        "categorical: each 0/1 value maps directly to a computational basis state |0⟩/|1⟩. "
        "The circuit uses only X gates (bit-flips), making it extremely shallow and efficient — "
        "ideal for decision problems and classification with discrete inputs."
    ),
    EncodingType.DATA_REUPLOADING: (
        "**Data re-uploading** is the best fit here: the data is inserted into the circuit in "
        "multiple layers, increasing model expressibility without requiring more qubits. "
        "Each layer applies Ry rotations followed by CNOT chains for entanglement, creating an "
        "implicitly nonlinear feature space — essential for quantum neural networks (QNN) and VQC."
    ),
    EncodingType.CUSTOM_FEATURE_MAP: (
        "**Custom feature map** is the best fit here: it combines rotations (H, Rz, Ry) with "
        "entanglement (CZ) to map data into a high-dimensional Hilbert space. This structural "
        "richness is essential for quantum kernel methods (QSVM) where sample similarity is "
        "computed in that space — the more expressive the feature map, the greater the potential "
        "quantum separability."
    ),
}


def _explain_recommendation_pt(
    profile: DataProfile,
    recommended: EncodingType,
    context: ProblemContext,
) -> str:
    base = _RECOMMENDATION_REASONS_PT[recommended]
    extra = ""
    if context.task != MLTask.UNKNOWN:
        task_notes = {
            MLTask.CLASSIFICATION: (
                f" Para classificação com {_fmt_features(profile.n_features, 'pt')}, "
                "o encoding precisa ser expressivo o suficiente para separar as classes no espaço quântico."
            ),
            MLTask.KERNEL_METHOD: (
                " Em métodos de kernel quântico, o encoding define o kernel implicitamente — "
                "a escolha do feature map impacta diretamente a capacidade de separação do modelo."
            ),
            MLTask.VARIATIONAL: (
                " Em circuitos variacionais, o encoding de entrada é treinado junto com o ansatz; "
                "re-uploading e angle encoding acoplam bem com camadas parametrizadas."
            ),
            MLTask.CLUSTERING: (
                " Para clusterização quântica, o encoding determina como os dados se distribuem "
                "no espaço de Hilbert — a separação entre clusters depende diretamente dessa escolha."
            ),
            MLTask.ENCODING_ONLY: (
                " Como o objetivo é apenas preparar o estado quântico (sem treinamento posterior), "
                "a eficiência do circuito em qubits e profundidade é o critério principal."
            ),
        }
        extra = task_notes.get(context.task, "")
    if context.algorithm:
        extra += f" (algoritmo alvo: {context.algorithm})"
    return base + extra


def _explain_recommendation_en(
    profile: DataProfile,
    recommended: EncodingType,
    context: ProblemContext,
) -> str:
    base = _RECOMMENDATION_REASONS_EN[recommended]
    extra = ""
    if context.task != MLTask.UNKNOWN:
        task_notes = {
            MLTask.CLASSIFICATION: (
                f" For classification with {_fmt_features(profile.n_features, 'en')}, "
                "the encoding needs to be expressive enough to separate classes in quantum space."
            ),
            MLTask.KERNEL_METHOD: (
                " In quantum kernel methods, the encoding implicitly defines the kernel — "
                "the feature map choice directly impacts the model's separation capacity."
            ),
            MLTask.VARIATIONAL: (
                " In variational circuits, the input encoding is trained alongside the ansatz; "
                "re-uploading and angle encoding couple well with parameterized layers."
            ),
            MLTask.CLUSTERING: (
                " For quantum clustering, the encoding determines how data distributes in "
                "Hilbert space — cluster separation depends directly on this choice."
            ),
            MLTask.ENCODING_ONLY: (
                " Since the goal is state preparation only (no downstream training), "
                "circuit efficiency in qubits and depth is the primary criterion."
            ),
        }
        extra = task_notes.get(context.task, "")
    if context.algorithm:
        extra += f" (target algorithm: {context.algorithm})"
    return base + extra


# --- Métricas do circuito simulado ---

def _explain_circuit_pt(
    sim: SimulationResult,
    recommended: EncodingType,
    profile: DataProfile,
) -> str:
    depth_comment = _depth_comment_pt(sim.depth)
    qubit_comment = _qubit_comment_pt(sim.num_qubits, profile)
    top_outcomes = sorted(sim.counts.items(), key=lambda x: -x[1])[:3]
    top_str = ", ".join(f"|{s}⟩ ({c}×)" for s, c in top_outcomes)
    return (
        f"O circuito {recommended.value} simulado usa **{sim.num_qubits} qubit{'s' if sim.num_qubits != 1 else ''}** "
        f"e tem **profundidade {sim.depth}**. {depth_comment} {qubit_comment} "
        f"Nas {sim.shots} medições simuladas, os estados mais frequentes foram: {top_str}."
    )


def _explain_circuit_en(
    sim: SimulationResult,
    recommended: EncodingType,
    profile: DataProfile,
) -> str:
    depth_comment = _depth_comment_en(sim.depth)
    qubit_comment = _qubit_comment_en(sim.num_qubits, profile)
    top_outcomes = sorted(sim.counts.items(), key=lambda x: -x[1])[:3]
    top_str = ", ".join(f"|{s}⟩ ({c}×)" for s, c in top_outcomes)
    return (
        f"The simulated {recommended.value} circuit uses **{sim.num_qubits} qubit{'s' if sim.num_qubits != 1 else ''}** "
        f"and has **depth {sim.depth}**. {depth_comment} {qubit_comment} "
        f"Over {sim.shots} simulated shots, the most frequent states were: {top_str}."
    )


def _depth_comment_pt(depth: int) -> str:
    if depth <= 3:
        return "Profundidade muito baixa: compatível com hardware NISQ atual sem correção de erros."
    if depth <= 10:
        return "Profundidade moderada: viável em dispositivos NISQ com boa fidelidade de portas."
    if depth <= 30:
        return "Profundidade média: requer hardware com baixa taxa de erro ou simulação clássica."
    return "Profundidade alta: ideal apenas para simulação clássica ou hardware tolerante a falhas."


def _depth_comment_en(depth: int) -> str:
    if depth <= 3:
        return "Very low depth: compatible with current NISQ hardware without error correction."
    if depth <= 10:
        return "Moderate depth: feasible on NISQ devices with good gate fidelity."
    if depth <= 30:
        return "Medium depth: requires low-error hardware or classical simulation."
    return "High depth: suitable only for classical simulation or fault-tolerant hardware."


def _qubit_comment_pt(n_qubits: int, profile: DataProfile) -> str:
    if profile.n_features > 0 and n_qubits < profile.n_features:
        savings = profile.n_features - n_qubits
        return (
            f"Usa {savings} qubit{'s' if savings != 1 else ''} a menos que o número de features "
            f"({profile.n_features}), graças à compressão por amplitudes."
        )
    if profile.n_features > 0 and n_qubits == profile.n_features:
        return f"Usa exatamente 1 qubit por feature ({profile.n_features} features → {n_qubits} qubits)."
    return ""


def _qubit_comment_en(n_qubits: int, profile: DataProfile) -> str:
    if profile.n_features > 0 and n_qubits < profile.n_features:
        savings = profile.n_features - n_qubits
        return (
            f"Uses {savings} fewer qubit{'s' if savings != 1 else ''} than the feature count "
            f"({profile.n_features}), thanks to amplitude compression."
        )
    if profile.n_features > 0 and n_qubits == profile.n_features:
        return f"Uses exactly 1 qubit per feature ({profile.n_features} features → {n_qubits} qubits)."
    return ""


# --- Comparação com alternativas ---

_ALTERNATIVES_PT: dict[EncodingType, dict[EncodingType, str]] = {
    EncodingType.ANGLE: {
        EncodingType.AMPLITUDE: "Amplitude encoding usaria menos qubits (⌈log₂(n)⌉) mas com circuito muito mais profundo — desvantajoso para poucos features.",
        EncodingType.BASIS: "Basis encoding só faz sentido para dados binários; com valores contínuos, perderia toda a informação de magnitude.",
        EncodingType.DATA_REUPLOADING: "Data re-uploading seria mais expressivo, mas com profundidade maior e mais portas — desnecessário para dados simples.",
        EncodingType.CUSTOM_FEATURE_MAP: "Custom feature map traz entrelaçamento sofisticado, mas é overkill para dados de baixa dimensão sem um kernel definido.",
    },
    EncodingType.AMPLITUDE: {
        EncodingType.ANGLE: "Angle encoding usaria 1 qubit por feature — com muitas features, o custo em qubits seria impraticável.",
        EncodingType.BASIS: "Basis encoding é só para dados binários; amplitude encoding preserva toda a informação contínua do vetor.",
        EncodingType.DATA_REUPLOADING: "Data re-uploading usa tantos qubits quanto features — amplitude encoding compacta tudo em log₂(n) qubits.",
        EncodingType.CUSTOM_FEATURE_MAP: "Custom feature map também usa 1 qubit por feature e adiciona profundidade com entrelaçamento — mais custoso sem ganho claro aqui.",
    },
    EncodingType.BASIS: {
        EncodingType.ANGLE: "Angle encoding aplicaria rotações a valores binários — funcionaria, mas desperdiçaria a natureza discreta dos dados.",
        EncodingType.AMPLITUDE: "Amplitude encoding normalizaria os bits como amplitudes — perde o alinhamento natural entre bits e estados da base.",
        EncodingType.DATA_REUPLOADING: "Data re-uploading adicionaria múltiplas camadas desnecessárias para dados já binários.",
        EncodingType.CUSTOM_FEATURE_MAP: "Custom feature map adicionaria rotações e entrelaçamento sem benefício para dados discretos binários.",
    },
    EncodingType.DATA_REUPLOADING: {
        EncodingType.ANGLE: "Angle encoding é mais raso, mas menos expressivo — com uma única camada, limita a capacidade do modelo variacional.",
        EncodingType.AMPLITUDE: "Amplitude encoding compacta qubits mas não se integra naturalmente com ansatze variacionais camada a camada.",
        EncodingType.BASIS: "Basis encoding é inadequado para features contínuas em modelos variacionais.",
        EncodingType.CUSTOM_FEATURE_MAP: "Custom feature map é ótimo para kernels, mas não é o padrão para camadas variacionais treináveis.",
    },
    EncodingType.CUSTOM_FEATURE_MAP: {
        EncodingType.ANGLE: "Angle encoding é muito simples para definir um kernel implícito: sem entrelaçamento, o espaço de features é pouco expressivo.",
        EncodingType.AMPLITUDE: "Amplitude encoding compacta o dado mas não cria o entrelaçamento necessário para um kernel quântico rico.",
        EncodingType.BASIS: "Basis encoding é para dados binários — kernels quânticos trabalham com dados contínuos em espaço de Hilbert.",
        EncodingType.DATA_REUPLOADING: "Data re-uploading é mais orientado a QNN/VQC; feature maps com CZ/ZZ têm estrutura mais adequada para QSVM.",
    },
}

_ALTERNATIVES_EN: dict[EncodingType, dict[EncodingType, str]] = {
    EncodingType.ANGLE: {
        EncodingType.AMPLITUDE: "Amplitude encoding would use fewer qubits (⌈log₂(n)⌉) but with a much deeper circuit — disadvantageous for few features.",
        EncodingType.BASIS: "Basis encoding only makes sense for binary data; with continuous values, it would lose all magnitude information.",
        EncodingType.DATA_REUPLOADING: "Data re-uploading would be more expressive but with greater depth and more gates — unnecessary for simple data.",
        EncodingType.CUSTOM_FEATURE_MAP: "Custom feature map brings sophisticated entanglement, but is overkill for low-dimensional data without a defined kernel.",
    },
    EncodingType.AMPLITUDE: {
        EncodingType.ANGLE: "Angle encoding would use 1 qubit per feature — with many features, the qubit cost would be impractical.",
        EncodingType.BASIS: "Basis encoding is for binary data only; amplitude encoding preserves all continuous vector information.",
        EncodingType.DATA_REUPLOADING: "Data re-uploading uses as many qubits as features — amplitude encoding compresses everything into log₂(n) qubits.",
        EncodingType.CUSTOM_FEATURE_MAP: "Custom feature map also uses 1 qubit per feature and adds entanglement depth — more costly with no clear advantage here.",
    },
    EncodingType.BASIS: {
        EncodingType.ANGLE: "Angle encoding would apply rotations to binary values — it would work, but wastes the discrete nature of the data.",
        EncodingType.AMPLITUDE: "Amplitude encoding would normalize the bits as amplitudes — loses the natural alignment between bits and basis states.",
        EncodingType.DATA_REUPLOADING: "Data re-uploading would add unnecessary multiple layers for already-binary data.",
        EncodingType.CUSTOM_FEATURE_MAP: "Custom feature map would add rotations and entanglement with no benefit for discrete binary data.",
    },
    EncodingType.DATA_REUPLOADING: {
        EncodingType.ANGLE: "Angle encoding is shallower but less expressive — with a single layer, it limits variational model capacity.",
        EncodingType.AMPLITUDE: "Amplitude encoding compresses qubits but doesn't naturally integrate with layer-by-layer variational ansatze.",
        EncodingType.BASIS: "Basis encoding is inadequate for continuous features in variational models.",
        EncodingType.CUSTOM_FEATURE_MAP: "Custom feature map is great for kernels but not the standard for trainable variational layers.",
    },
    EncodingType.CUSTOM_FEATURE_MAP: {
        EncodingType.ANGLE: "Angle encoding is too simple to define an implicit kernel: without entanglement, the feature space is insufficiently expressive.",
        EncodingType.AMPLITUDE: "Amplitude encoding compresses the data but doesn't create the entanglement needed for a rich quantum kernel.",
        EncodingType.BASIS: "Basis encoding is for binary data — quantum kernels work with continuous data in Hilbert space.",
        EncodingType.DATA_REUPLOADING: "Data re-uploading is more QNN/VQC-oriented; CZ/ZZ feature maps have a structure better suited for QSVM.",
    },
}


def _explain_vs_alternatives_pt(
    recommended: EncodingType,
    profile: DataProfile,
    context: ProblemContext,
) -> str:
    alts = _ALTERNATIVES_PT.get(recommended, {})
    if not alts:
        return ""
    lines = ["**Por que não os outros encodings?**"]
    for enc, reason in alts.items():
        lines.append(f"- *{enc.value}*: {reason}")
    return "\n".join(lines)


def _explain_vs_alternatives_en(
    recommended: EncodingType,
    profile: DataProfile,
    context: ProblemContext,
) -> str:
    alts = _ALTERNATIVES_EN.get(recommended, {})
    if not alts:
        return ""
    lines = ["**Why not the other encodings?**"]
    for enc, reason in alts.items():
        lines.append(f"- *{enc.value}*: {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geração de código Qiskit completo e copiável
# ---------------------------------------------------------------------------

def generate_qiskit_code(
    encoding: EncodingType,
    data_sample: list[float],
    profile: DataProfile,
    n_qubits: int | None = None,
    lang: str = "pt",
) -> str:
    """
    Gera código Python/Qiskit completo e copiável para o encoding recomendado.
    O código inclui: imports, dados de exemplo, construção do circuito e visualização.
    """
    generators = {
        EncodingType.AMPLITUDE: _code_amplitude,
        EncodingType.ANGLE: _code_angle,
        EncodingType.BASIS: _code_basis,
        EncodingType.DATA_REUPLOADING: _code_data_reuploading,
        EncodingType.CUSTOM_FEATURE_MAP: _code_custom_feature_map,
    }
    fn = generators.get(encoding)
    if fn is None:
        return f"# Encoding '{encoding.value}' não suportado para geração de código."
    return fn(data_sample, profile, n_qubits, lang)


def _header(encoding: EncodingType, lang: str) -> str:
    if lang == "en":
        return (
            f"# Qiskit code — {encoding.value} encoding\n"
            f"# Generated by llama-qiskit-agents\n"
            f"# Run: pip install qiskit qiskit-aer\n"
        )
    return (
        f"# Código Qiskit — {encoding.value} encoding\n"
        f"# Gerado por llama-qiskit-agents\n"
        f"# Instale: pip install qiskit qiskit-aer\n"
    )


def _simulation_block(lang: str) -> str:
    if lang == "en":
        return '''
# ── Simulate ──────────────────────────────────────────────────────────────
from qiskit_aer import AerSimulator
from qiskit import transpile

qc_measured = qc.copy()
qc_measured.measure_all()

sim = AerSimulator()
transpiled = transpile(qc_measured, sim)
result = sim.run(transpiled, shots=1024).result()
counts = result.get_counts()
print("Measurement counts:", counts)
'''
    return '''
# ── Simulação ─────────────────────────────────────────────────────────────
from qiskit_aer import AerSimulator
from qiskit import transpile

qc_measured = qc.copy()
qc_measured.measure_all()

sim = AerSimulator()
transpiled = transpile(qc_measured, sim)
result = sim.run(transpiled, shots=1024).result()
counts = result.get_counts()
print("Contagens de medição:", counts)
'''


def _draw_block(lang: str) -> str:
    if lang == "en":
        return "\n# Draw the circuit\nprint(qc.draw('text'))\n"
    return "\n# Visualizar o circuito\nprint(qc.draw('text'))\n"


def _code_amplitude(
    data: list[float], profile: DataProfile, n_qubits: int | None, lang: str
) -> str:
    import math
    n = n_qubits or max(1, math.ceil(math.log2(max(1, len(data)))))
    data_repr = repr(data)
    if lang == "en":
        comment_norm = "# Normalize the vector (required by amplitude encoding)"
        comment_enc = "# StatePreparation encodes the vector as quantum amplitudes"
        comment_tradeoff = (
            f"# Trade-off: only {n} qubit(s) for {len(data)} values,\n"
            f"# but StatePreparation creates a deep circuit."
        )
    else:
        comment_norm = "# Normalizar o vetor (obrigatório para amplitude encoding)"
        comment_enc = "# StatePreparation codifica o vetor como amplitudes quânticas"
        comment_tradeoff = (
            f"# Trade-off: apenas {n} qubit(s) para {len(data)} valores,\n"
            f"# mas StatePreparation gera um circuito mais profundo."
        )

    return f"""{_header(EncodingType.AMPLITUDE, lang)}
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

data = np.array({data_repr}, dtype=float)

{comment_norm}
norm = np.linalg.norm(data)
if norm > 1e-10:
    data = data / norm

n_amplitudes = 2 ** {n}
if len(data) < n_amplitudes:
    data = np.pad(data, (0, n_amplitudes - len(data)))
else:
    data = data[:n_amplitudes]

{comment_enc}
qc = QuantumCircuit({n}, name="amplitude")
qc.append(StatePreparation(data.astype(complex)), range({n}))

{comment_tradeoff}
{_draw_block(lang)}{_simulation_block(lang)}"""


def _code_angle(
    data: list[float], profile: DataProfile, n_qubits: int | None, lang: str
) -> str:
    n = n_qubits or max(1, len(data))
    data_repr = repr(data[:n] if len(data) >= n else data + [0.0] * (n - len(data)))
    if lang == "en":
        comment = f"# 1 qubit per feature — {n} features → {n} qubit(s)\n# Each feature becomes an Ry rotation angle"
        tradeoff = f"# Trade-off: very shallow circuit (depth ≈ 1), {n} qubit(s) needed."
    else:
        comment = f"# 1 qubit por feature — {n} features → {n} qubit(s)\n# Cada feature vira um ângulo de rotação Ry"
        tradeoff = f"# Trade-off: circuito muito raso (profundidade ≈ 1), {n} qubit(s) necessários."

    return f"""{_header(EncodingType.ANGLE, lang)}
import numpy as np
from qiskit import QuantumCircuit

data = np.array({data_repr}, dtype=float)

{comment}
qc = QuantumCircuit({n}, name="angle")
for i, val in enumerate(data):
    qc.ry(val, i)

# {tradeoff}
{_draw_block(lang)}{_simulation_block(lang)}"""


def _code_basis(
    data: list[float], profile: DataProfile, n_qubits: int | None, lang: str
) -> str:
    n = n_qubits or max(1, len(data))
    data_repr = repr(data[:n] if len(data) >= n else data + [0.0] * (n - len(data)))
    if lang == "en":
        comment = "# Binarize: non-zero → 1, zero → 0\n# Apply X gate (bit-flip) for each '1' bit"
        tradeoff = f"# Trade-off: minimal depth (only X gates), {n} qubit(s). Best for binary/categorical data."
    else:
        comment = "# Binarizar: não-zero → 1, zero → 0\n# Aplicar porta X (bit-flip) para cada bit '1'"
        tradeoff = f"# Trade-off: profundidade mínima (apenas portas X), {n} qubit(s). Ideal para dados binários/categóricos."

    return f"""{_header(EncodingType.BASIS, lang)}
import numpy as np
from qiskit import QuantumCircuit

data = np.array({data_repr}, dtype=float)

{comment}
binary = (data != 0).astype(int)
n_qubits = {n}
qc = QuantumCircuit(n_qubits, name="basis")
for i, b in enumerate(binary[:n_qubits]):
    if b:
        qc.x(i)

# {tradeoff}
{_draw_block(lang)}{_simulation_block(lang)}"""


def _code_data_reuploading(
    data: list[float], profile: DataProfile, n_qubits: int | None, lang: str
) -> str:
    n = n_qubits or max(1, len(data))
    n_layers = 2
    data_repr = repr(data[:n] if len(data) >= n else data + [0.0] * (n - len(data)))
    if lang == "en":
        comment = (
            f"# {n_layers} layers of Ry rotations + CNOT entanglement\n"
            f"# Re-uploading the data in each layer increases expressibility"
        )
        tradeoff = (
            f"# Trade-off: {n_layers} layers × {n} qubit(s) = medium depth.\n"
            f"# Best for QNN / VQC where expressibility is critical."
        )
    else:
        comment = (
            f"# {n_layers} camadas de rotações Ry + entrelaçamento CNOT\n"
            f"# Re-inserir o dado a cada camada aumenta a expressibilidade"
        )
        tradeoff = (
            f"# Trade-off: {n_layers} camadas × {n} qubit(s) = profundidade média.\n"
            f"# Ideal para QNN / VQC onde expressibilidade é crítica."
        )

    return f"""{_header(EncodingType.DATA_REUPLOADING, lang)}
import numpy as np
from qiskit import QuantumCircuit

data = np.array({data_repr}, dtype=float)
n_layers = {n_layers}

{comment}
qc = QuantumCircuit({n}, name="reupload")
for layer in range(n_layers):
    for i, val in enumerate(data[:{n}]):
        qc.ry(val, i)
    for i in range({n} - 1):
        qc.cx(i, i + 1)

# {tradeoff}
{_draw_block(lang)}{_simulation_block(lang)}"""


def _code_custom_feature_map(
    data: list[float], profile: DataProfile, n_qubits: int | None, lang: str
) -> str:
    n = n_qubits or max(1, len(data))
    data_repr = repr(data[:n] if len(data) >= n else data + [0.0] * (n - len(data)))
    if lang == "en":
        comment = (
            "# Per layer: H + Rz + Ry on each qubit, then full pairwise CZ entanglement\n"
            "# ZZ-like feature map — rich Hilbert space for quantum kernel methods"
        )
        tradeoff = (
            f"# Trade-off: deep circuit (H + Rz + Ry + CZ pairs), {n} qubit(s).\n"
            f"# Best for QSVM / quantum kernel similarity computation."
        )
    else:
        comment = (
            "# Por camada: H + Rz + Ry em cada qubit, depois CZ pareado completo\n"
            "# Feature map ZZ-like — espaço de Hilbert rico para kernels quânticos"
        )
        tradeoff = (
            f"# Trade-off: circuito profundo (H + Rz + Ry + pares CZ), {n} qubit(s).\n"
            f"# Ideal para QSVM / cálculo de similaridade por kernel quântico."
        )

    return f"""{_header(EncodingType.CUSTOM_FEATURE_MAP, lang)}
import numpy as np
from qiskit import QuantumCircuit

data = np.array({data_repr}, dtype=float)
n_layers = 1

{comment}
qc = QuantumCircuit({n}, name="custom_fm")
for layer in range(n_layers):
    for i in range({n}):
        qc.h(i)
        qc.rz(data[i] if i < len(data) else 0.0, i)
        qc.ry(data[i] if i < len(data) else 0.0, i)
    for i in range({n}):
        for j in range(i + 1, {n}):
            qc.cz(i, j)

# {tradeoff}
{_draw_block(lang)}{_simulation_block(lang)}"""
