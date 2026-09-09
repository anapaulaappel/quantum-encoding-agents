"""
Registro unificado de ferramentas quânticas para agentes (OpenClaw, Ollama, Llama Stack, HTTP).

Toda lógica exposta a LLMs passa por ``dispatch_tool``; endpoints REST e CLI delegam aqui.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np

from llama_qiskit_agents.quantum.data_analysis import (
    infer_data_profile,
    load_csv_from_string_with_labels,
    feature_names_from_csv_text,
    recommend_encoding,
    effective_n_features,
)
from llama_qiskit_agents.quantum.encodings import EncodingType, build_encoding_circuit
from llama_qiskit_agents.quantum.hardware_profile import HardwareProfile
from llama_qiskit_agents.quantum.kernel import compute_kernel, kernel_caption
from llama_qiskit_agents.quantum.problem_context import scenario_guide_when_unspecified
from llama_qiskit_agents.quantum.simulate import (
    compare_embeddings,
    compare_embeddings_report,
    explain_tradeoffs as _explain_tradeoffs,
    format_comparison_report,
    simulate_encoding_circuit,
)

ToolFormat = Literal["openai", "openclaw"]

ENCODING_NAMES = [e.value for e in EncodingType]

_TASK_ENUM = ["classification", "clustering", "encoding", "kernel", "variational"]


def _parse_encoding(name: str) -> EncodingType:
    return EncodingType(name.strip().lower().replace(" ", "_"))


def _parse_numeric_data(data: Any, default: list[float] | None = None) -> list[float] | list[list[float]] | str:
    if data is None:
        return default or [0.1, 0.2, 0.3]
    if isinstance(data, str):
        stripped = data.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return parsed
            except json.JSONDecodeError:
                return data
        return data
    return data


def _flatten_sample(data: Any) -> list[float]:
    parsed = _parse_numeric_data(data)
    arr = np.asarray(parsed, dtype=float).flatten()
    if len(arr) == 0:
        arr = np.array([0.1, 0.2, 0.3])
    return arr.tolist()


def _hardware_from_dict(raw: dict[str, Any] | None) -> HardwareProfile | None:
    if not raw:
        return None
    return HardwareProfile(
        gate_error_rate=raw.get("gate_error_rate"),
        max_depth_budget=raw.get("max_depth_budget"),
        max_qubits=raw.get("max_qubits"),
        connectivity=raw.get("connectivity") or "all-to-all",
        backend_name=raw.get("backend_name"),
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _format_profile_line(profile) -> str:
    line = (
        f"Perfil: n_samples={profile.n_samples}, n_features={profile.n_features}, "
        f"binário={profile.is_binary}, categórico={profile.is_categorical}, "
        f"contínuo={profile.is_continuous}, has_negative={profile.has_negative}"
    )
    if profile.intrinsic_dimension is not None:
        cols = profile.selected_columns or []
        names = profile.selected_feature_names
        col_txt = ", ".join(names) if names else ", ".join(str(c) for c in cols)
        line += (
            f", E={profile.embedding_dimension}, D2={profile.intrinsic_dimension:.4f}, "
            f"qubit_budget={profile.qubit_budget}, n_used={effective_n_features(profile)}, "
            f"selected_columns=[{col_txt}], "
            f"fractal_selection_applied={profile.fractal_selection_applied}"
        )
    else:
        line += (
            f", intrinsic_dimension={profile.intrinsic_dimension}, "
            f"qubit_budget={profile.qubit_budget}, selected_columns={profile.selected_columns}"
        )
    return f"{line}, descrição={profile.description}"


def analyze_data(
    dataset_or_description: str | list[float] | list[list[float]] | None = None,
    csv_path: str | None = None,
    apply_fractal_budget: bool = True,
) -> str:
    """Analisa dataset ou descrição e devolve DataProfile, incluindo D2/q*/FD-ASE quando houver matriz."""
    data = csv_path if csv_path else dataset_or_description
    if data is None:
        return "Nenhum dado fornecido. Passe dataset_or_description ou csv_path."
    profile = infer_data_profile(data, apply_fractal_budget=apply_fractal_budget)
    return _format_profile_line(profile)


def recommend_embedding_strategy(
    dataset_or_description: str | list[float] | list[list[float]] | None = None,
    csv_path: str | None = None,
    task: str | None = None,
    algorithm: str | None = None,
    problem_description: str | None = None,
    hardware_profile: dict[str, Any] | None = None,
    apply_fractal_budget: bool = True,
) -> str:
    """Recomenda encoding; com matriz 2D aplica FD-ASE antes (D2, q*, colunas)."""
    data = csv_path if csv_path else dataset_or_description
    if data is None:
        return "Nenhum dado fornecido. Passe dataset_or_description ou csv_path."
    profile = infer_data_profile(data, apply_fractal_budget=apply_fractal_budget)
    hw = _hardware_from_dict(hardware_profile)
    encoding, reason, ctx = recommend_encoding(
        profile,
        task=task,
        algorithm=algorithm,
        problem_description=problem_description,
        hardware_profile=hw,
    )
    extra = ""
    if ctx.has_explicit_info():
        extra = f" Contexto QML: tarefa={ctx.task.value}"
        if ctx.algorithm:
            extra += f", algoritmo={ctx.algorithm}"
        extra += "."
    hw_note = " Hardware profile aplicado." if hw else ""
    fractal = _format_profile_line(profile)
    return f"Recomendação: {encoding.value}.{extra} Motivo: {reason}{hw_note}\n{fractal}"


def generate_qiskit_circuit(
    encoding_name: str,
    data: list[float] | list[list[float]] | str,
    n_qubits: int | None = None,
) -> str:
    try:
        enc = _parse_encoding(encoding_name)
    except ValueError:
        return f"Encoding inválido. Use um de: {ENCODING_NAMES}"
    arr = _flatten_sample(data)
    qc = build_encoding_circuit(enc, arr, n_qubits=n_qubits)
    return f"Circuito {encoding_name}: {qc.num_qubits} qubits, profundidade {qc.depth()}.\n{qc.draw('text')}"


def simulate_circuit(
    encoding_name: str,
    data: list[float] | str,
    n_qubits: int | None = None,
    shots: int = 1024,
) -> str:
    try:
        enc = _parse_encoding(encoding_name)
    except ValueError:
        return f"Encoding inválido. Use um de: {ENCODING_NAMES}"
    arr = _flatten_sample(data)
    qc = build_encoding_circuit(enc, arr, n_qubits=n_qubits)
    res = simulate_encoding_circuit(qc, enc, shots=shots)
    top = sorted(res.counts.items(), key=lambda x: -x[1])[:10]
    return (
        f"Simulação {encoding_name}: {res.num_qubits} qubits, profundidade {res.depth}, shots={shots}. "
        f"Top medições: {dict(top)}"
    )


def compare_embeddings_report_tool(
    data: str | list[float] | list[list[float]],
    n_qubits: int | None = None,
    shots: int = 1024,
    task: str | None = None,
    algorithm: str | None = None,
    problem_description: str | None = None,
    apply_fractal_budget: bool = True,
    sweep_qubit_budget: bool = True,
) -> str:
    parsed = _parse_numeric_data(data)
    return compare_embeddings_report(
        parsed,
        n_qubits=n_qubits,
        shots=shots,
        task=task,
        algorithm=algorithm,
        problem_description=problem_description,
        apply_fractal_budget=apply_fractal_budget,
        sweep_qubit_budget=sweep_qubit_budget,
    )


def compare_csv_embeddings(
    csv_content: str,
    problem_description: str | None = None,
    shots: int = 1024,
    n_qubits: int | None = None,
    label_column: str | None = None,
    optimize_features: bool = False,
    optimization_encoding: str | None = None,
    apply_fractal_budget: bool = True,
    sweep_qubit_budget: bool = True,
) -> str:
    if not csv_content or not csv_content.strip():
        return "csv_content vazio."
    try:
        arr, labels = load_csv_from_string_with_labels(
            csv_content,
            label_column=(label_column or "").strip() or None,
        )
        col_names = feature_names_from_csv_text(
            csv_content,
            label_column=(label_column or "").strip() or None,
        )
    except ValueError as exc:
        return f"Erro ao ler CSV: {exc}"
    opt_enc: EncodingType | None = None
    if optimization_encoding:
        try:
            opt_enc = _parse_encoding(optimization_encoding)
        except ValueError:
            return f"optimization_encoding inválido. Use um de: {ENCODING_NAMES}"
    cr = compare_embeddings(
        arr,
        n_qubits=n_qubits,
        shots=max(1, shots),
        problem_description=(problem_description or "").strip() or None,
        labels=labels,
        optimize_features=optimize_features,
        optimization_encoding=opt_enc,
        feature_column_names=col_names,
        apply_fractal_budget=apply_fractal_budget,
        sweep_qubit_budget=sweep_qubit_budget,
    )
    return format_comparison_report(cr)


def compute_quantum_kernel(
    data: list[list[float]] | str,
    encoding_name: str = "custom_feature_map",
    labels: list[int] | None = None,
    n_qubits: int | None = None,
    lang: str = "pt",
) -> str:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            return f"data deve ser JSON array N×F: {exc}"
    if not isinstance(data, list) or len(data) < 2:
        return "Mínimo de 2 amostras (lista de listas de features)."
    try:
        enc = _parse_encoding(encoding_name)
    except ValueError:
        return f"Encoding inválido. Use um de: {ENCODING_NAMES}"
    try:
        result = compute_kernel(
            data=data,
            encoding_type=enc,
            labels=labels,
            n_qubits=n_qubits,
            lang=lang,
        )
    except Exception as exc:
        return f"Erro ao calcular kernel: {exc}"
    cap = kernel_caption(result.n_samples, enc.value, result.stats, lang=lang)
    kta = result.stats.get("kta")
    kta_line = f"\nKTA = {kta:.4f}" if kta is not None else ""
    alive = result.stats.get("kernel_alive")
    alive_line = ""
    if alive is not None:
        status = "ALIVE" if alive >= 0.5 else "DEAD"
        alive_line = (
            f"\nKernel alive: {status} "
            f"(near={result.stats.get('fid_near', 0):.3f}, "
            f"far={result.stats.get('fid_far', 0):.3f}, "
            f"ratio={result.stats.get('near_far_ratio', 0):.2f})"
        )
    n = result.n_samples
    preview = result.kernel_matrix[: min(5, n), : min(5, n)].tolist()
    return (
        f"Kernel {enc.value}: {n} amostras × {result.n_features} features.{kta_line}{alive_line}\n"
        f"Stats: {result.stats}\n"
        f"Preview K (até 5×5): {preview}\n"
        f"{cap}"
    )


def explain_tradeoffs() -> str:
    return _explain_tradeoffs()


def scenarios_guide() -> str:
    return scenario_guide_when_unspecified()


# ---------------------------------------------------------------------------
# Schemas (OpenAI function-calling)
# ---------------------------------------------------------------------------

_OPENAI_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": "Analisa dataset ou descrição textual e retorna DataProfile (dimensão, tipos, D2/q*/FD-ASE).",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_or_description": {"type": "string"},
                    "csv_path": {"type": "string", "description": "Caminho local CSV (CLI); preferir csv_content na API."},
                    "apply_fractal_budget": {
                        "type": "boolean",
                        "default": True,
                        "description": "Se True, recorta FD-ASE antes de recomendar. Se False, usa todas as colunas mas ainda devolve D2 e a seleção.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_embedding_strategy",
            "description": "Recomenda encoding quântico com base no perfil, tarefa QML, hardware opcional e orçamento fractal (D2/q*).",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_or_description": {"type": "string"},
                    "task": {"type": "string", "enum": _TASK_ENUM},
                    "algorithm": {"type": "string"},
                    "problem_description": {"type": "string"},
                    "apply_fractal_budget": {
                        "type": "boolean",
                        "default": True,
                        "description": "Se True, recorta FD-ASE antes da recomendação. Se False, recomenda na largura cheia; D2/seleção continuam no perfil.",
                    },
                    "hardware_profile": {
                        "type": "object",
                        "properties": {
                            "gate_error_rate": {"type": "number"},
                            "max_depth_budget": {"type": "integer"},
                            "max_qubits": {"type": "integer"},
                            "connectivity": {"type": "string"},
                            "backend_name": {"type": "string"},
                        },
                    },
                },
                "required": ["dataset_or_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_qiskit_circuit",
            "description": "Gera diagrama ASCII do circuito Qiskit para um encoding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "encoding_name": {"type": "string", "enum": ENCODING_NAMES},
                    "data": {"type": "array", "items": {"type": "number"}},
                    "n_qubits": {"type": "integer"},
                },
                "required": ["encoding_name", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_circuit",
            "description": "Simula encoding no AerSimulator e retorna contagens.",
            "parameters": {
                "type": "object",
                "properties": {
                    "encoding_name": {"type": "string", "enum": ENCODING_NAMES},
                    "data": {"type": "array", "items": {"type": "number"}},
                    "n_qubits": {"type": "integer"},
                    "shots": {"type": "integer", "default": 1024},
                },
                "required": ["encoding_name", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_embeddings_report",
            "description": "Compara os 7 encodings: simulação, ranking, KTA (se labels), kernel-alive (N≥4), sweep q (FD-ASE vs PCA vs prefixo), preprocess, simulabilidade clássica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Descrição ou JSON array de features/amostras."},
                    "n_qubits": {"type": "integer"},
                    "shots": {"type": "integer", "default": 1024},
                    "task": {"type": "string"},
                    "algorithm": {"type": "string"},
                    "problem_description": {"type": "string"},
                    "apply_fractal_budget": {
                        "type": "boolean",
                        "default": True,
                        "description": "Se True, aplica o recorte FD-ASE antes do ranking KTA. D2/seleção são estimados mesmo se False.",
                    },
                    "sweep_qubit_budget": {
                        "type": "boolean",
                        "default": True,
                        "description": "Se True e D2 foi estimado, varre q em FD-ASE vs PCA vs prefixo (probe = angle).",
                    },
                },
                "required": ["data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_csv_embeddings",
            "description": (
                "Compara encodings a partir de CSV (texto). Com labels, ranking por KTA. "
                "Com D2, sweep q FD-ASE vs PCA vs prefixo. "
                "optimize_features=true busca ordem/seleção/peso de colunas (Fioravanti et al.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_content": {"type": "string"},
                    "problem_description": {"type": "string"},
                    "label_column": {"type": "string"},
                    "n_qubits": {"type": "integer"},
                    "shots": {"type": "integer", "default": 1024},
                    "optimize_features": {
                        "type": "boolean",
                        "default": False,
                        "description": "Otimizar ordem/seleção/peso de features por KTA antes do compare.",
                    },
                    "optimization_encoding": {
                        "type": "string",
                        "description": "Encoding fixo para otimização; default = melhor KTA baseline.",
                    },
                    "apply_fractal_budget": {
                        "type": "boolean",
                        "default": True,
                        "description": "Se True, recorta FD-ASE antes do KTA guloso. D2/seleção continuam no relatório se False.",
                    },
                    "sweep_qubit_budget": {
                        "type": "boolean",
                        "default": True,
                        "description": "Se True e D2 foi estimado, varre q em FD-ASE vs PCA vs prefixo (probe = angle).",
                    },
                },
                "required": ["csv_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_quantum_kernel",
            "description": "Calcula matriz K_ij = |⟨φ(x_i)|φ(x_j)⟩|², KTA se labels, e kernel-alive (geometria near/far).",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "N amostras × F features.",
                    },
                    "encoding_name": {"type": "string", "enum": ENCODING_NAMES, "default": "custom_feature_map"},
                    "labels": {"type": "array", "items": {"type": "integer"}},
                    "n_qubits": {"type": "integer"},
                    "lang": {"type": "string", "enum": ["pt", "en"], "default": "pt"},
                },
                "required": ["data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_tradeoffs",
            "description": "Explica trade-offs entre famílias de encoding quântico.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scenarios_guide",
            "description": "Guia de qual encoding tende a servir por tarefa QML (classificação, kernel, etc.).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_TOOL_HANDLERS: dict[str, Any] = {
    "analyze_data": analyze_data,
    "recommend_embedding_strategy": recommend_embedding_strategy,
    "generate_qiskit_circuit": generate_qiskit_circuit,
    "simulate_circuit": simulate_circuit,
    "compare_embeddings_report": compare_embeddings_report_tool,
    "compare_csv_embeddings": compare_csv_embeddings,
    "compute_quantum_kernel": compute_quantum_kernel,
    "explain_tradeoffs": explain_tradeoffs,
    "scenarios_guide": scenarios_guide,
}


def list_tool_names() -> list[str]:
    return list(_TOOL_HANDLERS.keys())


def get_tool_definitions(fmt: ToolFormat = "openai") -> list[dict[str, Any]]:
    if fmt == "openai":
        return list(_OPENAI_TOOL_DEFINITIONS)
    if fmt == "openclaw":
        return _openclaw_skill_tools()
    raise ValueError(f"Formato desconhecido: {fmt}")


def encoding_agent_tools_for_llama() -> list[dict[str, Any]]:
    """Alias retrocompatível."""
    return get_tool_definitions("openai")


def _openclaw_skill_tools() -> list[dict[str, Any]]:
    """Metadados enxutos para skills OpenClaw (HTTP dispatch)."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "http": {
                "method": "POST",
                "url": "/v1/tools/dispatch",
                "body": {"name": t["function"]["name"], "arguments": "<JSON object>"},
            },
        }
        for t in _OPENAI_TOOL_DEFINITIONS
    ]


def openclaw_skill_markdown(api_base: str = "http://llama-qiskit-agents:8080") -> str:
    """Fragmento Markdown para SKILL.md do OpenClaw."""
    lines = [
        "# Quantum encoding tools (llama-qiskit-agents)",
        "",
        f"Base URL: `{api_base}`",
        "",
        "Chame `POST /v1/tools/dispatch` com JSON `{\"name\": \"...\", \"arguments\": {...}}`.",
        "",
        "Liste schemas: `GET /v1/tools?format=openai`.",
        "",
        "## Tools",
        "",
    ]
    for name in list_tool_names():
        handler = _TOOL_HANDLERS[name]
        lines.append(f"- **{name}** — {handler.__doc__.strip().split(chr(10))[0] if handler.__doc__ else ''}")
    return "\n".join(lines)


def dispatch_tool(name: str, arguments: dict[str, Any] | None = None) -> str:
    """Executa ferramenta pelo nome; retorno sempre string para o LLM."""
    args = arguments or {}
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        known = ", ".join(list_tool_names())
        return f"Ferramenta desconhecida: {name}. Disponíveis: {known}."
    try:
        return str(handler(**args))
    except TypeError as exc:
        return f"Argumentos inválidos para {name}: {exc}"
    except Exception as exc:
        return f"Erro em {name}: {exc}"
