"""
Ferramentas para o agente Llama Stack: análise de dados, encoding quântico,
simulação e comparação de embeddings.
"""

import json
from typing import Any

import numpy as np

from llama_qiskit_agents.quantum.data_analysis import (
    get_encoding_tradeoffs,
    infer_data_profile,
    recommend_encoding,
)
from llama_qiskit_agents.quantum.encodings import (
    EncodingType,
    build_encoding_circuit,
)
from llama_qiskit_agents.quantum.simulate import (
    compare_embeddings,
    compare_embeddings_report,
    explain_tradeoffs as _explain_tradeoffs,
    simulate_encoding_circuit,
)


def analyze_data(
    dataset_or_description: str | list[float] | list[list[float]] | None = None,
    csv_path: str | None = None,
) -> str:
    """
    Analisa o tipo de dado (dataset, descrição ou path de CSV).
    Retorna perfil: número de amostras/features, binário/categórico/contínuo.
    """
    data = csv_path if csv_path else dataset_or_description
    if data is None:
        return "Nenhum dado fornecido. Passe dataset_or_description ou csv_path."
    profile = infer_data_profile(data)
    return (
        f"Perfil: n_features={profile.n_features}, "
        f"binário={profile.is_binary}, categórico={profile.is_categorical}, "
        f"contínuo={profile.is_continuous}, descrição={profile.description}"
    )


def recommend_embedding_strategy(
    dataset_or_description: str | list[float] | list[list[float]] | None = None,
    csv_path: str | None = None,
    task: str | None = None,
    algorithm: str | None = None,
    problem_description: str | None = None,
) -> str:
    """
    Recomenda a melhor estratégia de embedding com base no perfil do dado e no problema QML.
    Aceita path de CSV. Opcional: task (classification, clustering, encoding, kernel, variational),
    algorithm (ex.: QSVM, VQC), problem_description (texto livre).
    """
    data = csv_path if csv_path else dataset_or_description
    if data is None:
        return "Nenhum dado fornecido. Passe dataset_or_description ou csv_path."
    profile = infer_data_profile(data)
    encoding, reason, ctx = recommend_encoding(
        profile,
        task=task,
        algorithm=algorithm,
        problem_description=problem_description,
    )
    extra = ""
    if ctx.has_explicit_info():
        extra = f" Contexto QML: tarefa={ctx.task.value}"
        if ctx.algorithm:
            extra += f", algoritmo={ctx.algorithm}"
        extra += "."
    return f"Recomendação: {encoding.value}.{extra} Motivo: {reason}"


def generate_qiskit_circuit(
    encoding_name: str,
    data: list[float] | list[list[float]],
    n_qubits: int | None = None,
) -> str:
    """
    Gera o circuito Qiskit para o encoding escolhido.
    encoding_name: amplitude | angle | dense_angle | iqp | basis | data_reuploading | custom_feature_map
    """
    try:
        enc = EncodingType(encoding_name.strip().lower().replace(" ", "_"))
    except ValueError:
        return f"Encoding inválido. Use um de: {[e.value for e in EncodingType]}"
    arr = np.asarray(data).flatten()
    if len(arr) == 0:
        arr = np.array([0.1, 0.2, 0.3])
    qc = build_encoding_circuit(enc, arr, n_qubits=n_qubits)
    return f"Circuito {encoding_name}: {qc.num_qubits} qubits, profundidade {qc.depth()}. Diagrama:\n{qc.draw('text')}"


def simulate_circuit(
    encoding_name: str,
    data: list[float],
    n_qubits: int | None = None,
    shots: int = 1024,
) -> str:
    """
    Gera o circuito para o encoding, simula no AerSimulator e retorna as contagens.
    """
    try:
        enc = EncodingType(encoding_name.strip().lower().replace(" ", "_"))
    except ValueError:
        return f"Encoding inválido. Use um de: {[e.value for e in EncodingType]}"
    arr = np.asarray(data).flatten()
    if len(arr) == 0:
        arr = np.array([0.1, 0.2, 0.3])
    qc = build_encoding_circuit(enc, arr, n_qubits=n_qubits)
    res = simulate_encoding_circuit(qc, enc, shots=shots)
    top = sorted(res.counts.items(), key=lambda x: -x[1])[:10]
    return (
        f"Simulação {encoding_name}: {res.num_qubits} qubits, profundidade {res.depth}, shots={shots}. "
        f"Top medições: {dict(top)}"
    )


def explain_tradeoffs() -> str:
    """Explica os trade-offs entre amplitude, angle, basis, data re-uploading e feature maps customizados."""
    return _explain_tradeoffs()


def encoding_agent_tools_for_llama() -> list[dict[str, Any]]:
    """
    Retorna lista de definições de tools no formato esperado por APIs de chat
    com function calling (nome, descrição, parameters).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_data",
                "description": "Analisa o tipo de dado (dataset ou descrição em texto) e retorna perfil (binário, categórico, contínuo, dimensão).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_or_description": {
                            "type": "string",
                            "description": "Descrição do dataset em texto ou representação (ex: 'dados binários de 8 bits', '[0.1, 0.2, 0.3]').",
                        },
                    },
                    "required": ["dataset_or_description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommend_embedding_strategy",
                "description": "Recomenda encoding para o dado e para a tarefa QML (classificação, clusterização, kernel/QSVM, variacional, só encoding).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_or_description": {
                            "type": "string",
                            "description": "Descrição do dataset ou dos dados.",
                        },
                        "task": {
                            "type": "string",
                            "description": "classification | clustering | encoding | kernel | variational",
                        },
                        "algorithm": {
                            "type": "string",
                            "description": "Nome do algoritmo (ex.: QSVM, VQC, QAOA).",
                        },
                        "problem_description": {
                            "type": "string",
                            "description": "Texto livre sobre o problema QML.",
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
                "description": "Gera o circuito Qiskit para um tipo de encoding e dados fornecidos.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "encoding_name": {
                            "type": "string",
                            "enum": ["amplitude", "angle", "dense_angle", "iqp", "basis", "data_reuploading", "custom_feature_map"],
                            "description": "Tipo de encoding quântico.",
                        },
                        "data": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Lista de números (features ou amostra).",
                        },
                        "n_qubits": {
                            "type": "integer",
                            "description": "Número de qubits (opcional).",
                        },
                    },
                    "required": ["encoding_name", "data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "simulate_circuit",
                "description": "Simula o circuito de encoding no simulador quântico e retorna as contagens de medição.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "encoding_name": {"type": "string", "enum": ["amplitude", "angle", "dense_angle", "iqp", "basis", "data_reuploading", "custom_feature_map"]},
                        "data": {"type": "array", "items": {"type": "number"}},
                        "n_qubits": {"type": "integer"},
                        "shots": {"type": "integer", "description": "Número de shots da simulação.", "default": 1024},
                    },
                    "required": ["encoding_name", "data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_embeddings_report",
                "description": "Compara todos os encodings no dado: simula cada um e retorna relatório com perfil, recomendação, resultados e trade-offs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "Descrição do dado ou lista de números em JSON string (ex: '[0.1, 0.2, 0.3]').",
                        },
                        "n_qubits": {"type": "integer"},
                        "shots": {"type": "integer", "default": 1024},
                        "task": {"type": "string"},
                        "algorithm": {"type": "string"},
                        "problem_description": {"type": "string"},
                    },
                    "required": ["data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "explain_tradeoffs",
                "description": "Explica os trade-offs entre os encodings: amplitude, angle, basis, data re-uploading e feature maps customizados.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Chama a ferramenta pelo nome com os argumentos (ex.: vindos do LLM) e retorna a resposta em string.
    """
    if name == "analyze_data":
        return analyze_data(arguments.get("dataset_or_description", ""))
    if name == "recommend_embedding_strategy":
        return recommend_embedding_strategy(
            arguments.get("dataset_or_description", ""),
            task=arguments.get("task"),
            algorithm=arguments.get("algorithm"),
            problem_description=arguments.get("problem_description"),
        )
    if name == "generate_qiskit_circuit":
        data = arguments.get("data", [])
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = [0.1, 0.2, 0.3]
        return generate_qiskit_circuit(
            arguments.get("encoding_name", "angle"),
            data,
            n_qubits=arguments.get("n_qubits"),
        )
    if name == "simulate_circuit":
        data = arguments.get("data", [])
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = [0.1, 0.2, 0.3]
        return simulate_circuit(
            arguments.get("encoding_name", "angle"),
            data,
            n_qubits=arguments.get("n_qubits"),
            shots=arguments.get("shots", 1024),
        )
    if name == "compare_embeddings_report":
        data = arguments.get("data", "")
        if isinstance(data, str) and data.strip().startswith("["):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
        return compare_embeddings_report(
            data,
            n_qubits=arguments.get("n_qubits"),
            shots=arguments.get("shots", 1024),
            task=arguments.get("task"),
            algorithm=arguments.get("algorithm"),
            problem_description=arguments.get("problem_description"),
        )
    if name == "explain_tradeoffs":
        return explain_tradeoffs()
    return f"Ferramenta desconhecida: {name}"
