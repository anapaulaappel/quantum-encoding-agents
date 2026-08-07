"""
Componente KFP: recomenda encoding via API llama-qiskit-agents.
Aceita perfil do dado + contexto QML opcional + hardware_profile opcional.
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests>=2.31"],
)
def recommend_encoding(
    api_url: str,
    profile_input: Input[Artifact],
    recommendation_output: Output[Artifact],
    task: str = "",
    algorithm: str = "",
    problem_description: str = "",
    gate_error_rate: float = 0.0,
    connectivity: str = "all-to-all",
    max_depth_budget: int = 0,
    lang: str = "pt",
):
    """
    Etapa 2 do pipeline: POST /v1/recommend/explain com perfil + contexto QML.
    Salva a recomendação completa (encoding, explicação, código Qiskit, métricas).
    """
    import json
    import requests

    with open(profile_input.path) as f:
        profile = json.load(f)

    sample = profile.get("_sample_used", [0.1, 0.2, 0.3])

    payload: dict = {
        "data": sample,
        "lang": lang,
        "include_bloch": True,   # captura esfera de Bloch quando ≤6 qubits
    }
    if task:
        payload["task"] = task
    if algorithm:
        payload["algorithm"] = algorithm
    if problem_description:
        payload["problem_description"] = problem_description

    # Hardware profile NISQ-aware (apenas se gate_error_rate foi informado)
    if gate_error_rate > 0:
        hw: dict = {"connectivity": connectivity}
        hw["gate_error_rate"] = gate_error_rate
        if max_depth_budget > 0:
            hw["max_depth_budget"] = max_depth_budget
        payload["hardware_profile"] = hw

    response = requests.post(
        f"{api_url}/v1/recommend/explain",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()

    # Adicionar metadados do profile
    result["_profile"] = profile
    result["_input_sample"] = sample

    with open(recommendation_output.path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Recomendado: {result['recommended_encoding']}")
    print(f"Qubits: {result.get('circuit_qubits')}, "
          f"Depth: {result.get('circuit_depth')}")
    print(f"HW constraints applied: {result.get('hardware_constraints_applied')}")
    if result.get("bloch_sphere_b64"):
        print("Bloch sphere: gerado")
