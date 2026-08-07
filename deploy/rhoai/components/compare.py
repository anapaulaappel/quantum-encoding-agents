"""
Componente KFP: compara todos os 7 encodings via API llama-qiskit-agents.
Gera o relatório de ranking completo.
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests>=2.31"],
)
def compare_encodings(
    api_url: str,
    profile_input: Input[Artifact],
    comparison_output: Output[Artifact],
    task: str = "",
    algorithm: str = "",
    shots: int = 512,
):
    """
    Etapa 3 do pipeline: POST /v1/compare com todos os 7 encodings.
    Salva o relatório de ranking plain-text.
    """
    import json
    import requests

    with open(profile_input.path) as f:
        profile = json.load(f)

    sample = profile.get("_sample_used", [0.1, 0.2, 0.3])

    payload: dict = {"data": sample, "shots": shots}
    if task:
        payload["task"] = task
    if algorithm:
        payload["algorithm"] = algorithm

    response = requests.post(
        f"{api_url}/v1/compare",
        json=payload,
        timeout=300,   # comparar 7 encodings pode levar tempo
    )
    response.raise_for_status()
    report = response.text

    with open(comparison_output.path, "w") as f:
        f.write(report)

    # Log resumo
    lines = [l for l in report.split("\n") if l.strip().startswith(("1.", "2.", "3."))]
    print("Top 3 encodings:")
    for line in lines[:3]:
        print(" ", line.strip())
