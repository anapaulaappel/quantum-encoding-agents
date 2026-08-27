"""
Componente KFP: compara encodings via POST /v1/tools/dispatch (compare_csv_embeddings).

Usa o contrato unificado de tools — mesmo endpoint que OpenClaw, Workbench e agentes.
"""

from kfp import dsl
from kfp.dsl import Output, Artifact


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests>=2.31"],
)
def compare_csv_via_dispatch(
    api_url: str,
    csv_content: str,
    comparison_output: Output[Artifact],
    problem_description: str = "",
    label_column: str = "",
    shots: int = 512,
):
    """
    Etapa: compare_csv_embeddings via /v1/tools/dispatch.
    Com labels no CSV, o relatório inclui ranking por KTA.
    """
    import requests

    arguments: dict = {
        "csv_content": csv_content,
        "shots": shots,
    }
    if problem_description:
        arguments["problem_description"] = problem_description
    if label_column:
        arguments["label_column"] = label_column

    response = requests.post(
        f"{api_url.rstrip('/')}/v1/tools/dispatch",
        json={"name": "compare_csv_embeddings", "arguments": arguments},
        timeout=600,
    )
    response.raise_for_status()
    report = response.json()["result"]

    with open(comparison_output.path, "w") as f:
        f.write(report)

    preview = "\n".join(report.splitlines()[:15])
    print(preview)
    if len(report.splitlines()) > 15:
        print("...")
