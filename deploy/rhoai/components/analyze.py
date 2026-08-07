"""
Componente KFP: analisa o perfil do dado via API llama-qiskit-agents.
Entrada: CSV como string ou dados numéricos JSON.
Saída: ProfileResponse JSON + n_features + tipo de dado.
"""

from kfp import dsl
from kfp.dsl import Output, Artifact, Input


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests>=2.31"],
)
def analyze_data(
    api_url: str,
    csv_content: str,
    profile_output: Output[Artifact],
):
    """
    Etapa 1 do pipeline: POST /v1/analyze com o CSV e salva o perfil.
    """
    import json
    import requests
    import io
    import csv as csvlib

    # Converter CSV para lista de floats (primeira linha como amostra)
    rows = []
    reader = csvlib.reader(io.StringIO(csv_content.strip()))
    for row in reader:
        vals = []
        for cell in row:
            try:
                vals.append(float(cell.strip()))
            except (ValueError, AttributeError):
                pass
        if vals:
            rows.append(vals)

    if not rows:
        raise ValueError("CSV não contém dados numéricos válidos.")

    # POST /v1/analyze com a primeira linha como amostra
    sample = rows[0]
    payload = {"data": sample}
    response = requests.post(f"{api_url}/v1/analyze", json=payload, timeout=60)
    response.raise_for_status()
    profile = response.json()

    # Adicionar metadados úteis para próximas etapas
    profile["_n_rows_csv"] = len(rows)
    profile["_sample_used"] = sample

    with open(profile_output.path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Profile: {profile['n_features']} features, "
          f"binary={profile['is_binary']}, continuous={profile['is_continuous']}")
