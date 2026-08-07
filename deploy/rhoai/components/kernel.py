"""
Componente KFP: calcula a matriz de kernel quântico via /v1/kernel.
Opcional — só é executado quando labels de classe são fornecidos.
"""

from kfp import dsl
from kfp.dsl import Output, Input, Artifact


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests>=2.31"],
)
def compute_kernel(
    api_url: str,
    csv_content: str,
    recommendation_input: Input[Artifact],
    kernel_output: Output[Artifact],
    labels_csv: str = "",    # ex: "0,0,1,1,0,1" — rótulos separados por vírgula
    max_samples: int = 20,   # limite para custo O(N²) em simulação clássica
    lang: str = "pt",
):
    """
    Etapa 4 do pipeline (opcional): calcula K[i,j]=|⟨φ(xᵢ)|φ(xⱼ)⟩|² para o dataset.
    Usa o encoding recomendado pela etapa anterior.
    """
    import json
    import io
    import csv as csvlib
    import requests

    with open(recommendation_input.path) as f:
        rec = json.load(f)

    encoding = rec.get("recommended_encoding", "angle")

    # Carregar CSV e extrair amostras
    rows: list[list[float]] = []
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
        raise ValueError("CSV não contém dados numéricos.")

    # Limitar N para viabilidade computacional
    rows = rows[:max_samples]

    # Labels (opcional)
    labels = None
    if labels_csv.strip():
        try:
            labels = [int(x.strip()) for x in labels_csv.split(",")]
            labels = labels[:max_samples]  # alinhar com rows
        except ValueError:
            labels = None

    payload: dict = {
        "data": rows,
        "encoding_name": encoding,
        "lang": lang,
    }
    if labels:
        payload["labels"] = labels

    response = requests.post(
        f"{api_url}/v1/kernel",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()

    with open(kernel_output.path, "w") as f:
        json.dump(result, f, indent=2)

    stats = result.get("stats", {})
    print(f"Kernel {encoding}: {result['n_samples']}x{result['n_samples']}")
    print(f"  off_diagonal_mean: {stats.get('off_diagonal_mean', 'N/A')}")
    print(f"  separability_hint: {stats.get('separability_hint', 'N/A')}")
    if "kta" in stats:
        print(f"  KTA: {stats['kta']}")
    if result.get("heatmap_b64"):
        print("  Heatmap: gerado")
