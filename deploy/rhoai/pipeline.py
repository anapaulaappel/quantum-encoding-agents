"""
Kubeflow Pipeline v2 — Quantum Encoding Analysis
Para Red Hat OpenShift AI (RHOAI) com Data Science Pipelines.

Etapas:
  1. analyze   — perfil do dado (n_features, tipo, distribuição)
  2. recommend — encoding ideal + explicação + código Qiskit + Bloch sphere
  3. compare   — ranking comparativo dos 7 encodings
  4. kernel    — matriz de kernel quântico + KTA (opcional, quando labels fornecidos)
  5. log       — registra resultados no MLflow

Pré-requisito: pip install kfp>=2.0
Compilar:      python pipeline.py
Saída:         quantum_encoding_pipeline.yaml
Submeter:      python submit.py (ou via RHOAI UI)
"""

from kfp import dsl, compiler
from components.analyze import analyze_data
from components.recommend import recommend_encoding
from components.compare import compare_encodings
from components.kernel import compute_kernel
from components.log_mlflow import log_to_mlflow


@dsl.pipeline(
    name="quantum-encoding-analysis",
    description=(
        "Pipeline de análise e recomendação de encoding de dados quânticos. "
        "Integra llama-qiskit-agents API com MLflow e RHOAI Data Science Pipelines."
    ),
)
def quantum_encoding_pipeline(
    # ── Dados ───────────────────────────────────────────────────────────────
    csv_content: str = "0.31,0.72,0.55\n0.18,0.90,0.44\n0.62,0.33,0.77",
    labels_csv: str = "",  # "0,0,1" — vazio = sem cálculo de kernel

    # ── Contexto QML ────────────────────────────────────────────────────────
    task: str = "",             # classification | kernel | variational | ...
    algorithm: str = "",        # QSVM, VQC, QNN, ...
    problem_description: str = "",

    # ── Hardware ─────────────────────────────────────────────────────────────
    gate_error_rate: float = 0.0,   # 0 = sem restrição; 5e-3 = IBM Eagle
    connectivity: str = "all-to-all",
    max_depth_budget: int = 0,      # 0 = sem limite

    # ── API e infraestrutura ─────────────────────────────────────────────────
    api_url: str = "http://llama-qiskit-agents.openclaw.svc.cluster.local:8080",
    mlflow_tracking_uri: str = "",  # vazio = só métricas KFP nativas
    experiment_name: str = "quantum-encoding",
    run_name: str = "",
    lang: str = "pt",
    shots: int = 512,
    max_kernel_samples: int = 20,
):
    # ── Etapa 1: Análise do perfil ──────────────────────────────────────────
    analyze_task = analyze_data(
        api_url=api_url,
        csv_content=csv_content,
    )
    analyze_task.set_display_name("1. Analisar perfil do dado")
    analyze_task.set_cpu_request("200m").set_memory_request("256Mi")
    analyze_task.set_cpu_limit("500m").set_memory_limit("512Mi")

    # ── Etapa 2: Recomendação de encoding ───────────────────────────────────
    recommend_task = recommend_encoding(
        api_url=api_url,
        profile_input=analyze_task.outputs["profile_output"],
        task=task,
        algorithm=algorithm,
        problem_description=problem_description,
        gate_error_rate=gate_error_rate,
        connectivity=connectivity,
        max_depth_budget=max_depth_budget,
        lang=lang,
    )
    recommend_task.set_display_name("2. Recomendar encoding (+ Bloch sphere)")
    recommend_task.set_cpu_request("200m").set_memory_request("256Mi")
    recommend_task.set_cpu_limit("500m").set_memory_limit("512Mi")

    # ── Etapa 3: Comparação dos 7 encodings ─────────────────────────────────
    compare_task = compare_encodings(
        api_url=api_url,
        profile_input=analyze_task.outputs["profile_output"],
        task=task,
        algorithm=algorithm,
        shots=shots,
    )
    compare_task.set_display_name("3. Comparar todos os 7 encodings")
    compare_task.set_cpu_request("500m").set_memory_request("512Mi")
    compare_task.set_cpu_limit("2").set_memory_limit("2Gi")

    # ── Etapa 4: Matriz de kernel (condicional) ──────────────────────────────
    # Kubeflow v2 não suporta if/else nativo no pipeline — usamos always-run
    # e o componente trata labels_csv vazio como caso sem kernel.
    kernel_task = compute_kernel(
        api_url=api_url,
        csv_content=csv_content,
        recommendation_input=recommend_task.outputs["recommendation_output"],
        labels_csv=labels_csv,
        max_samples=max_kernel_samples,
        lang=lang,
    )
    kernel_task.set_display_name("4. Calcular kernel quântico (KTA)")
    kernel_task.set_cpu_request("500m").set_memory_request("512Mi")
    kernel_task.set_cpu_limit("2").set_memory_limit("2Gi")

    # ── Etapa 5: Log no MLflow ───────────────────────────────────────────────
    log_task = log_to_mlflow(
        recommendation_input=recommend_task.outputs["recommendation_output"],
        kernel_input=kernel_task.outputs["kernel_output"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    log_task.set_display_name("5. Registrar no MLflow")
    log_task.set_cpu_request("200m").set_memory_request("256Mi")
    log_task.set_cpu_limit("500m").set_memory_limit("512Mi")


if __name__ == "__main__":
    output_file = "quantum_encoding_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=quantum_encoding_pipeline,
        package_path=output_file,
    )
    print(f"Pipeline compilado: {output_file}")
    print("Submeta via RHOAI UI ou com: python submit.py")
