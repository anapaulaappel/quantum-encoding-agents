"""
Componente KFP: loga artefatos e métricas no MLflow (integrado ao RHOAI).
Salva: encoding recomendado, métricas do circuito, KTA, heatmap, código Qiskit.
"""

from kfp import dsl
from kfp.dsl import Input, Artifact, Output, ClassificationMetrics, Metrics


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests>=2.31", "mlflow>=2.0"],
)
def log_to_mlflow(
    recommendation_input: Input[Artifact],
    kernel_input: Input[Artifact],
    mlflow_metrics: Output[Metrics],
    mlflow_tracking_uri: str = "",
    experiment_name: str = "quantum-encoding-pipeline",
    run_name: str = "",
):
    """
    Etapa 5 do pipeline: loga resultados no MLflow do RHOAI.
    Registra: parâmetros do encoding, métricas do circuito e do kernel.
    """
    import json
    import base64
    import os

    with open(recommendation_input.path) as f:
        rec = json.load(f)

    kernel_data = {}
    try:
        with open(kernel_input.path) as f:
            kernel_data = json.load(f)
    except Exception:
        pass

    # ── Métricas para o artefato KFP nativo ────────────────────────────────
    profile = rec.get("_profile", {})
    mlflow_metrics.log_metric("n_features",       profile.get("n_features", 0))
    mlflow_metrics.log_metric("circuit_depth",    rec.get("circuit_depth") or 0)
    mlflow_metrics.log_metric("circuit_qubits",   rec.get("circuit_qubits") or 0)
    if kernel_data.get("stats"):
        stats = kernel_data["stats"]
        mlflow_metrics.log_metric("kta",               stats.get("kta", 0.0))
        mlflow_metrics.log_metric("separability_hint", stats.get("separability_hint", 0.0))
        mlflow_metrics.log_metric("off_diagonal_mean", stats.get("off_diagonal_mean", 0.0))

    # ── MLflow (se tracking URI configurado) ──────────────────────────────
    if mlflow_tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)

            run_kwargs = {}
            if run_name:
                run_kwargs["run_name"] = run_name

            with mlflow.start_run(**run_kwargs):
                # Parâmetros
                mlflow.log_param("recommended_encoding", rec.get("recommended_encoding"))
                mlflow.log_param("task_interpreted",     rec.get("task_interpreted"))
                mlflow.log_param("lang",                 rec.get("lang"))
                mlflow.log_param("hw_constraints",       rec.get("hardware_constraints_applied"))
                mlflow.log_param("n_features",           profile.get("n_features"))
                mlflow.log_param("is_continuous",        profile.get("is_continuous"))
                mlflow.log_param("is_binary",            profile.get("is_binary"))

                # Métricas do circuito
                if rec.get("circuit_depth") is not None:
                    mlflow.log_metric("circuit_depth",  rec["circuit_depth"])
                    mlflow.log_metric("circuit_qubits", rec["circuit_qubits"])

                # Métricas do kernel
                if kernel_data.get("stats"):
                    stats = kernel_data["stats"]
                    for k, v in stats.items():
                        mlflow.log_metric(f"kernel_{k}", v)

                # Artefatos de texto
                mlflow.log_text(rec.get("explanation", ""), "explanation.md")
                mlflow.log_text(rec.get("qiskit_code", ""), "circuit_code.py")

                # Heatmap do kernel como imagem (se disponível)
                heatmap_b64 = kernel_data.get("heatmap_b64")
                if heatmap_b64:
                    img_bytes = base64.b64decode(heatmap_b64)
                    tmp_path = "/tmp/kernel_heatmap.png"
                    with open(tmp_path, "wb") as img_f:
                        img_f.write(img_bytes)
                    mlflow.log_artifact(tmp_path, artifact_path="visualizations")

                # Bloch sphere (se disponível)
                bloch_b64 = rec.get("bloch_sphere_b64")
                if bloch_b64:
                    img_bytes = base64.b64decode(bloch_b64)
                    tmp_path = "/tmp/bloch_sphere.png"
                    with open(tmp_path, "wb") as img_f:
                        img_f.write(img_bytes)
                    mlflow.log_artifact(tmp_path, artifact_path="visualizations")

                print(f"MLflow run registrado em: {mlflow_tracking_uri}")
        except Exception as exc:
            print(f"MLflow logging falhou (não crítico): {exc}")

    print(f"Encoding recomendado: {rec.get('recommended_encoding')}")
    print(f"Métricas KFP registradas com sucesso.")
