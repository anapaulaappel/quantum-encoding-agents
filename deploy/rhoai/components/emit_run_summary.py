"""
Componente KFP: consolida artefatos do run em um JSON (sem MLflow).

Substituto leve da etapa de logging quando MLflow não está configurado.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Artifact, Metrics


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests>=2.31"],
)
def emit_run_summary(
    recommendation_input: Input[Artifact],
    comparison_input: Input[Artifact],
    kernel_input: Input[Artifact],
    agent_input: Input[Artifact],
    run_summary_output: Output[Artifact],
    agent_reply_output: Output[Artifact],
    run_metrics: Output[Metrics],
):
    """
    Agrega recomendação, comparação, kernel e (opcional) resumo agentico em um artefato.
    Persiste agent_reply.txt como Output[Artifact] para o Kubeflow Pipelines guardar o ficheiro.
    """
    import json

    with open(recommendation_input.path) as f:
        rec = json.load(f)

    with open(comparison_input.path) as f:
        comparison_report = f.read()

    with open(kernel_input.path) as f:
        kernel_data = json.load(f)

    with open(agent_input.path) as f:
        agent_data = json.load(f)

    profile = rec.get("_profile", {})
    summary = {
        "recommended_encoding": rec.get("recommended_encoding"),
        "task_interpreted": rec.get("task_interpreted"),
        "circuit_depth": rec.get("circuit_depth"),
        "circuit_qubits": rec.get("circuit_qubits"),
        "n_features": profile.get("n_features"),
        "kernel_stats": kernel_data.get("stats"),
        "agent_tool_calls": [t.get("name") for t in agent_data.get("tool_calls", [])],
        "comparison_report_preview": comparison_report[:4000],
    }

    with open(run_summary_output.path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    run_metrics.log_metric("n_features", profile.get("n_features", 0))
    run_metrics.log_metric("circuit_depth", rec.get("circuit_depth") or 0)
    run_metrics.log_metric("circuit_qubits", rec.get("circuit_qubits") or 0)
    if kernel_data.get("stats", {}).get("kta") is not None:
        run_metrics.log_metric("kta", kernel_data["stats"]["kta"])

    with open(agent_reply_output.path, "w") as f:
        f.write(agent_data.get("reply") or "")

    print(json.dumps(summary, indent=2))
