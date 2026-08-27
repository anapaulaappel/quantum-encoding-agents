"""
Kubeflow Pipeline v2 — Quantum Encoding Analysis (variante agentica)

Para Red Hat OpenShift AI (RHOAI). Usa:
  - POST /v1/tools/dispatch (compare_csv_embeddings)
  - POST /v1/agent/chat (resumo LLM + tools; requer AGENT_* no deployment da API)

Sem etapa MLflow nesta variante — ver RHOAI-AGENTS.md (MLflow documentado, integração futura).

Compilar:  python pipeline_agents.py
Submeter: python submit.py --variant agents [--dsp-endpoint ...]
"""

from kfp import dsl, compiler
from components.analyze import analyze_data
from components.recommend import recommend_encoding
from components.compare_csv_dispatch import compare_csv_via_dispatch
from components.kernel import compute_kernel
from components.agent_summarize import agent_summarize
from components.emit_run_summary import emit_run_summary


@dsl.pipeline(
    name="quantum-encoding-analysis-agents",
    description=(
        "Pipeline RHOAI com compare via /v1/tools/dispatch e resumo agentico "
        "(/v1/agent/chat). MLflow: ver RHOAI-AGENTS.md (não incluído nesta variante)."
    ),
)
def quantum_encoding_pipeline_agents(
    csv_content: str = "0.31,0.72,0.55\n0.18,0.90,0.44\n0.62,0.33,0.77",
    labels_csv: str = "",
    task: str = "",
    algorithm: str = "",
    problem_description: str = "",
    gate_error_rate: float = 0.0,
    connectivity: str = "all-to-all",
    max_depth_budget: int = 0,
    api_url: str = "http://llama-qiskit-agents.openclaw.svc.cluster.local:8080",
    lang: str = "pt",
    shots: int = 512,
    max_kernel_samples: int = 20,
    agent_persona: str = "expert",
    label_column: str = "",
    max_tool_rounds: int = 6,
):
    analyze_task = analyze_data(api_url=api_url, csv_content=csv_content)
    analyze_task.set_display_name("1. Analisar perfil do dado")

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
    recommend_task.set_display_name("2. Recomendar encoding (+ Bloch)")

    agent_task = agent_summarize(
        api_url=api_url,
        csv_content=csv_content,
        problem_description=problem_description,
        persona=agent_persona,
        label_column=label_column,
        max_tool_rounds=max_tool_rounds,
    )
    agent_task.set_display_name("3. Resumo agentico (/v1/agent/chat)")
    agent_task.set_cpu_request("200m").set_memory_request("256Mi")
    agent_task.set_cpu_limit("1").set_memory_limit("512Mi")

    compare_task = compare_csv_via_dispatch(
        api_url=api_url,
        csv_content=csv_content,
        problem_description=problem_description,
        label_column=label_column,
        shots=shots,
    )
    compare_task.set_display_name("4. Comparar encodings (dispatch + KTA)")
    compare_task.set_cpu_request("500m").set_memory_request("512Mi")
    compare_task.set_cpu_limit("2").set_memory_limit("2Gi")

    kernel_task = compute_kernel(
        api_url=api_url,
        csv_content=csv_content,
        recommendation_input=recommend_task.outputs["recommendation_output"],
        labels_csv=labels_csv,
        max_samples=max_kernel_samples,
        lang=lang,
    )
    kernel_task.set_display_name("5. Kernel quântico K_{ij} + KTA")

    summary_task = emit_run_summary(
        recommendation_input=recommend_task.outputs["recommendation_output"],
        comparison_input=compare_task.outputs["comparison_output"],
        kernel_input=kernel_task.outputs["kernel_output"],
        agent_input=agent_task.outputs["agent_output"],
    )
    summary_task.set_display_name("6. Consolidar artefatos (sem MLflow)")


if __name__ == "__main__":
    out = "quantum_encoding_pipeline_agents.yaml"
    compiler.Compiler().compile(
        pipeline_func=quantum_encoding_pipeline_agents,
        package_path=out,
    )
    print(f"Pipeline compilado: {out}")
    print("Submeta: python submit.py --variant agents [--dsp-endpoint ...]")
