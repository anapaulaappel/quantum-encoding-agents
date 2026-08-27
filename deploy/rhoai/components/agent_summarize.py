"""
Componente KFP: turno agentico via POST /v1/agent/chat (LLM + tools no pod da API).

Requer AGENT_* configurado no deployment da API (Model Serving / Ollama).
Não registra MLflow — ver RHOAI-AGENTS.md (MLflow planejado para v2).
"""

from kfp import dsl
from kfp.dsl import Output, Artifact


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests>=2.31"],
)
def agent_summarize(
    api_url: str,
    csv_content: str,
    agent_output: Output[Artifact],
    problem_description: str = "",
    persona: str = "expert",
    label_column: str = "",
    max_tool_rounds: int = 6,
):
    """
    Chama /v1/agent/chat com CSV anexado; salva reply + tool_calls como JSON.
    """
    import json
    import requests

    message = problem_description.strip() or (
        "Analise o CSV anexado, compare encodings quando fizer sentido "
        "e recomende a melhor estratégia de encoding quântico com justificativa."
    )

    payload: dict = {
        "message": message,
        "persona": persona if persona in ("default", "expert", "mentor") else "expert",
        "csv_content": csv_content,
        "max_tool_rounds": max_tool_rounds,
    }
    if label_column:
        payload["label_column"] = label_column

    response = requests.post(
        f"{api_url.rstrip('/')}/v1/agent/chat",
        json=payload,
        timeout=900,
    )
    response.raise_for_status()
    data = response.json()

    artifact = {
        "reply": data.get("reply", ""),
        "persona": data.get("persona"),
        "backend": data.get("backend"),
        "tool_calls": data.get("tool_calls", []),
        "problem_description": message,
    }

    with open(agent_output.path, "w") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(f"Agent ({data.get('persona')}) backend={data.get('backend')}")
    print(f"Tools: {[t.get('name') for t in data.get('tool_calls', [])]}")
    print(data.get("reply", "")[:2000])
