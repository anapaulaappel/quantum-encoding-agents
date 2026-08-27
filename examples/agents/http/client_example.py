"""Exemplo: agente externo (LangGraph, CrewAI, etc.) via HTTP dispatch."""

from __future__ import annotations

import json
import os
import urllib.request


def dispatch(name: str, arguments: dict, api_base: str | None = None) -> str:
    base = (api_base or os.environ.get("QISKIT_API_URL", "http://localhost:8080")).rstrip("/")
    payload = json.dumps({"name": name, "arguments": arguments}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/v1/tools/dispatch",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["result"]


if __name__ == "__main__":
    print(dispatch("analyze_data", {"dataset_or_description": "binary data, 10 features"}))
    print(
        dispatch(
            "recommend_embedding_strategy",
            {
                "dataset_or_description": "binary data, 10 features",
                "task": "classification",
                "algorithm": "QSVM",
            },
        )
    )
