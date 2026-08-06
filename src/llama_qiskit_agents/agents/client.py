"""Cliente Llama Stack para chat e agentes."""

import os
from llama_stack_client import LlamaStackClient


def create_client(base_url: str | None = None, api_key: str | None = None) -> LlamaStackClient:
    """Cria um cliente Llama Stack (usa variáveis de ambiente se não passado)."""
    return LlamaStackClient(
        base_url=base_url or os.environ.get("LLAMA_STACK_CLIENT_BASE_URL"),
        api_key=api_key or os.environ.get("LLAMA_STACK_CLIENT_API_KEY"),
    )


def chat_completion(
    message: str,
    model: str = "Llama-3.3-70B-Instruct",
    client: LlamaStackClient | None = None,
):
    """Envia uma mensagem e retorna a resposta do modelo."""
    if client is None:
        client = create_client()
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": message}],
        model=model,
    )
    if resp.choices:
        return resp.choices[0].message.content
    return None
