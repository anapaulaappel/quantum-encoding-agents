"""
Harness ReAct mínimo: LLM + function calling → dispatch_tool.

Backends suportados (via env ou argumentos):
  - ollama / openai_compatible — API OpenAI em Ollama/vLLM
  - llama_stack — LlamaStackClient
  - remote — só executa tools localmente (sem LLM; útil para testes)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal

from llama_qiskit_agents.agents.tool_registry import dispatch_tool, get_tool_definitions

Persona = Literal["default", "expert", "mentor"]
BackendKind = Literal["ollama", "openai_compatible", "llama_stack", "remote"]

SYSTEM_PROMPTS: dict[Persona, str] = {
    "default": (
        "You are a quantum machine learning assistant specialized in data encoding selection. "
        "Use the provided tools to analyze data, recommend encodings, simulate circuits, "
        "compare encodings, and compute quantum kernels (K_ij). Prefer tools over guessing. "
        "Respond in the same language as the user (Portuguese or English)."
    ),
    "expert": (
        "You are Circuit, an expert QML engineer. Be concise and metric-driven. "
        "Call tools to obtain KTA, depth, qubit counts, and classical simulability. "
        "Skip elementary quantum mechanics unless asked."
    ),
    "mentor": (
        "You are Quanta, a patient QML mentor. Before recommending an encoding, "
        "briefly explain the physical intuition (Hilbert space, kernels, NISQ depth). "
        "Then use tools to justify with data. Encourage learning."
    ),
}


@dataclass
class AgentTurnResult:
    reply: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.get("id", fn.get("name", "call")), "name": fn.get("name"), "arguments": args})
    return calls


class ChatBackend:
    """Backend OpenAI-compatible (Ollama, vLLM, OpenAI)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "tools": tools, "stream": False}
        return _http_post_json(f"{self.base_url}/chat/completions", payload, headers=headers)


class LlamaStackChatBackend:
    def __init__(self, model: str = "Llama-3.3-70B-Instruct") -> None:
        from llama_qiskit_agents.agents.client import create_client

        self.client = create_client()
        self.model = model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        resp = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            tools=tools,
        )
        choice = resp.choices[0].message
        return {
            "choices": [
                {
                    "message": {
                        "role": choice.role,
                        "content": choice.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (choice.tool_calls or [])
                        ],
                    }
                }
            ]
        }


def build_backend(kind: BackendKind | None = None) -> ChatBackend | LlamaStackChatBackend | None:
    kind = kind or os.environ.get("AGENT_BACKEND", "ollama")  # type: ignore[assignment]
    if kind == "remote":
        return None
    if kind == "llama_stack":
        return LlamaStackChatBackend(model=os.environ.get("AGENT_MODEL", "Llama-3.3-70B-Instruct"))
    base = os.environ.get(
        "OPENAI_API_BASE",
        os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1",
    )
    model = os.environ.get("AGENT_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b"))
    key = os.environ.get("OPENAI_API_KEY")
    return ChatBackend(base_url=base, model=model, api_key=key)


def run_agent_turn(
    user_message: str,
    *,
    history: list[dict[str, Any]] | None = None,
    persona: Persona = "default",
    backend: ChatBackend | LlamaStackChatBackend | None = None,
    max_tool_rounds: int = 8,
) -> AgentTurnResult:
    """
    Um turno de conversa: o LLM pode chamar tools até ``max_tool_rounds`` vezes.
    """
    tools = get_tool_definitions("openai")
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPTS[persona]}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    backend = backend if backend is not None else build_backend()
    tool_log: list[dict[str, Any]] = []

    if backend is None:
        return AgentTurnResult(
            reply=(
                "Backend 'remote': nenhum LLM configurado. "
                "Use dispatch_tool ou POST /v1/tools/dispatch."
            ),
            messages=messages,
        )

    for _ in range(max_tool_rounds):
        try:
            data = backend.chat(messages, tools)
        except urllib.error.URLError as exc:
            return AgentTurnResult(
                reply=f"Erro ao contactar LLM: {exc}. Verifique OLLAMA_HOST / OPENAI_API_BASE.",
                messages=messages,
                tool_calls=tool_log,
            )
        except Exception as exc:
            return AgentTurnResult(
                reply=f"Erro no backend LLM: {exc}",
                messages=messages,
                tool_calls=tool_log,
            )

        msg = data["choices"][0]["message"]
        calls = _parse_tool_calls(msg)

        if not calls:
            content = msg.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            return AgentTurnResult(reply=content, messages=messages, tool_calls=tool_log)

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                }
                for c in calls
            ],
        }
        messages.append(assistant_msg)

        for call in calls:
            result = dispatch_tool(call["name"], call["arguments"])
            tool_log.append({"name": call["name"], "arguments": call["arguments"], "result_preview": result[:500]})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )

    return AgentTurnResult(
        reply="Limite de chamadas de ferramenta atingido. Reformule a pergunta ou aumente max_tool_rounds.",
        messages=messages,
        tool_calls=tool_log,
    )
