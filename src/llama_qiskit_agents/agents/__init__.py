"""Agentes com Llama Stack."""

from llama_qiskit_agents.agents.client import create_client, chat_completion
from llama_qiskit_agents.agents.encoding_agent import (
    analyze_data,
    recommend_embedding_strategy,
    generate_qiskit_circuit,
    simulate_circuit,
    compare_embeddings_report,
    explain_tradeoffs,
    encoding_agent_tools_for_llama,
    dispatch_tool,
)

__all__ = [
    "create_client",
    "chat_completion",
    "analyze_data",
    "recommend_embedding_strategy",
    "generate_qiskit_circuit",
    "simulate_circuit",
    "compare_embeddings_report",
    "explain_tradeoffs",
    "encoding_agent_tools_for_llama",
    "dispatch_tool",
]
