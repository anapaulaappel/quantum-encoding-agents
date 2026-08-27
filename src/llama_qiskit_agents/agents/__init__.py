"""Agentes: tools quânticas, harness ReAct e clientes LLM."""

from llama_qiskit_agents.agents.client import chat_completion, create_client
from llama_qiskit_agents.agents.encoding_agent import (
    analyze_data,
    compare_csv_embeddings,
    compare_embeddings_report,
    compute_quantum_kernel,
    dispatch_tool,
    encoding_agent_tools_for_llama,
    explain_tradeoffs,
    generate_qiskit_circuit,
    get_tool_definitions,
    list_tool_names,
    openclaw_skill_markdown,
    recommend_embedding_strategy,
    scenarios_guide,
    simulate_circuit,
)
from llama_qiskit_agents.agents.harness import (
    AgentTurnResult,
    SYSTEM_PROMPTS,
    build_backend,
    run_agent_turn,
)

__all__ = [
    "create_client",
    "chat_completion",
    "analyze_data",
    "recommend_embedding_strategy",
    "generate_qiskit_circuit",
    "simulate_circuit",
    "compare_embeddings_report",
    "compare_csv_embeddings",
    "compute_quantum_kernel",
    "explain_tradeoffs",
    "scenarios_guide",
    "encoding_agent_tools_for_llama",
    "get_tool_definitions",
    "list_tool_names",
    "openclaw_skill_markdown",
    "dispatch_tool",
    "AgentTurnResult",
    "SYSTEM_PROMPTS",
    "build_backend",
    "run_agent_turn",
]
