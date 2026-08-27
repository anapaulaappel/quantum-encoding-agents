"""
Ferramentas para agentes Llama Stack / OpenClaw / Ollama.

Implementação canônica em ``tool_registry``; este módulo mantém imports legados.
"""

from llama_qiskit_agents.agents.tool_registry import (
    analyze_data,
    compare_csv_embeddings,
    compare_embeddings_report_tool as compare_embeddings_report,
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

__all__ = [
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
]
