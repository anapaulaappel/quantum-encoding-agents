#!/usr/bin/env python3
"""
Harness CLI — conversa com LLM + tools quânticas locais.

Exemplos:
  python scripts/run_agent_harness.py "Tenho CSV com 8 features contínuas, quero QSVM"
  python scripts/run_agent_harness.py --persona mentor "Explique angle vs IQP"
  AGENT_BACKEND=llama_stack python scripts/run_agent_harness.py "Compare encodings para [0.1,0.2,0.3]"
  python scripts/run_agent_harness.py --dispatch recommend_embedding_strategy '{"dataset_or_description":"5 continuous features"}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_qiskit_agents.agents.harness import build_backend, run_agent_turn
from llama_qiskit_agents.agents.tool_registry import dispatch_tool, list_tool_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum encoding agent harness (local LLM + tools)")
    parser.add_argument("message", nargs="?", help="Mensagem do usuário")
    parser.add_argument("--persona", choices=["default", "expert", "mentor"], default="default")
    parser.add_argument("--backend", choices=["ollama", "openai_compatible", "llama_stack", "remote"])
    parser.add_argument("--max-tool-rounds", type=int, default=8)
    parser.add_argument("--list-tools", action="store_true", help="Lista nomes de tools e sai")
    parser.add_argument(
        "--dispatch",
        nargs=2,
        metavar=("NAME", "JSON_ARGS"),
        help="Executa uma tool diretamente (sem LLM)",
    )
    args = parser.parse_args()

    if args.list_tools:
        for name in list_tool_names():
            print(name)
        return 0

    if args.dispatch:
        name, raw_args = args.dispatch
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            print(f"JSON inválido: {exc}", file=sys.stderr)
            return 1
        print(dispatch_tool(name, arguments))
        return 0

    if not args.message:
        parser.print_help()
        return 1

    backend = build_backend(args.backend)
    turn = run_agent_turn(
        args.message,
        persona=args.persona,
        backend=backend,
        max_tool_rounds=args.max_tool_rounds,
    )
    if turn.tool_calls:
        print("--- tool calls ---", file=sys.stderr)
        for tc in turn.tool_calls:
            print(f"  {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)[:120]}...)", file=sys.stderr)
    print(turn.reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
