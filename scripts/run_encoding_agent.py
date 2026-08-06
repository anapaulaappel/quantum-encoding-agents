#!/usr/bin/env python3
"""
Pipeline do agente de encoding: o usuário fornece um dataset ou descrição.
O fluxo:
  1. Analisa o tipo de dado
  2. Recomenda a melhor estratégia de embedding
  3. Gera o circuito Qiskit (e opcionalmente para todos os encodings)
  4. Simula
  5. Compara embeddings
  6. Explica trade-offs

Uso:
  python scripts/run_encoding_agent.py "dados contínuos com 5 features"
  python scripts/run_encoding_agent.py --data 0.1 0.2 0.3 0.4 0.5
  python scripts/run_encoding_agent.py --data 0 1 0 1  # basis
  python scripts/run_encoding_agent.py --csv dados.csv
  python scripts/run_encoding_agent.py --data 0.1 0.2 --task kernel --algorithm QSVM
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_qiskit_agents.quantum.simulate import (
    compare_embeddings_report,
    explain_tradeoffs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agente de encoding quântico: analisa dados, recomenda embedding, gera circuito, simula e compara."
    )
    parser.add_argument(
        "description",
        nargs="?",
        default=None,
        help="Descrição do dataset (ex: 'dados binários de 8 bits') ou omitir e usar --data.",
    )
    parser.add_argument(
        "--data",
        nargs="*",
        type=float,
        default=None,
        help="Lista de números como uma amostra (ex: 0.1 0.2 0.3).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="ARQUIVO",
        help="Caminho para arquivo CSV. Analisa o dataset e recomenda o melhor encoding.",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=None,
        help="Número de qubits (opcional).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Shots da simulação (default 1024).",
    )
    parser.add_argument(
        "--tradeoffs-only",
        action="store_true",
        help="Apenas imprime os trade-offs entre encodings e sai.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Tarefa QML: classification, clustering, encoding, kernel, variational",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Nome do algoritmo (ex.: QSVM, VQC, QAOA)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="Descrição livre do problema (classificação, kernel, etc.)",
    )
    args = parser.parse_args()

    if args.tradeoffs_only:
        print(explain_tradeoffs())
        return

    if args.csv:
        data = args.csv
        print(f"Analisando CSV: {args.csv}\n")
    elif args.data is not None and len(args.data) > 0:
        data = args.data
    elif args.description:
        data = args.description
    else:
        data = [0.1, 0.2, 0.3, 0.4]
        print("Nenhum dado fornecido; usando amostra [0.1, 0.2, 0.3, 0.4].\n")

    report = compare_embeddings_report(
        data,
        n_qubits=args.n_qubits,
        shots=args.shots,
        task=args.task,
        algorithm=args.algorithm,
        problem_description=args.problem,
    )
    print(report)


if __name__ == "__main__":
    main()
