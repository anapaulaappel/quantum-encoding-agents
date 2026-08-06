#!/usr/bin/env python3
"""Exemplo: executa circuitos Qiskit (não depende de Llama Stack)."""

import sys
from pathlib import Path

# Permite importar o pacote sem instalar
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_qiskit_agents.quantum import run_bell_circuit, run_simple_circuit


def main() -> None:
    print("Circuito simples (H no qubit 0):")
    counts = run_simple_circuit(shots=1024)
    print(counts)
    print("\nCircuito de Bell:")
    counts_bell = run_bell_circuit(shots=1024)
    print(counts_bell)


if __name__ == "__main__":
    main()
