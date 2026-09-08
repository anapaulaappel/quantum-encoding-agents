#!/usr/bin/env python3
"""Demo: D2 / q* / FD-ASE no Breast Cancer (exemplo do arXiv:2609.00475).

Uso (na raiz do repo):
  PYTHONPATH=src python scripts/demo_fractal_breast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from llama_qiskit_agents.quantum.data_analysis import (
    effective_n_features,
    infer_data_profile,
    recommend_encoding,
)
from llama_qiskit_agents.quantum.encodings import build_encoding_circuit
from llama_qiskit_agents.quantum.fractal import apply_column_selection, estimate_fractal_budget


def main() -> None:
    try:
        from sklearn.datasets import load_breast_cancer
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn é necessário. Instale com: pip install -e '.[benchmark]'"
        ) from exc

    data = load_breast_cancer()
    x = np.asarray(data.data, dtype=float)
    names = [str(n) for n in data.feature_names]
    budget = estimate_fractal_budget(x, feature_names=names)
    profile = infer_data_profile(x, feature_names=names)
    enc, reason, _ctx = recommend_encoding(profile)
    sample = apply_column_selection(x[:1], profile.selected_columns)[0]
    qc = build_encoding_circuit(enc, sample)

    print("Breast Cancer (sklearn, 30 atributos originais)")
    print(f"  E            = {budget.embedding_dimension}")
    print(f"  D2           = {budget.intrinsic_dimension}")
    print(f"  q*           = {budget.qubit_budget}")
    print(f"  selected     = {budget.selected_columns}")
    print(f"  names        = {budget.selected_feature_names}")
    print(f"  n_used       = {effective_n_features(profile)}")
    print(f"  encoding     = {enc.value}")
    print(f"  circuit      = {qc.num_qubits} qubits, depth {qc.depth()}")
    print(f"  reason       = {reason}")


if __name__ == "__main__":
    main()
