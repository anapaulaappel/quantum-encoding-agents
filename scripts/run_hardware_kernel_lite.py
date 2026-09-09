#!/usr/bin/env python3
"""Kernel-lite on IBM Quantum: 6 overlap pairs, one Sampler job.

Also replays the week1_iris histogram circuits on Aer for a QPU vs simulator table.

  python scripts/run_hardware_kernel_lite.py --backend ibm_fez --execute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.datasets import load_iris_binary  # noqa: E402
from benchmarks.hardware_runner import (  # noqa: E402
    _subsample_stratified,
    build_hardware_job_specs,
)
from llama_qiskit_agents.quantum.encodings import EncodingType, build_encoding_circuit  # noqa: E402
from llama_qiskit_agents.quantum.kernel import kernel_stats  # noqa: E402
from llama_qiskit_agents.quantum.preprocessing import FeatureBounds  # noqa: E402
from llama_qiskit_agents.quantum.visualization import simulate_statevector  # noqa: E402

SHOTS = 512


def _overlap_circuit(xi: np.ndarray, xj: np.ndarray, enc: EncodingType, bounds: FeatureBounds) -> QuantumCircuit:
    qi = build_encoding_circuit(enc, xi, feature_bounds=bounds)
    qj = build_encoding_circuit(enc, xj, feature_bounds=bounds)
    qi.remove_final_measurements(inplace=True)
    qj.remove_final_measurements(inplace=True)
    qc = qi.compose(qj.inverse())
    qc.measure_all()
    return qc


def _p_zero(counts: dict[str, int], n_qubits: int) -> float:
    total = sum(counts.values()) or 1
    zero = "0" * n_qubits
    rev = zero  # identical
    return float(counts.get(zero, 0) + (0 if zero == rev else counts.get(rev, 0))) / float(total)


def _exact_fidelity(xi: np.ndarray, xj: np.ndarray, enc: EncodingType, bounds: FeatureBounds) -> float:
    qi = build_encoding_circuit(enc, xi, feature_bounds=bounds)
    qj = build_encoding_circuit(enc, xj, feature_bounds=bounds)
    si = simulate_statevector(qi)
    sj = simulate_statevector(qj)
    if si is None or sj is None:
        return float("nan")
    return float(abs(np.vdot(si, sj)) ** 2)


def _aer_shots(qc: QuantumCircuit, shots: int, seed: int = 42) -> dict[str, int]:
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    return dict(sim.run(tqc, shots=shots, seed_simulator=seed).result().get_counts())


def _mode(counts: dict[str, int]) -> tuple[str, float]:
    if not counts:
        return "", 0.0
    key = max(counts, key=lambda k: counts[k])
    total = sum(counts.values()) or 1
    return key, float(counts[key]) / float(total)


def replay_histograms_on_aer(iris_json: Path) -> list[dict]:
    payload = json.loads(iris_json.read_text())
    ds = load_iris_binary(max_samples=50)
    result, _, baseline_plan, optimized_plan = build_hardware_job_specs(
        ds, encoding=EncodingType.ANGLE, max_samples=12
    )
    X = np.asarray(ds.X, dtype=float)
    y = ds.y
    X, y = _subsample_stratified(X, y, 12)
    plans = {"baseline": baseline_plan, "optimized": optimized_plan}
    rows: list[dict] = []
    for job in payload["jobs"]:
        spec = job["spec"]
        plan = plans[spec["plan"]]
        mapped = plan.apply(X)
        bounds = FeatureBounds.fit(mapped)
        row = np.asarray(spec["row"], dtype=float)
        qc = build_encoding_circuit(EncodingType.ANGLE, row, feature_bounds=bounds)
        if qc.num_clbits == 0:
            qc.measure_all()
        aer = _aer_shots(qc, SHOTS)
        qpu = job["run"]["counts"]
        qpu_mode, qpu_p = _mode(qpu)
        aer_mode, aer_p = _mode(aer)
        aer_p_qpu_mode = float(aer.get(qpu_mode, 0)) / float(sum(aer.values()) or 1)
        rec = {
            "label": spec["label"],
            "plan": spec["plan"],
            "class_label": spec["class_label"],
            "job_id": job["run"]["job_id"],
            "n_qubits": int(qc.num_qubits),
            "isa_depth": job["transpile"]["depth"],
            "qpu_mode": qpu_mode,
            "qpu_p_mode": round(qpu_p, 3),
            "aer_mode": aer_mode,
            "aer_p_mode": round(aer_p, 3),
            "aer_p_at_qpu_mode": round(aer_p_qpu_mode, 3),
        }
        rows.append(rec)
        print(
            f"  {rec['label']:20s} QPU {qpu_mode} p={qpu_p:.3f}  "
            f"Aer mode={aer_mode} p={aer_p:.3f}  Aer@QPU-mode={aer_p_qpu_mode:.3f}"
        )
    return rows


def kernel_lite_pairs() -> tuple[list[tuple[str, int, int, np.ndarray, np.ndarray]], FeatureBounds, list[int]]:
    ds = load_iris_binary(max_samples=50)
    X = np.asarray(ds.X, dtype=float)
    y = ds.y
    X, y = _subsample_stratified(X, y, 12)
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for i, lab in enumerate(y):
        if len(by_class[int(lab)]) < 2:
            by_class[int(lab)].append(i)
    idx = by_class[0] + by_class[1]
    labels = [y[i] for i in idx]
    bounds = FeatureBounds.fit(X)
    names = ["c0a", "c0b", "c1a", "c1b"]
    samples = [X[i] for i in idx]
    pair_idx = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
    pair_names = ["intra0", "intra1", "inter_0a1a", "inter_0a1b", "inter_0b1a", "inter_0b1b"]
    pairs = []
    for name, (i, j) in zip(pair_names, pair_idx):
        pairs.append((name, i, j, samples[i], samples[j]))
    print("kernel-lite samples:", list(zip(names, labels)))
    return pairs, bounds, labels


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="ibm_fez")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--json", default="benchmarks/results/hardware_kernel_lite.json")
    args = p.parse_args()

    iris_json = ROOT / "benchmarks/results/hardware_iris.json"
    print("=== Aer replay of week1_iris histograms ===")
    hist_rows = replay_histograms_on_aer(iris_json) if iris_json.exists() else []

    print("=== Kernel-lite (6 overlap pairs, angle, Iris) ===")
    pairs, bounds, labels = kernel_lite_pairs()
    enc = EncodingType.ANGLE
    circuits = [_overlap_circuit(xi, xj, enc, bounds) for _, _, _, xi, xj in pairs]
    nq = circuits[0].num_qubits

    lite_rows: list[dict] = []
    for (name, i, j, xi, xj), qc in zip(pairs, circuits):
        exact = _exact_fidelity(xi, xj, enc, bounds)
        aer_counts = _aer_shots(qc, SHOTS)
        aer_k = _p_zero(aer_counts, nq)
        rec = {
            "pair": name,
            "i": i,
            "j": j,
            "exact": round(exact, 3),
            "aer_shots": round(aer_k, 3),
            "qpu": None,
        }
        lite_rows.append(rec)
        print(f"  {name:12s} exact={exact:.3f}  aer={aer_k:.3f}")

    qpu_job_id = None
    if args.execute:
        from llama_qiskit_agents.quantum.hardware_run import run_encoding_circuits

        print(f"Submitting {len(circuits)} overlap circuits as one job on {args.backend}...")
        outcomes = run_encoding_circuits(circuits, backend_name=args.backend, shots=SHOTS)
        qpu_job_id = outcomes[0].job_id if outcomes else None
        print(f"  job_id={qpu_job_id}")
        for rec, out in zip(lite_rows, outcomes):
            rec["qpu"] = round(_p_zero(out.counts, nq), 3)
            rec["qpu_counts_top"] = dict(sorted(out.counts.items(), key=lambda kv: -kv[1])[:3])
            print(f"  {rec['pair']:12s} QPU K={rec['qpu']:.3f}  top={rec['qpu_counts_top']}")

    k_exact = np.eye(4)
    k_aer = np.eye(4)
    k_qpu = np.eye(4)
    for rec in lite_rows:
        i, j = rec["i"], rec["j"]
        k_exact[i, j] = k_exact[j, i] = rec["exact"]
        k_aer[i, j] = k_aer[j, i] = rec["aer_shots"]
        if rec["qpu"] is not None:
            k_qpu[i, j] = k_qpu[j, i] = rec["qpu"]
    kta_exact = kernel_stats(k_exact, labels).get("kta")
    kta_aer = kernel_stats(k_aer, labels).get("kta")
    kta_qpu = kernel_stats(k_qpu, labels).get("kta") if args.execute else None
    print(f"mini-KTA 4×4  exact={kta_exact}  aer={kta_aer}  qpu={kta_qpu}")

    out = {
        "backend": args.backend if args.execute else None,
        "shots": SHOTS,
        "histograms_vs_aer": hist_rows,
        "kernel_lite": lite_rows,
        "mini_kta": {"exact": kta_exact, "aer": kta_aer, "qpu": kta_qpu},
        "qpu_job_id": qpu_job_id,
    }
    path = ROOT / args.json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"JSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
