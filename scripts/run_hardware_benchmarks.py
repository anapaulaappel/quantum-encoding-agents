#!/usr/bin/env python3
"""
Benchmark em hardware IBM Quantum — preset para ~10 min/mês.

Simulador primeiro (KTA + otimização); hardware valida transpile + execução enxuta.

Uso:
  pip install -e ".[benchmark,ibm]"
  export QISKIT_IBM_TOKEN=...                    # ou ibm-quantum-login
  export IBM_QUANTUM_BACKEND=ibm_fez          # opcional

  # Só transpile (não gasta QPU — planeje a corrida)
  python scripts/run_hardware_benchmarks.py --preset week1_iris --dry-run

  # Executar no hardware (4 jobs ≈ poucos minutos de QPU + fila)
  python scripts/run_hardware_benchmarks.py --preset week1_iris --execute

  # Salvar JSON
  python scripts/run_hardware_benchmarks.py --preset week1_iris --execute --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.datasets import list_benchmark_slugs, load_benchmark_dataset  # noqa: E402
from benchmarks.hardware_runner import (  # noqa: E402
    MONTHLY_PRESETS,
    format_hardware_benchmark_report,
    run_hardware_benchmark,
)
from llama_qiskit_agents.quantum.encodings import EncodingType  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark encoding no IBM Quantum")
    p.add_argument(
        "--preset",
        choices=list(MONTHLY_PRESETS.keys()),
        help="Roteiro mensal enxuto (1 preset ≈ 1 semana / 1 corrida)",
    )
    p.add_argument("--dataset", default="", help=f"Slug: {', '.join(list_benchmark_slugs())}")
    p.add_argument("--backend", default="", help="Backend IBM (default: IBM_QUANTUM_BACKEND ou ibm_fez)")
    p.add_argument("--encoding", default="angle", help="Encoding shallow: angle, dense_angle, basis")
    p.add_argument("--shots", type=int, default=512)
    p.add_argument("--max-jobs", type=int, default=4)
    p.add_argument("--execute", action="store_true", help="Submeter jobs (consome cota QPU)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias de não --execute; mostra plano e transpile se token disponível",
    )
    p.add_argument("--json", dest="json_out", default="", help="Salvar resultados JSON")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    execute = args.execute and not args.dry_run

    max_samples = 12
    if args.preset:
        cfg = MONTHLY_PRESETS[args.preset]
        slug = cfg["dataset"]
        encoding = cfg["encoding"]
        shots = cfg.get("shots", args.shots)
        max_jobs = cfg.get("max_jobs", args.max_jobs)
        max_samples = int(cfg.get("max_samples", max_samples))
        print(f"Preset: {args.preset}")
        print(f"  {cfg['description']}")
    else:
        if not args.dataset:
            print("Informe --preset ou --dataset", file=sys.stderr)
            return 1
        slug = args.dataset
        encoding = EncodingType(args.encoding)
        shots = args.shots
        max_jobs = args.max_jobs

    ds = load_benchmark_dataset(slug)
    backend = args.backend or None

    print()
    print(f"Dataset: {slug}  encoding: {encoding.value if isinstance(encoding, EncodingType) else encoding}")
    print(f"Modo: {'EXECUTE (QPU)' if execute else 'DRY-RUN / transpile-only'}")
    print(f"max_jobs={max_jobs}, shots={shots}, max_samples={max_samples}")
    print()

    enc = encoding if isinstance(encoding, EncodingType) else EncodingType(str(encoding))
    result = run_hardware_benchmark(
        ds,
        backend_name=backend,
        encoding=enc,
        shots=shots,
        execute=execute,
        max_jobs=max_jobs,
        max_samples=max_samples,
    )

    print(format_hardware_benchmark_report(result))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        print(f"JSON: {args.json_out}")

    if not execute:
        print(
            "Próximo passo: quando tiver janela de QPU (~10 min/mês), rode com --execute.\n"
            "Sugestão: 1 preset por semana → acumula comparativos sem estourar cota."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
