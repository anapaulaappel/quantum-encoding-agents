#!/usr/bin/env python3
"""
Benchmark suite — encoding, KTA e otimização de mapeamento de features.

Datasets públicos (sklearn): Iris, Breast Cancer, Wine + sintético order-sensitive.

Uso:
  pip install -e ".[benchmark]"
  python scripts/run_benchmarks.py
  python scripts/run_benchmarks.py --quick          # só iris + sintético, sem compare 7x
  python scripts/run_benchmarks.py --dataset iris_binary
  python scripts/run_benchmarks.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.datasets import (  # noqa: E402
    list_benchmark_slugs,
    load_all_benchmark_datasets,
    load_benchmark_dataset,
)
from benchmarks.runner import (  # noqa: E402
    format_result_detail,
    format_results_table,
    run_benchmark_suite,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark KTA / optimize_features")
    p.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help=f"Slug(s): {', '.join(list_benchmark_slugs())}. Repetível.",
    )
    p.add_argument("--quick", action="store_true", help="iris_binary + synthetic_order; sem compare 7 encodings")
    p.add_argument("--max-kta-samples", type=int, default=25)
    p.add_argument("--shots", type=int, default=512)
    p.add_argument("--json", dest="json_out", help="Salvar resultados em JSON")
    p.add_argument("--no-compare", action="store_true", help="Pular compare_embeddings (mais rápido)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.quick:
        slugs = ["iris_binary", "synthetic_order"]
        run_compare = False
    else:
        slugs = args.datasets or list_benchmark_slugs()
        run_compare = not args.no_compare

    datasets = [load_benchmark_dataset(s) for s in slugs]

    print("Quantum Encoding Agents — benchmark suite")
    print(f"Datasets: {', '.join(slugs)}")
    print(f"max_kta_samples={args.max_kta_samples}, shots={args.shots}, compare={run_compare}")
    print()

    t0 = time.perf_counter()
    results = run_benchmark_suite(
        datasets,
        max_kta_samples=args.max_kta_samples,
        shots=args.shots,
        run_compare=run_compare,
    )
    elapsed = time.perf_counter() - t0

    print(format_results_table(results))
    print()
    for r in results:
        print(format_result_detail(r))
        print()

    print(f"Concluído em {elapsed:.1f}s")

    if args.json_out:
        payload = [asdict(r) for r in results]
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
