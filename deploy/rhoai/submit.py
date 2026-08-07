"""
Submete o pipeline quantum-encoding ao RHOAI Data Science Pipelines.

Uso:
  python submit.py                              # parâmetros padrão
  python submit.py --task kernel --alg QSVM    # contexto QML
  python submit.py --csv dados.csv --labels "0,0,1,1"
  python submit.py --hw-error 5e-3 --connectivity heavy-hex

Pré-requisitos:
  pip install kfp>=2.0 kfp-kubernetes
  oc login --server=<cluster>
  # Obter o endpoint do DSP:
  oc get route -n rhoai-dsp data-science-pipelines-api -o jsonpath='{.spec.host}'
"""

import argparse
import os
import sys


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Submete o pipeline de quantum encoding ao RHOAI DSP."
    )
    # Dados
    p.add_argument("--csv", help="Caminho para CSV local (carregado como string)")
    p.add_argument("--csv-content", default="0.31,0.72,0.55\n0.18,0.90,0.44\n0.62,0.33,0.77",
                   help="CSV inline (padrão: 3 amostras de exemplo)")
    p.add_argument("--labels", default="", help="Rótulos de classe separados por vírgula (ex: '0,0,1,1')")

    # Contexto QML
    p.add_argument("--task", default="", help="Tarefa QML: classification|kernel|variational|...")
    p.add_argument("--alg", "--algorithm", default="", dest="algorithm", help="Algoritmo: QSVM, VQC, ...")
    p.add_argument("--problem", default="", help="Descrição livre do problema")

    # Hardware
    p.add_argument("--hw-error", type=float, default=0.0,
                   help="gate_error_rate do hardware alvo (ex: 5e-3 para IBM Eagle)")
    p.add_argument("--connectivity", default="all-to-all",
                   choices=["all-to-all", "heavy-hex", "linear", "grid"],
                   help="Topologia de conectividade do hardware")
    p.add_argument("--max-depth", type=int, default=0, help="Budget máximo de profundidade")

    # Infraestrutura
    p.add_argument("--api-url",
                   default="http://llama-qiskit-agents.openclaw.svc.cluster.local:8080",
                   help="URL da API llama-qiskit-agents")
    p.add_argument("--dsp-endpoint", default="",
                   help="Endpoint do RHOAI Data Science Pipelines (https://...)")
    p.add_argument("--mlflow-uri", default="", help="MLflow tracking URI")
    p.add_argument("--experiment", default="quantum-encoding", help="Nome do experimento MLflow")
    p.add_argument("--run-name", default="", help="Nome do run MLflow")
    p.add_argument("--lang", default="pt", choices=["pt", "en"], help="Idioma das respostas")
    p.add_argument("--shots", type=int, default=512, help="Shots para simulação")
    p.add_argument("--max-kernel-samples", type=int, default=20,
                   help="Máximo de amostras para cálculo do kernel (custo O(N²))")

    # Compilação
    p.add_argument("--compile-only", action="store_true",
                   help="Apenas compila o pipeline YAML, não submete")
    p.add_argument("--pipeline-yaml", default="quantum_encoding_pipeline.yaml",
                   help="Caminho do YAML compilado")
    return p


def main():
    args = get_parser().parse_args()

    # Carrega CSV de arquivo se informado
    csv_content = args.csv_content
    if args.csv:
        try:
            with open(args.csv) as f:
                csv_content = f.read()
            print(f"CSV carregado: {args.csv} ({len(csv_content.splitlines())} linhas)")
        except FileNotFoundError:
            print(f"ERRO: arquivo '{args.csv}' não encontrado.", file=sys.stderr)
            sys.exit(1)

    # Adiciona deploy/rhoai ao path para importar os componentes
    rhoai_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, rhoai_dir)

    from pipeline import quantum_encoding_pipeline
    from kfp import compiler

    # Compilar
    print(f"Compilando pipeline → {args.pipeline_yaml}")
    compiler.Compiler().compile(
        pipeline_func=quantum_encoding_pipeline,
        package_path=args.pipeline_yaml,
    )
    print("Pipeline compilado com sucesso.")

    if args.compile_only:
        print("Modo --compile-only: pipeline não submetido.")
        return

    # Submeter
    if not args.dsp_endpoint:
        print("\nSem --dsp-endpoint: pipeline compilado mas não submetido.")
        print("Para submeter via RHOAI UI:")
        print(f"  1. Acesse Data Science Pipelines no RHOAI")
        print(f"  2. Importe o arquivo: {args.pipeline_yaml}")
        print(f"  3. Configure os parâmetros e execute.")
        print("\nPara submeter via CLI:")
        print(f"  DSP_HOST=$(oc get route -n rhoai-dsp data-science-pipelines-api -o jsonpath='{{.spec.host}}')")
        print(f"  python submit.py --dsp-endpoint https://$DSP_HOST [outros parâmetros]")
        return

    try:
        import kfp
        client = kfp.Client(host=args.dsp_endpoint)

        run = client.create_run_from_pipeline_func(
            quantum_encoding_pipeline,
            arguments={
                "csv_content":       csv_content,
                "labels_csv":        args.labels,
                "task":              args.task,
                "algorithm":         args.algorithm,
                "problem_description": args.problem,
                "gate_error_rate":   args.hw_error,
                "connectivity":      args.connectivity,
                "max_depth_budget":  args.max_depth,
                "api_url":           args.api_url,
                "mlflow_tracking_uri": args.mlflow_uri,
                "experiment_name":   args.experiment,
                "run_name":          args.run_name,
                "lang":              args.lang,
                "shots":             args.shots,
                "max_kernel_samples": args.max_kernel_samples,
            },
            experiment_name=args.experiment,
            run_name=args.run_name or "quantum-encoding-run",
        )
        print(f"\nPipeline submetido com sucesso!")
        print(f"Run ID: {run.run_id}")
        print(f"Acompanhe em: {args.dsp_endpoint}/#/runs/details/{run.run_id}")

    except Exception as exc:
        print(f"ERRO ao submeter pipeline: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
