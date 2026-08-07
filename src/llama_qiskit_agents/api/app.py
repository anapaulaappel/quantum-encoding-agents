"""
API HTTP (FastAPI) para o agente de encoding quântico.
Roda localmente ou em container no OpenShift.
"""

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from llama_qiskit_agents.agents.encoding_agent import (
    analyze_data,
    compare_embeddings_report,
    explain_tradeoffs,
    generate_qiskit_circuit,
    simulate_circuit,
)
from llama_qiskit_agents.quantum.data_analysis import (
    infer_data_profile,
    load_csv_from_string,
    recommend_encoding,
)
from llama_qiskit_agents.quantum.problem_context import scenario_guide_when_unspecified
from llama_qiskit_agents.quantum.simulate import (
    compare_embeddings,
    format_comparison_report,
    simulate_encoding_circuit,
)
from llama_qiskit_agents.quantum.encodings import build_encoding_circuit, EncodingType
from llama_qiskit_agents.quantum.explanation import (
    detect_language,
    build_natural_explanation,
    generate_qiskit_code,
)
from llama_qiskit_agents.quantum.hardware_profile import HardwareProfile
from llama_qiskit_agents.api.schemas import (
    CompareRequest,
    CircuitRequest,
    DataInput,
    ExplainRequest,
    ExplainResponse,
    HardwareProfileInput,
    HealthResponse,
    ProfileResponse,
    RecommendResponse,
    SimulateRequest,
    profile_to_response,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Llama Qiskit Agents API",
    description="Análise de dados, recomendação de encoding quântico e simulação Qiskit.",
    version="0.2.0",
)

_cors = os.environ.get("CORS_ORIGINS", "*").strip()
_origins = [o.strip() for o in _cors.split(",") if o.strip()]
_wildcard = _origins == ["*"] or _origins == []
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _wildcard else _origins,
    allow_credentials=not _wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_input(body: DataInput) -> str | list[float]:
    if body.data is not None and len(body.data) > 0:
        return body.data
    if body.description:
        return body.description
    raise HTTPException(status_code=400, detail="Informe description ou data.")


def _to_hardware_profile(hw_input: HardwareProfileInput | None) -> HardwareProfile | None:
    """Converte o schema Pydantic para o dataclass interno."""
    if hw_input is None:
        return None
    return HardwareProfile(
        gate_error_rate=hw_input.gate_error_rate,
        max_depth_budget=hw_input.max_depth_budget,
        max_qubits=hw_input.max_qubits,
        connectivity=hw_input.connectivity,
        backend_name=hw_input.backend_name,
    )


@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="llama-qiskit-agents")


@app.get("/")
def root() -> dict:
    return {
        "service": "llama-qiskit-agents",
        "docs": "/docs",
        "chat_ui": "/chat",
        "health": "/health",
    }


@app.post("/v1/analyze", response_model=ProfileResponse)
def analyze(body: DataInput) -> ProfileResponse:
    raw = _resolve_input(body)
    profile = infer_data_profile(raw)
    return profile_to_response(profile)


@app.post("/v1/recommend", response_model=RecommendResponse)
def recommend(body: DataInput) -> RecommendResponse:
    raw = _resolve_input(body)
    profile = infer_data_profile(raw)
    hw = _to_hardware_profile(body.hardware_profile)
    enc, reason, ctx = recommend_encoding(
        profile,
        task=body.task,
        algorithm=body.algorithm,
        problem_description=body.problem_description,
        hardware_profile=hw,
    )
    return RecommendResponse(
        profile=profile_to_response(profile),
        recommended_encoding=enc.value,
        reason=reason,
        task_interpreted=ctx.task.value,
        algorithm=ctx.algorithm,
        scenario_guide_included=not ctx.has_explicit_info(),
    )


@app.post("/v1/recommend/explain", response_model=ExplainResponse)
def recommend_explain(body: ExplainRequest) -> ExplainResponse:
    """
    Endpoint principal para o chat: recebe descrição livre e/ou dados numéricos,
    detecta o idioma automaticamente, e retorna:
      - Recomendação de encoding
      - Justificativa em linguagem natural (citando DataProfile + métricas do circuito)
      - Código Python/Qiskit completo e copiável
    """
    # Resolver input
    raw = _resolve_input(body)

    # Detectar idioma a partir de todos os campos de texto disponíveis
    all_text = " ".join(filter(None, [
        body.description or "",
        body.problem_description or "",
        body.task or "",
        body.algorithm or "",
    ]))
    lang = body.lang or detect_language(all_text)

    # Perfil + recomendação (com hardware_profile opcional)
    hw = _to_hardware_profile(body.hardware_profile)
    profile = infer_data_profile(raw)
    enc, reason, ctx = recommend_encoding(
        profile,
        task=body.task,
        algorithm=body.algorithm,
        problem_description=body.problem_description,
        hardware_profile=hw,
    )

    # Simular o circuito recomendado para obter métricas reais (depth, qubits)
    sim_result = None
    data_arr = raw if isinstance(raw, list) else None
    if data_arr and len(data_arr) > 0:
        try:
            qc = build_encoding_circuit(enc, data_arr, n_qubits=body.n_qubits)
            sim_result = simulate_encoding_circuit(qc, enc, shots=body.shots)
        except Exception:
            pass

    # Gerar explicação narrativa
    explanation = build_natural_explanation(
        profile=profile,
        recommended=enc,
        reason=reason,
        sim_result=sim_result,
        context=ctx,
        lang=lang,
    )

    # Gerar código Qiskit
    sample = data_arr or [0.1, 0.2, 0.3, 0.4]
    qiskit_code = generate_qiskit_code(
        encoding=enc,
        data_sample=sample,
        profile=profile,
        n_qubits=body.n_qubits or (sim_result.num_qubits if sim_result else None),
        lang=lang,
    )

    return ExplainResponse(
        lang=lang,
        recommended_encoding=enc.value,
        explanation=explanation,
        qiskit_code=qiskit_code,
        profile=profile_to_response(profile),
        task_interpreted=ctx.task.value,
        circuit_depth=sim_result.depth if sim_result else None,
        circuit_qubits=sim_result.num_qubits if sim_result else None,
        hardware_constraints_applied=hw is not None,
    )


@app.post("/v1/compare", response_class=PlainTextResponse)
def compare(body: CompareRequest) -> str:
    raw = _resolve_input(body)
    return compare_embeddings_report(
        raw,
        n_qubits=body.n_qubits,
        shots=body.shots,
        task=body.task,
        algorithm=body.algorithm,
        problem_description=body.problem_description,
    )


def _optional_int_form(v: str | None) -> int | None:
    if v is None or (isinstance(v, str) and not str(v).strip()):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@app.post("/v1/compare/csv", response_class=PlainTextResponse)
async def compare_csv(
    file: Annotated[UploadFile, File(description="Arquivo CSV com colunas numéricas")],
    problem_description: Annotated[str | None, Form()] = None,
    shots: Annotated[int, Form()] = 1024,
    n_qubits: Annotated[str | None, Form()] = None,
) -> str:
    """
    Multipart: `file` + opcionalmente `problem_description` (texto livre: problema, tarefa, algoritmo).
    Palavras-chave no texto são inferidas (classificação, QSVM, etc.).
    """
    text = (await file.read()).decode("utf-8-sig")
    try:
        arr = load_csv_from_string(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    n_q = _optional_int_form(n_qubits)
    results, profile, recommended, reason, ctx = compare_embeddings(
        arr,
        n_qubits=n_q,
        shots=max(1, shots),
        problem_description=(problem_description or "").strip() or None,
    )
    return format_comparison_report(results, profile, recommended, reason, ctx)


@app.get("/v1/tradeoffs", response_class=PlainTextResponse)
def tradeoffs() -> str:
    return explain_tradeoffs()


@app.get("/v1/scenarios-guide", response_class=PlainTextResponse)
def scenarios_guide() -> str:
    """Guia: qual encoding tende a servir para classificação, kernel, clusterização, etc."""
    return scenario_guide_when_unspecified()


@app.post("/v1/circuit", response_class=PlainTextResponse)
def circuit(body: CircuitRequest) -> str:
    return generate_qiskit_circuit(
        body.encoding_name,
        body.data,
        n_qubits=body.n_qubits,
    )


@app.post("/v1/simulate", response_class=PlainTextResponse)
def simulate(body: SimulateRequest) -> str:
    return simulate_circuit(
        body.encoding_name,
        body.data,
        n_qubits=body.n_qubits,
        shots=body.shots,
    )


@app.get("/v1/analyze/text", response_class=PlainTextResponse)
def analyze_text_legacy(q: str) -> str:
    """Compatível com testes rápidos: GET ?q=descrição"""
    return analyze_data(q)


@app.get("/chat", include_in_schema=False)
def chat_page() -> FileResponse:
    html = STATIC_DIR / "chat.html"
    if not html.is_file():
        raise HTTPException(status_code=404, detail="chat.html não encontrado")
    return FileResponse(html, media_type="text/html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
