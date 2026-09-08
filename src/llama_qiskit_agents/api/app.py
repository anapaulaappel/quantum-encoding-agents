"""
API HTTP (FastAPI) — microserviço quântico + surface de tools para agentes.

Pipeline determinístico (/v1/recommend/explain, /v1/compare, …) e camada agentica
(/v1/tools, /v1/agent/chat) compartilham ``dispatch_tool`` em tool_registry.
"""

import os
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from llama_qiskit_agents.agents.harness import build_backend, run_agent_turn
from llama_qiskit_agents.agents.tool_registry import (
    dispatch_tool,
    get_tool_definitions,
    list_tool_names,
    openclaw_skill_markdown,
)
from llama_qiskit_agents.quantum.data_analysis import (
    infer_data_profile,
    load_csv_from_string_with_labels,
    feature_names_from_csv_text,
    recommend_encoding,
)
from llama_qiskit_agents.quantum.fractal import apply_column_selection_row
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
from llama_qiskit_agents.quantum.visualization import render_bloch_sphere, bloch_caption as make_bloch_caption
from llama_qiskit_agents.quantum.kernel import compute_kernel
from llama_qiskit_agents.api.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    CompareRequest,
    CircuitRequest,
    DataInput,
    ExplainRequest,
    ExplainResponse,
    HardwareProfileInput,
    HealthResponse,
    KernelRequest,
    KernelResponse,
    ProfileResponse,
    RecommendResponse,
    SimulateRequest,
    ToolDispatchRequest,
    ToolDispatchResponse,
    ToolsListResponse,
    profile_to_response,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Llama Qiskit Agents API",
    description=(
        "Microserviço quântico (encoding, kernel, KTA) e surface de tools para agentes "
        "(OpenClaw, Ollama, Llama Stack, HTTP)."
    ),
    version="0.3.0",
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


def _resolve_input(body: DataInput) -> str | list[float] | list[list[float]]:
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
        "tools": "/v1/tools",
        "tool_dispatch": "/v1/tools/dispatch",
        "agent_chat": "/v1/agent/chat",
    }


@app.get("/v1/tools", response_model=ToolsListResponse)
def list_tools(
    fmt: Literal["openai", "openclaw"] = Query(default="openai"),
) -> ToolsListResponse:
    """Schemas de ferramentas para function-calling (OpenAI, OpenClaw, MCP-like HTTP)."""
    tools = get_tool_definitions(fmt)
    return ToolsListResponse(format=fmt, count=len(tools), tools=tools)


@app.get("/v1/tools/openclaw-skill.md", response_class=PlainTextResponse)
def openclaw_skill_doc(
    api_base: str = Query(default="http://llama-qiskit-agents:8080"),
) -> str:
    return openclaw_skill_markdown(api_base)


@app.post("/v1/tools/dispatch", response_model=ToolDispatchResponse)
def tools_dispatch(body: ToolDispatchRequest) -> ToolDispatchResponse:
    """Executa uma ferramenta quântica — ponto de integração para OpenClaw skills e outros agentes."""
    if body.name not in list_tool_names():
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{body.name}' não encontrada. Use GET /v1/tools.",
        )
    result = dispatch_tool(body.name, body.arguments)
    return ToolDispatchResponse(name=body.name, result=result)


@app.post("/v1/agent/chat", response_model=AgentChatResponse)
def agent_chat(body: AgentChatRequest) -> AgentChatResponse:
    """
    Turno agentico server-side: LLM + tools locais (Ollama/vLLM/Llama Stack via env).

    Variáveis: AGENT_BACKEND, AGENT_MODEL, OPENAI_API_BASE, OLLAMA_HOST, OPENAI_API_KEY.
    Com ``csv_content``, o prompt inclui instrução para usar ``compare_csv_embeddings``.
    """
    persona = body.persona if body.persona in ("default", "expert", "mentor") else "default"
    backend_kind = body.backend or os.environ.get("AGENT_BACKEND", "ollama")
    backend = build_backend(backend_kind)  # type: ignore[arg-type]

    user_message = body.message.strip()
    if body.csv_content and body.csv_content.strip():
        label_hint = (
            f" label_column={body.label_column!r}."
            if body.label_column
            else " (auto-detect label column if present)."
        )
        user_message += (
            "\n\n[CSV anexado pelo usuário — chame compare_csv_embeddings com csv_content "
            f"e problem_description derivada da pergunta.{label_hint}]\n"
            f"--- csv_content ---\n{body.csv_content.strip()[:120000]}\n--- fim csv ---"
        )

    turn = run_agent_turn(
        user_message,
        history=body.history,
        persona=persona,  # type: ignore[arg-type]
        backend=backend,
        max_tool_rounds=body.max_tool_rounds,
    )
    return AgentChatResponse(
        reply=turn.reply,
        persona=persona,
        tool_calls=turn.tool_calls,
        backend=backend_kind,
    )


@app.post("/v1/analyze", response_model=ProfileResponse)
def analyze(body: DataInput) -> ProfileResponse:
    raw = _resolve_input(body)
    profile = infer_data_profile(raw, apply_fractal_budget=body.apply_fractal_budget)
    return profile_to_response(profile)


@app.post("/v1/recommend", response_model=RecommendResponse)
def recommend(body: DataInput) -> RecommendResponse:
    raw = _resolve_input(body)
    profile = infer_data_profile(raw, apply_fractal_budget=body.apply_fractal_budget)
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
    profile = infer_data_profile(raw, apply_fractal_budget=body.apply_fractal_budget)
    enc, reason, ctx = recommend_encoding(
        profile,
        task=body.task,
        algorithm=body.algorithm,
        problem_description=body.problem_description,
        hardware_profile=hw,
    )

    # Simular o circuito recomendado para obter métricas reais (depth, qubits)
    # Se include_bloch=True, captura também o statevector para a esfera de Bloch
    sim_result = None
    data_arr = None
    if isinstance(raw, list) and len(raw) > 0:
        arr = np.asarray(raw, dtype=float)
        cols = profile.selected_columns if profile.fractal_selection_applied else None
        if arr.ndim == 2:
            data_arr = apply_column_selection_row(arr[0], cols).tolist()
        else:
            data_arr = apply_column_selection_row(arr, cols).tolist()
    if data_arr and len(data_arr) > 0:
        try:
            qc = build_encoding_circuit(enc, data_arr, n_qubits=body.n_qubits)
            sim_result = simulate_encoding_circuit(
                qc, enc, shots=body.shots,
                save_statevector=body.include_bloch,
            )
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

    # Gerar esfera de Bloch se solicitado e statevector disponível
    bloch_b64 = None
    bloch_cap = None
    if body.include_bloch and sim_result and sim_result.statevector is not None:
        bloch_b64 = render_bloch_sphere(
            sim_result.statevector, enc.value, lang=lang
        )
        if bloch_b64:
            bloch_cap = make_bloch_caption(sim_result.num_qubits, enc.value, lang=lang)

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
        bloch_sphere_b64=bloch_b64,
        bloch_caption=bloch_cap,
    )


@app.post("/v1/compare", response_class=PlainTextResponse)
def compare(body: CompareRequest) -> str:
    raw = _resolve_input(body)
    return dispatch_tool(
        "compare_embeddings_report",
        {
            "data": raw,
            "n_qubits": body.n_qubits,
            "shots": body.shots,
            "task": body.task,
            "algorithm": body.algorithm,
            "problem_description": body.problem_description,
            "apply_fractal_budget": body.apply_fractal_budget,
        },
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
    label_column: Annotated[str | None, Form()] = None,
    optimize_features: Annotated[bool, Form()] = False,
    optimization_encoding: Annotated[str | None, Form()] = None,
    apply_fractal_budget: Annotated[bool, Form()] = True,
) -> str:
    """
    Multipart: `file` + opcionalmente `problem_description` (texto livre: problema, tarefa, algoritmo).
    Coluna de label: auto-detect (label/class/target/y ou última coluna categórica) ou `label_column`.
    Com labels, o ranking usa KTA por encoding.
    `optimize_features=true` (requer labels): busca ordem/seleção/peso de colunas por KTA [EncOpt25].
    """
    text = (await file.read()).decode("utf-8-sig")
    try:
        arr, labels = load_csv_from_string_with_labels(
            text,
            label_column=(label_column or "").strip() or None,
        )
        col_names = feature_names_from_csv_text(
            text,
            label_column=(label_column or "").strip() or None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    n_q = _optional_int_form(n_qubits)
    opt_enc: EncodingType | None = None
    if optimization_encoding and optimization_encoding.strip():
        try:
            opt_enc = EncodingType(optimization_encoding.strip())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    cr = compare_embeddings(
        arr,
        n_qubits=n_q,
        shots=max(1, shots),
        problem_description=(problem_description or "").strip() or None,
        labels=labels,
        optimize_features=optimize_features,
        optimization_encoding=opt_enc,
        feature_column_names=col_names,
        apply_fractal_budget=apply_fractal_budget,
    )
    return format_comparison_report(cr)


@app.get("/v1/tradeoffs", response_class=PlainTextResponse)
def tradeoffs() -> str:
    return dispatch_tool("explain_tradeoffs", {})


@app.get("/v1/scenarios-guide", response_class=PlainTextResponse)
def scenarios_guide() -> str:
    return dispatch_tool("scenarios_guide", {})


@app.post("/v1/kernel", response_model=KernelResponse)
def kernel(body: KernelRequest) -> KernelResponse:
    """
    Calcula a matriz de kernel quântico K_{ij} = |⟨φ(xᵢ)|φ(xⱼ)⟩|² para um dataset.

    - Mínimo 2 amostras; recomendado ≤ 50 (custo O(N²)).
    - Retorna a matriz N×N, estatísticas (separability_hint, KTA se labels fornecidos)
      e um heatmap PNG base64 (colormap viridis).
    - Encoding padrão: custom_feature_map (melhor para kernels quânticos / QSVM).
    """
    if len(body.data) < 2:
        raise HTTPException(status_code=400, detail="Mínimo de 2 amostras para calcular o kernel.")
    if len(body.data) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo de 100 amostras para simulação local. Recebido: {len(body.data)}.",
        )

    try:
        enc = EncodingType(body.encoding_name)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Encoding desconhecido: '{body.encoding_name}'. "
                   f"Use: {', '.join(e.value for e in EncodingType)}",
        )

    # Detectar idioma a partir de qualquer texto no body
    lang = body.lang or "pt"

    try:
        result = compute_kernel(
            data=body.data,
            encoding_type=enc,
            labels=body.labels,
            n_qubits=body.n_qubits,
            lang=lang,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular kernel: {exc}") from exc

    from llama_qiskit_agents.quantum.kernel import kernel_caption
    cap = kernel_caption(result.n_samples, enc.value, result.stats, lang=lang)

    return KernelResponse(
        encoding_name=enc.value,
        n_samples=result.n_samples,
        n_features=result.n_features,
        kernel_matrix=result.kernel_matrix.tolist(),
        stats=result.stats,
        heatmap_b64=result.heatmap_b64,
        caption=cap,
    )


@app.post("/v1/circuit", response_class=PlainTextResponse)
def circuit(body: CircuitRequest) -> str:
    return dispatch_tool(
        "generate_qiskit_circuit",
        {
            "encoding_name": body.encoding_name,
            "data": body.data,
            "n_qubits": body.n_qubits,
        },
    )


@app.post("/v1/simulate", response_class=PlainTextResponse)
def simulate(body: SimulateRequest) -> str:
    return dispatch_tool(
        "simulate_circuit",
        {
            "encoding_name": body.encoding_name,
            "data": body.data,
            "n_qubits": body.n_qubits,
            "shots": body.shots,
        },
    )


@app.get("/v1/analyze/text", response_class=PlainTextResponse)
def analyze_text_legacy(q: str) -> str:
    """Compatível com testes rápidos: GET ?q=descrição"""
    return dispatch_tool("analyze_data", {"dataset_or_description": q})


@app.get("/chat", include_in_schema=False)
def chat_page() -> FileResponse:
    html = STATIC_DIR / "chat.html"
    if not html.is_file():
        raise HTTPException(status_code=404, detail="chat.html não encontrado")
    return FileResponse(html, media_type="text/html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
