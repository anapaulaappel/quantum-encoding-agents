"""
Matriz de kernel quântico via FidelityStatevectorKernel.

O kernel quântico K(x, x') = |⟨φ(x)|φ(x')⟩|² mede a similaridade entre
duas amostras no espaço de Hilbert definido pelo encoding φ(·).

Por que é relevante:
  - Em QSVM, o kernel substitui o kernel clássico (RBF, polinomial) — a
    separabilidade do modelo depende diretamente da expressividade do encoding.
  - Visualizar K como heatmap antes de treinar qualquer classificador permite
    avaliar se o encoding consegue separar as classes no espaço quântico.
  - FidelityStatevectorKernel (Qiskit ML 0.7+) é mais rápido que o kernel
    baseado em shots por usar simulação exata de statevector.

Referência: Havlíček et al., Nature 567, 209–212 (2019).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from qiskit import transpile
from qiskit_aer import StatevectorSimulator

from llama_qiskit_agents.quantum.encodings import EncodingType, build_encoding_circuit
from llama_qiskit_agents.quantum.preprocessing import FeatureBounds


# Limiares operacionais de Appel, arXiv:2609.00475 (calibrados em n≈32).
ALIVE_NEAR_FLOOR = 0.25
ALIVE_RATIO_FLOOR = 2.0
ALIVE_MEAN_OFFDIAG_FLOOR = 0.03
NEAR_FAR_PAIRS = 20
MAX_QUBITS_FOR_KERNEL_DIAGNOSTICS = 12


@dataclass
class KernelGeometry:
    """Geometria do kernel de fidelidade, independente de rótulo."""

    fid_near: float
    fid_far: float
    near_far_ratio: float
    mean_offdiag: float
    kernel_alive: bool


@dataclass
class KernelDiagnostics:
    """KTA (se houver labels) + geometria alive na mesma matriz K."""

    geometry: KernelGeometry
    kta: float | None = None


@dataclass
class KernelResult:
    """Resultado do cálculo da matriz de kernel quântico."""

    encoding_type: EncodingType
    kernel_matrix: np.ndarray       # N×N, valores em [0, 1]
    n_samples: int
    n_features: int
    heatmap_b64: str | None         # PNG base64 do heatmap
    stats: dict[str, float]         # diagonal_mean, off_diagonal_mean, separability_hint


def compute_kernel_matrix(
    data: list[list[float]] | np.ndarray,
    encoding_type: EncodingType,
    n_qubits: int | None = None,
    *,
    feature_bounds: FeatureBounds | None = None,
) -> np.ndarray:
    """
    Calcula a matriz de kernel N×N usando FidelityStatevectorKernel.
    K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|²  ∈ [0, 1]

    Complexidade: O(N² × profundidade_circuito).
    Prático para N ≤ 50 amostras em simulação clássica.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_samples, n_features = x.shape

    # Constrói o feature map para uma amostra (o Qiskit parametrizará automaticamente)
    # FidelityStatevectorKernel aceita QuantumCircuit parametrizado OU
    # calcula diretamente a partir de statevectors concretos quando passamos x.
    # Aqui usamos a rota mais simples: calcular statevectors manualmente e
    # depois computar os produtos internos (|⟨ψᵢ|ψⱼ⟩|²).
    sim = StatevectorSimulator()

    # Calcular statevector para cada amostra
    statevectors: list[np.ndarray] = []
    for row in x:
        try:
            qc = build_encoding_circuit(
                encoding_type,
                row,
                n_qubits=n_qubits,
                feature_bounds=feature_bounds,
            )
            qc_clean = qc.copy()
            qc_clean.remove_final_measurements(inplace=True)
            transpiled = transpile(qc_clean, sim)
            sv = np.array(sim.run(transpiled).result().get_statevector(), dtype=complex)
            statevectors.append(sv)
        except Exception:
            # Fallback: vetor nulo (será tratado abaixo)
            statevectors.append(np.zeros(2 ** (n_qubits or n_features), dtype=complex))

    # K[i,j] = |⟨ψᵢ|ψⱼ⟩|²
    K = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for j in range(n_samples):
            inner = np.dot(statevectors[i].conj(), statevectors[j])
            K[i, j] = float(abs(inner) ** 2)

    return K


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    x = np.asarray(X, dtype=np.float64)
    delta = x[:, None, :] - x[None, :, :]
    dist2 = np.sum(delta * delta, axis=-1)
    return np.sqrt(np.maximum(dist2, 0.0))


def near_far_fidelity(
    X: np.ndarray,
    kernel: np.ndarray,
    n_pairs: int = NEAR_FAR_PAIRS,
) -> tuple[float, float]:
    """Fidelidade média de pares próximos vs. distantes em X (euclidiano)."""
    x = np.asarray(X, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    n = x.shape[0]
    if n < 4 or k.shape != (n, n):
        return 0.0, 0.0
    dist = _pairwise_euclidean(x)
    iu = np.triu_indices(n, k=1)
    order = np.argsort(dist[iu])
    take = max(1, min(int(n_pairs), order.size // 2))
    near_idx = order[:take]
    far_idx = order[-take:]
    k_vals = k[iu]
    return float(k_vals[near_idx].mean()), float(k_vals[far_idx].mean())


def near_far_ratio(near: float, far: float) -> float:
    if far > 1e-12:
        return float(near / far)
    return float("inf") if near > 0 else 0.0


def kernel_is_alive(
    near: float,
    far: float,
    mean_offdiag: float,
    *,
    near_floor: float = ALIVE_NEAR_FLOOR,
    ratio_floor: float = ALIVE_RATIO_FLOOR,
    mean_floor: float = ALIVE_MEAN_OFFDIAG_FLOOR,
) -> bool:
    """Kernel ainda distingue geometria clássica (Appel, arXiv:2609.00475)."""
    ratio = near_far_ratio(near, far)
    return bool(near >= near_floor and ratio >= ratio_floor and mean_offdiag >= mean_floor)


def score_kernel_geometry(
    X: np.ndarray,
    kernel: np.ndarray,
    *,
    n_pairs: int = NEAR_FAR_PAIRS,
) -> KernelGeometry:
    """Near/far, média off-diagonal e regra alive na mesma K."""
    k = np.asarray(kernel, dtype=np.float64)
    n = k.shape[0]
    if n < 2:
        mean_off = 0.0
    else:
        off = k[~np.eye(n, dtype=bool)]
        mean_off = float(off.mean()) if off.size else 0.0
    near, far = near_far_fidelity(X, k, n_pairs=n_pairs)
    ratio = near_far_ratio(near, far)
    finite_ratio = 99.0 if not np.isfinite(ratio) else float(ratio)
    return KernelGeometry(
        fid_near=round(near, 4),
        fid_far=round(far, 4),
        near_far_ratio=round(finite_ratio, 4),
        mean_offdiag=round(mean_off, 4),
        kernel_alive=kernel_is_alive(near, far, mean_off),
    )


def compute_kernel_diagnostics_by_encoding(
    data: np.ndarray,
    labels: list[int] | None = None,
    encoding_types: list[EncodingType] | None = None,
    n_qubits: int | None = None,
    *,
    max_samples: int = 25,
    feature_bounds: FeatureBounds | None = None,
) -> dict[EncodingType, KernelDiagnostics]:
    """
    Calcula K por encoding (subamostra se N > max_samples) e devolve
    geometria alive + KTA quando houver ≥2 classes.
    """
    if encoding_types is None:
        encoding_types = list(EncodingType)

    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = x.shape[0]
    if n < 2:
        return {}

    idx = np.arange(n)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
    x_sub = x[idx]
    labels_sub: list[int] | None = None
    if labels is not None and len(labels) == n and len(set(labels)) >= 2:
        labels_sub = [labels[int(i)] for i in idx]

    out: dict[EncodingType, KernelDiagnostics] = {}
    for enc in encoding_types:
        try:
            k = compute_kernel_matrix(
                x_sub,
                enc,
                n_qubits=n_qubits,
                feature_bounds=feature_bounds,
            )
            geo = score_kernel_geometry(x_sub, k)
            stats = kernel_stats(k, labels_sub)
            kta = stats.get("kta")
            out[enc] = KernelDiagnostics(geometry=geo, kta=kta)
        except Exception:
            continue
    return out


def compute_kta_by_encoding(
    data: np.ndarray,
    labels: list[int],
    encoding_types: list[EncodingType] | None = None,
    n_qubits: int | None = None,
    *,
    max_samples: int = 25,
    feature_bounds: FeatureBounds | None = None,
) -> dict[EncodingType, float]:
    """
    Calcula KTA para cada encoding (subamostra se N > max_samples).
    Retorna apenas encodings que completaram sem erro.
    """
    if len(labels) < 2 or len(set(labels)) < 2:
        return {}
    diagnostics = compute_kernel_diagnostics_by_encoding(
        data,
        labels=labels,
        encoding_types=encoding_types,
        n_qubits=n_qubits,
        max_samples=max_samples,
        feature_bounds=feature_bounds,
    )
    return {enc: d.kta for enc, d in diagnostics.items() if d.kta is not None}


def kernel_stats(
    K: np.ndarray,
    labels: list[int] | None = None,
    X: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Estatísticas descritivas da matriz de kernel.
    Se labels fornecidos, calcula KTA (Kernel-Target Alignment) estimado.
    """
    n = K.shape[0]
    diag = np.diag(K)
    off_diag = K[~np.eye(n, dtype=bool)]

    stats: dict[str, float] = {
        "diagonal_mean": float(np.mean(diag)),
        "off_diagonal_mean": float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0,
        "off_diagonal_std": float(np.std(off_diag)) if len(off_diag) > 0 else 0.0,
        "min": float(K.min()),
        "max": float(K.max()),
        # Separability hint: quanto maior a diferença diagonal - off_diagonal,
        # mais o encoding tende a distinguir amostras diferentes.
        "separability_hint": float(np.mean(diag) - np.mean(off_diag)) if len(off_diag) > 0 else 0.0,
    }

    # KTA estimado (sem labels reais — usa identidade como target ideal)
    if labels is not None and len(set(labels)) >= 2:
        y = np.array(labels, dtype=float)
        # Target kernel: T[i,j] = yᵢ·yⱼ (para classes ±1)
        y_pm = np.where(y == 0, -1.0, 1.0)  # converte 0→-1 para duas classes
        T = np.outer(y_pm, y_pm)
        # KTA = ⟨K, T⟩_F / (‖K‖_F · ‖T‖_F)
        kta = float(np.sum(K * T) / (np.linalg.norm(K, "fro") * np.linalg.norm(T, "fro") + 1e-12))
        stats["kta"] = round(kta, 4)

    if X is not None:
        geo = score_kernel_geometry(X, K)
        stats["fid_near"] = geo.fid_near
        stats["fid_far"] = geo.fid_far
        stats["near_far_ratio"] = geo.near_far_ratio
        stats["kernel_alive"] = 1.0 if geo.kernel_alive else 0.0

    return {k: round(v, 4) for k, v in stats.items()}


def format_kernel_alive_section(
    geometry_by_encoding: dict[EncodingType, KernelGeometry] | None,
) -> list[str]:
    """Bloco de relatório: ALIVE/DEAD por encoding (independente de labels)."""
    if not geometry_by_encoding:
        return []
    ordered = [enc for enc in EncodingType if enc in geometry_by_encoding]
    for enc in geometry_by_encoding:
        if enc not in ordered:
            ordered.append(enc)
    lines = [
        "=== Kernel alive (geometria de K, arXiv:2609.00475) ===",
        "  Regra operacional: near ≥ 0.25, near/far ≥ 2, média off-diagonal ≥ 0.03.",
        "  Independente de labels (não substitui KTA). Subamostra até 25 pontos;",
        "  a regra foi calibrada em n≈32 no preprint.",
        "  Encodings com mais de 12 qubits não entram (limite de statevector no compare).",
    ]
    for enc in ordered:
        geo = geometry_by_encoding[enc]
        status = "ALIVE" if geo.kernel_alive else "DEAD"
        lines.append(
            f"  {enc.value:<20} {status:<5}  near={geo.fid_near:.3f}  "
            f"far={geo.fid_far:.3f}  ratio={geo.near_far_ratio:.2f}  "
            f"meanK={geo.mean_offdiag:.3f}"
        )
    lines.append("")
    return lines


def render_kernel_heatmap(
    K: np.ndarray,
    encoding_name: str,
    labels: list[int] | None = None,
    lang: str = "pt",
) -> str | None:
    """
    Renderiza a matriz de kernel como heatmap PNG base64.
    Usa colormap 'viridis': valores próximos de 1 (amarelo) = amostras similares,
    próximos de 0 (roxo) = amostras distintas no espaço de Hilbert.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        n = K.shape[0]
        fig, ax = plt.subplots(figsize=(max(4, n * 0.5), max(3.5, n * 0.5)))

        im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="K(x,x') = |⟨φ(x)|φ(x')⟩|²")

        title = (
            f"Quantum Kernel Matrix — {encoding_name}"
            if lang == "en"
            else f"Matriz de Kernel Quântico — {encoding_name}"
        )
        ax.set_title(title, fontsize=11, pad=10)

        xlabel = "Sample index" if lang == "en" else "Índice da amostra"
        ylabel = "Sample index" if lang == "en" else "Índice da amostra"
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        # Colorir ticks por label (se fornecido)
        if labels is not None and n <= 20:
            unique_labels = sorted(set(labels))
            cmap_labels = plt.colormaps["tab10"].resampled(max(len(unique_labels), 2))
            color_map = {lbl: tuple(cmap_labels(i)[:3]) for i, lbl in enumerate(unique_labels)}
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(range(n), fontsize=7)
            ax.set_yticklabels(range(n), fontsize=7)
            # Colorir cada tick individualmente
            for tick, i in zip(ax.get_xticklabels(), range(n)):
                tick.set_color(color_map[labels[i]])
            for tick, i in zip(ax.get_yticklabels(), range(n)):
                tick.set_color(color_map[labels[i]])
            # Legenda de classes
            patches = [
                mpatches.Patch(color=color_map[lbl], label=f"Class {lbl}")
                for lbl in unique_labels
            ]
            ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.7)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception:
        return None


def kernel_caption(
    n_samples: int,
    encoding_name: str,
    stats: dict[str, float],
    lang: str = "pt",
) -> str:
    """Legenda interpretativa do heatmap de kernel."""
    sep = stats.get("separability_hint", 0.0)
    sep_str = (
        ("high" if sep > 0.3 else "moderate" if sep > 0.1 else "low")
        if lang == "en"
        else ("alta" if sep > 0.3 else "moderada" if sep > 0.1 else "baixa")
    )
    kta_str = ""
    if "kta" in stats:
        kta_str = (
            f" KTA (Kernel-Target Alignment) = {stats['kta']:.3f}."
            if lang == "en"
            else f" KTA (alinhamento kernel-target) = {stats['kta']:.3f}."
        )
    alive_str = ""
    if "kernel_alive" in stats:
        status = "ALIVE" if stats["kernel_alive"] >= 0.5 else "DEAD"
        near = stats.get("fid_near", 0.0)
        far = stats.get("fid_far", 0.0)
        ratio = stats.get("near_far_ratio", 0.0)
        if lang == "en":
            alive_str = (
                f" Kernel geometry (Appel, arXiv:2609.00475): {status} "
                f"(near={near:.3f}, far={far:.3f}, near/far={ratio:.2f})."
            )
        else:
            alive_str = (
                f" Geometria do kernel (Appel, arXiv:2609.00475): {status} "
                f"(near={near:.3f}, far={far:.3f}, near/far={ratio:.2f})."
            )

    if lang == "en":
        return (
            f"Quantum kernel matrix K computed from {n_samples} samples using {encoding_name} encoding. "
            f"K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|² ∈ [0,1]: yellow = high similarity, purple = low similarity "
            f"in Hilbert space. Diagonal is always 1 (self-similarity). "
            f"Off-diagonal mean: {stats['off_diagonal_mean']:.3f}, "
            f"separability hint: {sep_str} ({sep:.3f}).{kta_str}{alive_str} "
            f"For QSVM, higher separability between classes suggests this encoding "
            f"may produce a useful quantum kernel."
        )
    return (
        f"Matriz de kernel quântico K calculada a partir de {n_samples} amostras usando {encoding_name}. "
        f"K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|² ∈ [0,1]: amarelo = alta similaridade, roxo = baixa similaridade "
        f"no espaço de Hilbert. A diagonal é sempre 1 (auto-similaridade). "
        f"Média fora da diagonal: {stats['off_diagonal_mean']:.3f}, "
        f"separabilidade: {sep_str} ({sep:.3f}).{kta_str}{alive_str} "
        f"Em QSVM, maior separabilidade entre classes indica que este encoding "
        f"pode produzir um kernel quântico útil."
    )


def compute_kernel(
    data: list[list[float]] | np.ndarray,
    encoding_type: EncodingType,
    labels: list[int] | None = None,
    n_qubits: int | None = None,
    lang: str = "pt",
    feature_bounds: FeatureBounds | None = None,
) -> KernelResult:
    """
    Pipeline completo: calcula K, gera estatísticas e heatmap.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n_samples, n_features = x.shape
    bounds = feature_bounds
    if bounds is None and n_samples >= 2:
        bounds = FeatureBounds.fit(x)

    K = compute_kernel_matrix(
        x,
        encoding_type,
        n_qubits=n_qubits,
        feature_bounds=bounds,
    )
    stats = kernel_stats(K, labels, X=x)
    heatmap = render_kernel_heatmap(K, encoding_type.value, labels=labels, lang=lang)

    return KernelResult(
        encoding_type=encoding_type,
        kernel_matrix=K,
        n_samples=n_samples,
        n_features=n_features,
        heatmap_b64=heatmap,
        stats=stats,
    )
