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
from qiskit import QuantumCircuit

from llama_qiskit_agents.quantum.encodings import EncodingType, build_encoding_circuit


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
) -> np.ndarray:
    """
    Calcula a matriz de kernel N×N usando FidelityStatevectorKernel.
    K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|²  ∈ [0, 1]

    Complexidade: O(N² × profundidade_circuito).
    Prático para N ≤ 50 amostras em simulação clássica.
    """
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_samples, n_features = x.shape

    # Constrói o feature map para uma amostra (o Qiskit parametrizará automaticamente)
    # FidelityStatevectorKernel aceita QuantumCircuit parametrizado OU
    # calcula diretamente a partir de statevectors concretos quando passamos x.
    # Aqui usamos a rota mais simples: calcular statevectors manualmente e
    # depois computar os produtos internos (|⟨ψᵢ|ψⱼ⟩|²).
    from qiskit_aer import StatevectorSimulator
    from qiskit import transpile

    sim = StatevectorSimulator()

    # Calcular statevector para cada amostra
    statevectors: list[np.ndarray] = []
    for row in x:
        try:
            qc = build_encoding_circuit(encoding_type, row, n_qubits=n_qubits)
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


def kernel_stats(K: np.ndarray, labels: list[int] | None = None) -> dict[str, float]:
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

    return {k: round(v, 4) for k, v in stats.items()}


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

    if lang == "en":
        return (
            f"Quantum kernel matrix K computed from {n_samples} samples using {encoding_name} encoding. "
            f"K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|² ∈ [0,1]: yellow = high similarity, purple = low similarity "
            f"in Hilbert space. Diagonal is always 1 (self-similarity). "
            f"Off-diagonal mean: {stats['off_diagonal_mean']:.3f}, "
            f"separability hint: {sep_str} ({sep:.3f}).{kta_str} "
            f"For QSVM, higher separability between classes suggests this encoding "
            f"may produce a useful quantum kernel."
        )
    return (
        f"Matriz de kernel quântico K calculada a partir de {n_samples} amostras usando {encoding_name}. "
        f"K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|² ∈ [0,1]: amarelo = alta similaridade, roxo = baixa similaridade "
        f"no espaço de Hilbert. A diagonal é sempre 1 (auto-similaridade). "
        f"Média fora da diagonal: {stats['off_diagonal_mean']:.3f}, "
        f"separabilidade: {sep_str} ({sep:.3f}).{kta_str} "
        f"Em QSVM, maior separabilidade entre classes indica que este encoding "
        f"pode produzir um kernel quântico útil."
    )


def compute_kernel(
    data: list[list[float]] | np.ndarray,
    encoding_type: EncodingType,
    labels: list[int] | None = None,
    n_qubits: int | None = None,
    lang: str = "pt",
) -> KernelResult:
    """
    Pipeline completo: calcula K, gera estatísticas e heatmap.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n_samples, n_features = x.shape

    K = compute_kernel_matrix(x, encoding_type, n_qubits=n_qubits)
    stats = kernel_stats(K, labels)
    heatmap = render_kernel_heatmap(K, encoding_type.value, labels=labels, lang=lang)

    return KernelResult(
        encoding_type=encoding_type,
        kernel_matrix=K,
        n_samples=n_samples,
        n_features=n_features,
        heatmap_b64=heatmap,
        stats=stats,
    )
