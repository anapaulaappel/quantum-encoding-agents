"""Kernel-alive: geometria near/far independente de rótulo (Appel, arXiv:2609.00475)."""

from __future__ import annotations

import numpy as np

from llama_qiskit_agents.quantum.encodings import EncodingType
from llama_qiskit_agents.quantum.kernel import (
    kernel_is_alive,
    score_kernel_geometry,
)
from llama_qiskit_agents.quantum.simulate import (
    compare_embeddings,
    format_comparison_report,
)


def test_kernel_is_alive_thresholds() -> None:
    assert kernel_is_alive(0.5, 0.1, 0.1) is True
    assert kernel_is_alive(0.2, 0.01, 0.1) is False
    assert kernel_is_alive(0.5, 0.4, 0.1) is False
    assert kernel_is_alive(0.5, 0.1, 0.01) is False
    assert kernel_is_alive(0.5, 0.0, 0.1) is True


def test_alive_on_two_clusters_with_block_kernel() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.1, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.0, 10.1],
            [10.1, 10.0],
            [10.1, 10.1],
        ],
        dtype=float,
    )
    k = np.zeros((8, 8), dtype=float)
    k[:4, :4] = 1.0
    k[4:, 4:] = 1.0
    geo = score_kernel_geometry(x, k)
    assert geo.kernel_alive is True
    assert geo.fid_near >= 0.25
    assert geo.near_far_ratio >= 2.0
    assert geo.mean_offdiag >= 0.03


def test_dead_on_identity_kernel() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 2))
    k = np.eye(8, dtype=float)
    geo = score_kernel_geometry(x, k)
    assert geo.kernel_alive is False
    assert geo.mean_offdiag < 0.03


def test_near_far_skipped_for_tiny_n() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=float)
    k = np.eye(3)
    geo = score_kernel_geometry(x, k)
    assert geo.fid_near == 0.0
    assert geo.fid_far == 0.0
    assert geo.kernel_alive is False


def test_compare_report_includes_alive_or_dead() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 3))
    cr = compare_embeddings(
        x,
        encoding_types=[EncodingType.ANGLE],
        shots=8,
        apply_fractal_budget=False,
    )
    assert cr.kernel_geometry_by_encoding is not None
    assert EncodingType.ANGLE in cr.kernel_geometry_by_encoding
    geo = cr.kernel_geometry_by_encoding[EncodingType.ANGLE]
    assert geo.fid_near >= 0.0
    assert geo.fid_far >= 0.0
    text = format_comparison_report(cr)
    assert "Kernel alive" in text
    assert "arXiv:2609.00475" in text
    assert "ALIVE" in text or "DEAD" in text
    assert "kernel=ALIVE" in text or "kernel=DEAD" in text


def test_compare_keeps_kta_beside_alive() -> None:
    rng = np.random.default_rng(1)
    x = np.vstack(
        [
            rng.normal(size=(5, 2)),
            rng.normal(loc=3.0, size=(5, 2)),
        ]
    )
    labels = [0] * 5 + [1] * 5
    cr = compare_embeddings(
        x,
        encoding_types=[EncodingType.ANGLE],
        shots=8,
        labels=labels,
        apply_fractal_budget=False,
    )
    assert cr.kta_by_encoding is not None
    assert EncodingType.ANGLE in cr.kta_by_encoding
    assert cr.kernel_geometry_by_encoding is not None
    text = format_comparison_report(cr)
    assert "KTA=" in text
    assert "Kernel alive" in text
