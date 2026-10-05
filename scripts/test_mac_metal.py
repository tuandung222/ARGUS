#!/usr/bin/env python3
"""Verification script for ARGUS on Apple Silicon Mac (Metal Performance Shaders - MPS)."""

import sys
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from argus.defense import adaptive_alpha
from argus.probes import OrthogonalLogisticRegression, ParallelLogisticRegression


def test_metal_support():
    print("=" * 60)
    print("ARGUS Apple Silicon Metal (MPS) Test Suite")
    print("=" * 60)

    # 1. Device check
    mps_available = torch.backends.mps.is_available()
    print(f"1. Apple MPS (Metal) Available: {mps_available}")
    device = torch.device("mps" if mps_available else "cpu")
    print(f"   Target Device: {device}")

    # 2. Adaptive Alpha calculation
    print("\n2. Testing adaptive_alpha on", device)
    hidden = torch.randn(2, 4096, device=device)
    w = torch.randn(4096, device=device)
    b = torch.tensor(0.5, device=device)
    epsilon = 1.0
    alpha = adaptive_alpha(hidden, w, b, epsilon)
    print(f"   adaptive_alpha output shape: {alpha.shape}")
    print(f"   alpha values: {alpha}")
    assert (alpha >= 0).all(), "Alpha must be non-negative"
    print("   [PASS] adaptive_alpha works on", device)

    # 3. Orthogonal Logistic Regression on MPS
    print("\n3. Testing OrthogonalLogisticRegression on", device)
    np.random.seed(42)
    basis = np.random.randn(3, 128)
    x = np.random.randn(100, 128)
    y = np.random.randint(0, 2, size=(100,))
    ortho_probe = OrthogonalLogisticRegression(basis=basis, epochs=50)
    ortho_probe.fit(x, y)
    score = ortho_probe.score(x, y)
    print(f"   Ortho probe fit score: {score:.4f}")
    assert ortho_probe.coef_ is not None, "Coefficients must be learned"
    # Verify orthogonality with basis
    dots = [float(np.dot(ortho_probe.coef_, b_vec)) for b_vec in basis]
    print(f"   Orthogonality check (should be near 0): {dots}")
    for d in dots:
        assert abs(d) < 1e-4, f"Failed orthogonality: {d}"
    print("   [PASS] Orthogonal probe works on", device)

    # 4. Parallel Logistic Regression on MPS
    print("\n4. Testing ParallelLogisticRegression on", device)
    direction = np.random.randn(128)
    direction = direction / np.linalg.norm(direction)
    parallel_probe = ParallelLogisticRegression(direction=direction, epochs=50)
    parallel_probe.fit(x, y)
    p_score = parallel_probe.score(x, y)
    print(f"   Parallel probe fit score: {p_score:.4f}")
    assert parallel_probe.coef_ is not None
    print("   [PASS] Parallel probe works on", device)

    print("\n" + "=" * 60)
    print("ALL MAC METAL (MPS) TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_metal_support()
