"""
tests/conftest.py
=================
Shared pytest fixtures used across unit, integration, and privacy audit tests.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root is on the Python path so imports work cleanly
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fixed random seed for reproducibility
SEED = 42
N_FEATURES = 64
N_SAMPLES = 400


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def synthetic_data(rng):
    """
    Small synthetic dataset for unit tests.
    NOT used for actual fraud detection — only for testing model plumbing.
    """
    X = rng.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)
    y = (rng.random(N_SAMPLES) < 0.15).astype(np.int32)  # ~15% positive
    return X, y


@pytest.fixture(scope="session")
def synthetic_train_val(rng):
    X = rng.standard_normal((500, N_FEATURES)).astype(np.float32)
    y = (rng.random(500) < 0.15).astype(np.int32)
    X_train, y_train = X[:400], y[:400]
    X_val, y_val = X[400:], y[400:]
    return (X_train, y_train), (X_val, y_val)


@pytest.fixture(scope="session")
def five_bank_keypairs():
    """X25519 keypairs for 5 banks, generated once per test session."""
    from fl.secure_agg.keys import KeyPair
    return {i: KeyPair.generate() for i in range(5)}


@pytest.fixture(scope="session")
def five_bank_seeds(five_bank_keypairs):
    """All pairwise seeds for 5 banks."""
    from fl.secure_agg.keys import derive_shared_seed
    kps = five_bank_keypairs
    seeds = {}
    for i in range(5):
        for j in range(i + 1, 5):
            seed = derive_shared_seed(kps[i], kps[j].public_bytes, i, j, "test_round_1")
            seeds[(i, j)] = seed
    return seeds
