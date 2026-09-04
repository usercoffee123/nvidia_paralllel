"""Shared fixtures and helpers for the worker test suite.

The most important piece here is :func:`reference_z_expectations`, an independent
statevector implementation of the same circuit that ``python/worker.py`` builds.
It is used as ground truth so that *any* worker implementation - the current
Qiskit Aer CPU worker or a future CUDA-Q / GPU worker - can be validated against
the exact same analytic result. See ``tests/test_worker_equivalence.py``.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"


def _load_worker():
    """Import python/worker.py under its real name so loky children can re-import it.

    joblib pickles ``process_batch`` by reference (module + qualname), and the
    loky worker processes re-import that module by name. Loading it via a synthetic
    name would break multi-process runs, so we put ``python/`` on sys.path and
    import it as ``worker``.
    """
    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))
    return importlib.import_module("worker")


worker = _load_worker()


@pytest.fixture(scope="session")
def worker_module():
    """The imported worker module under test."""
    return worker


@pytest.fixture(scope="session")
def reservoir_small(worker_module):
    """A small deterministic reservoir: 3 qubits, 2 layers."""
    return worker_module.build_reservoir_params(3, 2, worker_module.QRC_SEED)


@pytest.fixture
def dataset_small():
    """A small deterministic dataset: 5 rows x 3 cols in [-1, 1]."""
    rng = np.random.default_rng(worker.QRC_SEED)
    return rng.uniform(-1.0, 1.0, size=(5, 3))


# ---------------------------------------------------------------------------
# Analytic reference implementation (ground truth)
# ---------------------------------------------------------------------------

# Single-qubit gate matrices (little-endian qubit ordering, qubit 0 is LSB).
_I2 = np.eye(2, dtype=complex)


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    e = np.exp(1j * theta / 2.0)
    return np.array([[1.0 / e, 0.0], [0.0, e]], dtype=complex)


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _apply_single(state: np.ndarray, gate: np.ndarray, target: int, qubits: int) -> np.ndarray:
    """Apply a 1-qubit gate to `target` using tensor reshaping (LSB = qubit 0)."""
    # Reshape so axis ordering is [q_{n-1}, ..., q_1, q_0].
    shaped = state.reshape([2] * qubits)
    axis = qubits - 1 - target
    moved = np.moveaxis(shaped, axis, 0)
    out = np.tensordot(gate, moved, axes=([1], [0]))
    out = np.moveaxis(out, 0, axis)
    return out.reshape(-1)


def _apply_cz(state: np.ndarray, a: int, b: int, qubits: int) -> np.ndarray:
    """Apply a CZ between qubits a and b (symmetric)."""
    shaped = state.reshape([2] * qubits)
    idx = [slice(None)] * qubits
    axis_a = qubits - 1 - a
    axis_b = qubits - 1 - b
    idx[axis_a] = 1
    idx[axis_b] = 1
    shaped[tuple(idx)] *= -1.0
    return shaped.reshape(-1)


def reference_statevector(row_values, qubits: int, layers: int, reservoir) -> np.ndarray:
    """Independent statevector simulation of the worker circuit (pre-measurement)."""
    def feature_angle(index: int) -> float:
        return math.pi * math.tanh(row_values[index % qubits])

    state = np.zeros(2 ** qubits, dtype=complex)
    state[0] = 1.0

    for q in range(qubits):
        theta = feature_angle(q)
        state = _apply_single(state, _ry(theta), q, qubits)
        state = _apply_single(state, _rz(0.5 * theta), q, qubits)

    if qubits > 1:
        for q in range(qubits):
            state = _apply_cz(state, q, (q + 1) % qubits, qubits)

    for layer in range(layers):
        for q in range(qubits):
            base_rx, base_rz = reservoir[layer][q]
            injection = 0.35 * feature_angle(q + layer)
            state = _apply_single(state, _rx(base_rx + injection), q, qubits)
            state = _apply_single(state, _rz(base_rz - 0.25 * injection), q, qubits)
        if qubits > 1:
            for q in range(qubits):
                state = _apply_cz(state, q, (q + 1) % qubits, qubits)

    return state


def reference_z_expectations(row_values, qubits: int, layers: int, reservoir):
    """Exact <Z> expectations (no shot noise) using the worker's output convention.

    The worker reads Qiskit count bitstrings as ``bitstring[q]`` for output index
    ``q``. Qiskit count strings are big-endian (leftmost char = highest-numbered
    qubit), so output index ``q`` corresponds to physical qubit ``qubits - 1 - q``.
    We mirror that mapping here so the reference is directly comparable to the
    worker's output, and so any future GPU/CUDA-Q worker is held to the same
    ordering convention.
    """
    state = reference_statevector(row_values, qubits, layers, reservoir)
    probs = np.abs(state) ** 2
    expectations = []
    for q in range(qubits):
        physical_qubit = qubits - 1 - q
        exp = 0.0
        for basis in range(2 ** qubits):
            bit = (basis >> physical_qubit) & 1
            exp += probs[basis] * (1.0 - 2.0 * bit)
        expectations.append(exp)
    return expectations
