"""Worker correctness and equivalence tests.

These tests are the harness for validating any worker backend (the current
Qiskit Aer CPU worker, or a future CUDA-Q / GPU worker) against an independent
analytic statevector reference. The GPU worker only needs to expose a
``process_batch``-compatible callable; feed it through ``assert_worker_matches``
below with the same reservoir/data and it will be held to the same ground truth.
"""
import numpy as np
import pytest

from conftest import reference_z_expectations

def _reference_matrix(data, qubits, layers, reservoir):
    return [reference_z_expectations(row, qubits, layers, reservoir) for row in data]


def assert_worker_matches(process_batch, data, qubits, layers, reservoir, tol=1e-9):
    """Assert a process_batch-style callable matches the analytic reference.

    Reusable by future backends. ``process_batch`` must have the signature
    ``(chunk, qubits, layers, reservoir) -> flat list of floats``.
    """
    flat = process_batch(list(data), qubits, layers, reservoir)
    produced = np.asarray(flat, dtype=float).reshape(len(data), qubits)
    expected = np.asarray(_reference_matrix(data, qubits, layers, reservoir), dtype=float)
    assert produced.shape == expected.shape
    np.testing.assert_allclose(produced, expected, atol=tol)
    return produced


@pytest.mark.slow
def test_process_batch_matches_reference(worker_module, dataset_small, reservoir_small):
    assert_worker_matches(
        worker_module.process_batch, dataset_small, 3, 2, reservoir_small
    )


@pytest.mark.slow
def test_single_row_single_qubit(worker_module):
    reservoir = worker_module.build_reservoir_params(1, 1, worker_module.QRC_SEED)
    data = np.array([[0.3]])
    assert_worker_matches(worker_module.process_batch, data, 1, 1, reservoir)


@pytest.mark.slow
def test_process_batch_output_length(worker_module, dataset_small, reservoir_small):
    out = worker_module.process_batch(list(dataset_small), 3, 2, reservoir_small)
    assert len(out) == len(dataset_small) * 3


@pytest.mark.slow
def test_expectations_in_range(worker_module, dataset_small, reservoir_small):
    out = worker_module.process_batch(list(dataset_small), 3, 2, reservoir_small)
    assert all(-1.0 <= v <= 1.0 for v in out)
