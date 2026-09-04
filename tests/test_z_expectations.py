"""Unit tests for exact statevector Z expectations."""
import numpy as np


def test_all_zeros_bitstring(worker_module):
    statevector = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    exp = worker_module.compute_z_expectations(statevector, 2)
    assert exp == [1.0, 1.0]


def test_all_ones_bitstring(worker_module):
    statevector = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
    exp = worker_module.compute_z_expectations(statevector, 2)
    assert exp == [-1.0, -1.0]


def test_mixed_bitstring(worker_module):
    statevector = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    exp = worker_module.compute_z_expectations(statevector, 2)
    assert exp == [1.0, -1.0]


def test_even_split(worker_module):
    statevector = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    exp = worker_module.compute_z_expectations(statevector, 1)
    assert exp == [0.0]


def test_weighted_average(worker_module):
    statevector = np.array([np.sqrt(0.75), np.sqrt(0.25)], dtype=complex)
    exp = worker_module.compute_z_expectations(statevector, 1)
    np.testing.assert_allclose(exp, [0.5])


def test_range_bounds(worker_module):
    statevector = np.ones(4, dtype=complex) / 2.0
    exp = worker_module.compute_z_expectations(statevector, 2)
    assert all(-1.0 <= e <= 1.0 for e in exp)


def test_length_matches_qubits(worker_module):
    statevector = np.zeros(16, dtype=complex)
    statevector[0] = 1.0
    exp = worker_module.compute_z_expectations(statevector, 4)
    assert len(exp) == 4
