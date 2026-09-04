"""Unit tests for build_reservoir_params."""
import math


def test_shape(worker_module):
    params = worker_module.build_reservoir_params(4, 3, 42)
    assert len(params) == 3
    assert all(len(layer) == 4 for layer in params)
    assert all(len(pair) == 2 for layer in params for pair in layer)


def test_deterministic(worker_module):
    a = worker_module.build_reservoir_params(5, 2, 42)
    b = worker_module.build_reservoir_params(5, 2, 42)
    assert a == b


def test_seed_changes_output(worker_module):
    a = worker_module.build_reservoir_params(5, 2, 42)
    b = worker_module.build_reservoir_params(5, 2, 43)
    assert a != b


def test_angles_within_2pi(worker_module):
    params = worker_module.build_reservoir_params(6, 4, 7)
    for layer in params:
        for rx, rz in layer:
            assert 0.0 <= rx < 2.0 * math.pi + 1e-9
            assert 0.0 <= rz < 2.0 * math.pi + 1e-9


def test_zero_layers(worker_module):
    assert worker_module.build_reservoir_params(4, 0, 42) == []


def test_matches_reference_formula(worker_module):
    seed, layers, qubits = 42, 2, 3
    params = worker_module.build_reservoir_params(qubits, layers, seed)
    for layer in range(layers):
        for q in range(qubits):
            h = (seed + 101 * (layer + 1) + 1009 * (q + 1)) & 0xFFFFFFFF
            rx = ((h % 10007) / 10007.0) * (2.0 * math.pi)
            rz = (((h * 17 + 23) % 10009) / 10009.0) * (2.0 * math.pi)
            assert params[layer][q] == (rx, rz)
