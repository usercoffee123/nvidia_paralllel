"""Unit tests for build_circuit structure."""
import pytest


@pytest.fixture
def reservoir(worker_module):
    return worker_module.build_reservoir_params(3, 2, worker_module.QRC_SEED)


def test_registers(worker_module, reservoir):
    qc = worker_module.build_circuit([0.1, 0.2, 0.3], 3, 2, reservoir)
    assert qc.num_qubits == 3
    assert qc.num_clbits == 0


def test_gate_counts(worker_module, reservoir):
    qubits, layers = 3, 2
    qc = worker_module.build_circuit([0.1, 0.2, 0.3], qubits, layers, reservoir)
    ops = qc.count_ops()
    # Feature encoding: RY per qubit; reservoir layers add none.
    assert ops.get("ry", 0) == qubits
    # RZ: 1 per qubit in encoding + 1 per qubit per reservoir layer.
    assert ops.get("rz", 0) == qubits + qubits * layers
    # RX: 1 per qubit per reservoir layer.
    assert ops.get("rx", 0) == qubits * layers
    # CZ: one ring after encoding + one ring per reservoir layer.
    assert ops.get("cz", 0) == qubits * (1 + layers)


def test_single_qubit_no_cz(worker_module):
    reservoir = worker_module.build_reservoir_params(1, 2, 42)
    qc = worker_module.build_circuit([0.5], 1, 2, reservoir)
    assert qc.count_ops().get("cz", 0) == 0
    assert qc.num_qubits == 1


def test_zero_layers_circuit(worker_module):
    reservoir = worker_module.build_reservoir_params(3, 0, 42)
    qc = worker_module.build_circuit([0.1, 0.2, 0.3], 3, 0, reservoir)
    ops = qc.count_ops()
    assert ops.get("rx", 0) == 0
    assert ops.get("cz", 0) == 3  # only the encoding ring


def test_deterministic_structure(worker_module, reservoir):
    a = worker_module.build_circuit([0.1, 0.2, 0.3], 3, 2, reservoir)
    b = worker_module.build_circuit([0.1, 0.2, 0.3], 3, 2, reservoir)
    assert a == b
