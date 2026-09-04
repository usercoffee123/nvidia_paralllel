"""Unit tests for the CUDA-Q kernel builder."""
import pytest


@pytest.fixture
def reservoir(worker_module):
    return worker_module.build_reservoir_params(3, 2, worker_module.QRC_SEED)


def test_kernel_builds(worker_module, reservoir):
    kernel = worker_module._build_cudaq_kernel(3, 2, reservoir)
    assert kernel is not None


def test_single_qubit_kernel_builds(worker_module):
    reservoir = worker_module.build_reservoir_params(1, 2, 42)
    assert worker_module._build_cudaq_kernel(1, 2, reservoir) is not None
