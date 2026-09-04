"""CUDA-Q worker equivalence tests."""
import numpy as np
import pytest

from test_worker_equivalence import assert_worker_matches

cudaq = pytest.importorskip("cudaq")

gpu_required = pytest.mark.skipif(
    cudaq.num_available_gpus() == 0, reason="no NVIDIA GPU available"
)


@pytest.fixture
def cpu_cudaq_target():
    previous = cudaq.get_target().name
    cudaq.set_target("qpp-cpu")
    try:
        yield
    finally:
        cudaq.set_target(previous)


@pytest.mark.slow
def test_cudaq_worker_matches_reference(
    worker_module, dataset_small, reservoir_small, cpu_cudaq_target
):
    assert_worker_matches(
        lambda chunk, qubits, layers, reservoir: worker_module.process_batch_cudaq(
            chunk, qubits, layers, reservoir, target="qpp-cpu"
        ),
        dataset_small,
        3,
        2,
        reservoir_small,
    )


@pytest.mark.slow
def test_hybrid_output_matches_cpu_and_cudaq(
    worker_module, dataset_small, reservoir_small, cpu_cudaq_target
):
    expected = worker_module.run_reservoir(
        dataset_small, 3, 2, reservoir_small, n_jobs=1
    )
    gpu_rows = int(len(dataset_small) * 40.0 / 100.0)
    actual = worker_module.run_hybrid(
        dataset_small,
        3,
        2,
        reservoir_small,
        n_jobs=1,
        gpu_percent=40.0,
        cuda_target="qpp-cpu",
    )
    assert gpu_rows == 2
    np.testing.assert_allclose(actual, expected, atol=1e-9)


@gpu_required
@pytest.mark.gpu
@pytest.mark.slow
def test_cudaq_worker_matches_reference_on_gpu(
    worker_module, dataset_small, reservoir_small
):
    previous = cudaq.get_target().name
    try:
        assert_worker_matches(
            lambda chunk, qubits, layers, reservoir: worker_module.process_batch_cudaq(
                chunk, qubits, layers, reservoir, target="nvidia"
            ),
            dataset_small,
            3,
            2,
            reservoir_small,
            tol=1e-6,  # nvidia target defaults to fp32
        )
    finally:
        cudaq.set_target(previous)


@gpu_required
@pytest.mark.gpu
@pytest.mark.slow
def test_hybrid_on_gpu_matches_cpu(worker_module, dataset_small, reservoir_small):
    previous = cudaq.get_target().name
    try:
        expected = worker_module.run_reservoir(
            dataset_small, 3, 2, reservoir_small, n_jobs=1
        )
        actual = worker_module.run_hybrid(
            dataset_small,
            3,
            2,
            reservoir_small,
            n_jobs=1,
            gpu_percent=40.0,
            cuda_target="nvidia",
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    finally:
        cudaq.set_target(previous)