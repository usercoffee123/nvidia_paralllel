"""Row-order and output-shape tests for CUDA-Q scheduling."""
import numpy as np
import pytest

from conftest import reference_z_expectations


@pytest.fixture
def data(worker_module):
    rng = np.random.default_rng(worker_module.QRC_SEED)
    return rng.uniform(-1.0, 1.0, size=(6, 3))


@pytest.mark.slow
def test_result_length(worker_module, data, reservoir_small):
    out = worker_module.run_reservoir(data, 3, 2, reservoir_small, target="qpp-cpu")
    assert len(out) == data.shape[0] * 3


@pytest.mark.slow
def test_invariant_across_runs(worker_module, data, reservoir_small):
    baseline = worker_module.run_reservoir(data, 3, 2, reservoir_small, target="qpp-cpu")
    result = worker_module.run_reservoir(data, 3, 2, reservoir_small, target="qpp-cpu")
    np.testing.assert_allclose(result, baseline, atol=1e-9)


@pytest.mark.slow
def test_row_major_order_matches_reference(worker_module, data, reservoir_small):
    out = np.asarray(
        worker_module.run_reservoir(data, 3, 2, reservoir_small, target="qpp-cpu")
    ).reshape(data.shape[0], 3)
    expected = np.asarray(
        [reference_z_expectations(row, 3, 2, reservoir_small) for row in data]
    )
    np.testing.assert_allclose(out, expected, atol=1e-9)


@pytest.mark.slow
def test_small_batch(worker_module, reservoir_small):
    small = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    out = worker_module.run_reservoir(small, 3, 2, reservoir_small, target="qpp-cpu")
    assert len(out) == small.shape[0] * 3
