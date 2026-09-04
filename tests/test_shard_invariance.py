"""Shard-invariance tests for run_reservoir.

The result must not depend on how many joblib workers the rows are split across.
This is also the exact property a future GPU worker must preserve: splitting work
between CPU and GPU workers must produce the same row-major output as running it
all on one backend.
"""
import numpy as np
import pytest

from conftest import reference_z_expectations


@pytest.fixture
def data(worker_module):
    rng = np.random.default_rng(worker_module.QRC_SEED)
    return rng.uniform(-1.0, 1.0, size=(6, 3))


@pytest.mark.slow
def test_result_length(worker_module, data, reservoir_small):
    out = worker_module.run_reservoir(data, 3, 2, reservoir_small, 1)
    assert len(out) == data.shape[0] * 3


@pytest.mark.slow
def test_invariant_across_job_counts(worker_module, data, reservoir_small):
    baseline = worker_module.run_reservoir(data, 3, 2, reservoir_small, 1)
    for n_jobs in (2, 3, 6):
        result = worker_module.run_reservoir(data, 3, 2, reservoir_small, n_jobs)
        np.testing.assert_allclose(result, baseline, atol=1e-9)


@pytest.mark.slow
def test_row_major_order_matches_reference(worker_module, data, reservoir_small):
    out = np.asarray(
        worker_module.run_reservoir(data, 3, 2, reservoir_small, 3)
    ).reshape(data.shape[0], 3)
    expected = np.asarray(
        [reference_z_expectations(row, 3, 2, reservoir_small) for row in data]
    )
    np.testing.assert_allclose(out, expected, atol=1e-9)


@pytest.mark.slow
def test_more_jobs_than_rows_clamped(worker_module, reservoir_small):
    small = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    out = worker_module.run_reservoir(small, 3, 2, reservoir_small, 16)
    assert len(out) == small.shape[0] * 3
