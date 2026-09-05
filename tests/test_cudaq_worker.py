"""CUDA-Q worker equivalence and scheduling tests."""
import numpy as np
import pytest

from test_worker_equivalence import assert_worker_matches

cudaq = pytest.importorskip("cudaq")


def test_check_gpu_memory_allows_fitting_statevector(worker_module, monkeypatch):
    monkeypatch.setattr(worker_module, "get_gpu_memory_mb", lambda: (16000, 24000))
    assert worker_module.check_gpu_memory(10, 1) == (16000, 24000, 0.0078125)


def test_check_gpu_memory_rejects_insufficient_memory(worker_module, monkeypatch):
    monkeypatch.setattr(worker_module, "get_gpu_memory_mb", lambda: (100, 16000))
    with pytest.raises(RuntimeError, match="GPU memory is insufficient"):
        worker_module.check_gpu_memory(30, 1)


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


def test_gpu_count_controls_qpu_assignment(worker_module, monkeypatch):
    data = np.arange(12, dtype=float).reshape(4, 3)
    assignments = []

    class Future:
        def get(self):
            return self

        def expectation(self, _term):
            return 0.0

    monkeypatch.setattr(worker_module.cudaq, "set_target", lambda *args, **kwargs: None)
    class Spin:
        @staticmethod
        def z(_index):
            return Spin()

        def __add__(self, _other):
            return self

    monkeypatch.setattr(worker_module.cudaq, "spin", Spin)
    monkeypatch.setattr(
        worker_module.cudaq,
        "observe_async",
        lambda *args, **kwargs: assignments.append(kwargs["qpu_id"]) or Future(),
    )
    monkeypatch.setattr(
        worker_module.cudaq,
        "num_available_gpus",
        lambda: 2,
    )
    monkeypatch.setattr(worker_module, "_build_cudaq_kernel", lambda *args: object())

    worker_module.process_batch_cudaq(data, 3, 1, [], target="nvidia")
    assert assignments == [0, 0, 1, 1]
