# CUDA-Q Quantum Reservoir Benchmark

This project runs a quantum reservoir computing simulation with CUDA-Q. Rows are submitted asynchronously to CUDA-Q's `mqpu` target, which exposes one logical QPU for each available NVIDIA GPU. Python assigns balanced row batches to those QPUs, while CUDA-Q executes the futures concurrently and results remain in row order.

## Setup

```bash
source ve/bin/activate
pip install -r requirements.txt
```

The runtime dependencies are `numpy` and `cudaq`.

## Run

```bash
python python/worker.py --rows 3000 --cols 6 --qrc-layers 2
```

Options:

- `--rows`: number of dataset rows, default `3000`.
- `--cols`: columns per row and number of qubits, default `6`.
- `--qrc-layers`: reservoir layers, default `2`.

The NVIDIA target requires CUDA-Q and at least one available NVIDIA GPU. Tests use CUDA-Q's `qpp-cpu` target where a GPU is not required.

## Implementation

- Input data and reservoir angles use fixed seed `42`.
- Each row is encoded into a CUDA-Q kernel with `RY`/`RZ` feature gates, `CZ` entanglement, and `RX`/`RZ` reservoir layers.
- `cudaq.num_available_gpus()` determines the visible GPU count after selecting `mqpu`.
- Rows are assigned to contiguous balanced batches with an explicit `qpu_id`.
- `cudaq.observe_async` submits every queued row before `.get()` retrieves results, allowing concurrent execution.
- Zero-shot expectation values are returned in row-major order.

The `mqpu` target is for many independent circuit evaluations. The `mgpu` target is a different mode for spreading one large statevector across multiple GPUs.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
pytest -m "not slow"
```
