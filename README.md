# CUDA-Q Quantum Reservoir Benchmark

This project runs a quantum reservoir computing simulation with CUDA-Q. Rows are submitted asynchronously to CUDA-Q's `mqpu` target, which exposes one logical QPU for each available NVIDIA GPU. With one GPU, all rows run on that GPU; with multiple GPUs, successive rows are assigned round-robin while results remain in row order.

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
- `cudaq.get_target().num_qpus()` determines the available GPU count after selecting `mqpu`.
- `cudaq.observe_async` receives `qpu_id=row_index % num_gpus`, providing deterministic row-to-GPU distribution.
- Zero-shot expectation values are returned in row-major order.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
pytest -m "not slow"
```
