# Parallel Quantum Reservoir Computing Benchmark

This project runs a Qiskit Aer quantum reservoir computing (QRC) simulation over a
row-major dataset and benchmarks how the runtime scales as the work is sharded
across multiple worker processes with [joblib](https://joblib.readthedocs.io/).

Everything lives in a single Python entrypoint: [python/worker.py](python/worker.py).

## Setup

Create/activate the bundled virtual environment and install dependencies:

```bash
source ve/bin/activate
pip install -r requirements.txt
```

Dependencies (see [requirements.txt](requirements.txt)):

- `qiskit>=1.0`
- `qiskit-aer>=0.14`
- `numpy>=1.24`
- `joblib>=1.3`

## Run

```bash
python python/worker.py --rows 3000 --cols 6
```

Command-line options:

- `--rows <n>`: number of dataset rows to simulate, default `3000`.
- `--cols <n>`: columns per row, which is also the number of qubits, default `6`.
- `--qrc-layers <n>`: number of reservoir layers in the circuit, default `2`.
- `--jobs <n> [<n> ...]`: one or more joblib worker counts to sweep, default `1 2 4 8 16`.

Each requested job count is clamped to `min(jobs, rows)`.

## Current Behavior

- The dataset is generated with NumPy's `default_rng`, fixed seed `42`, and a
  uniform distribution over `[-1.0, 1.0]`, giving a `rows x cols` matrix.
- For each row, a Qiskit circuit is built with feature encoding (`RY`/`RZ`), a
  `CZ` entangling ring, and `--qrc-layers` reservoir layers (`RX`/`RZ` + `CZ`).
- The per-qubit `<Z>` expectation is computed exactly from the statevector.
- Reservoir rotation angles are derived deterministically from the fixed seed, so
  results are reproducible regardless of sharding.
- Rows are split into contiguous shards across `n_jobs` joblib workers (`loky`
  backend). Each worker only receives its own slice of the data, and dispatch
  order is preserved so concatenating the shard outputs yields row-major results.
- The program prints the dataset shape and the wall-clock runtime for each
  requested job count.

## Output

```
Dataset: 3000x6, qrc-layers=2

Runtime by joblib job count:
1	4.1234
2	2.2103
4	1.1876
8	0.7421
16	0.6210
```

Each row under the header is `<job count>\t<seconds>`.

## Determinism and Threading

- Input generation is seeded (`QRC_SEED = 42`).
- BLAS/OpenMP thread counts are pinned to `1` before importing NumPy/Qiskit to
  avoid oversubscription when multiple joblib workers run in parallel.
- The simulator uses explicit seeding and fixed shots for reproducible output.

## Tests

Install the dev dependencies and run the suite with pytest:

```bash
pip install -r requirements-dev.txt
pytest
```

Run only the fast unit tests (skip the ones that spawn joblib workers and run
Qiskit simulations):

```bash
pytest -m "not slow"
```

The suite covers:

- **Unit tests** for the pure functions (`build_reservoir_params`,
  `build_circuit`, `compute_z_expectations`).
- **Worker equivalence** ([tests/test_worker_equivalence.py](tests/test_worker_equivalence.py)):
  the worker output is compared against an independent analytic statevector
  reference in [tests/conftest.py](tests/conftest.py). The `assert_worker_matches`
  helper takes any `process_batch`-style callable, so a future GPU / CUDA-Q worker
  can be validated against the exact same ground truth as the CPU worker.
- **Shard invariance** ([tests/test_shard_invariance.py](tests/test_shard_invariance.py)):
  results are identical regardless of how many workers the rows are split across —
  the property a mixed CPU + GPU run must also preserve.
- **CLI integration** ([tests/test_cli.py](tests/test_cli.py)): argument
  validation and end-to-end runs of the entrypoint.

### Validating a future GPU / CUDA-Q worker

When you add a GPU worker, give it a `process_batch(chunk, qubits, layers,
reservoir)`-compatible function and assert it against the shared reference:

```python
from conftest import assert_worker_matches
from gpu_worker import process_batch as gpu_process_batch

assert_worker_matches(gpu_process_batch, data, qubits, layers, reservoir)
```

If it matches the analytic reference within tolerance, the GPU worker is doing the
same thing to each circuit as the CPU worker.
