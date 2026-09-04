# Parallel Quantum Reservoir Worker Utility Setup

## Purpose

This project is a small Python utility for running a quantum reservoir computing
(QRC) simulation in parallel over row-chunked data.

The current implementation focuses on:

- Simple process-based parallelism via joblib.
- A deterministic quantum kernel (seeded inputs and seeded Qiskit Aer simulator).
- Straightforward benchmarking across different worker counts.

## High-Level Architecture

1. Python builds a row-major dataset with NumPy.
2. Rows are split into contiguous shards.
3. Each shard is dispatched to a joblib worker process (`loky` backend).
4. Each worker runs a Qiskit Aer simulation per row and returns Z expectations.
5. joblib preserves dispatch order, so shard outputs concatenate into a
   row-major result.

Core file:

- `python/worker.py`: CLI parsing, dataset generation, circuit construction,
  joblib sharding, and the benchmark driver.

## Key Functions

- `build_reservoir_params(qubits, layers, seed)` – deterministic `(rx, rz)`
  rotation angles per layer and qubit.
- `build_circuit(row_values, qubits, layers, reservoir)` – constructs the Qiskit
- `QuantumCircuit` for one row (feature encoding, `CZ` ring, and reservoir layers).
- `compute_z_expectations(statevector, qubits)` – exact per-qubit `<Z>` from amplitudes.
- `process_batch(chunk, qubits, layers, reservoir)` – runs inside a joblib
  worker; builds its own simulator and processes only its slice of rows.
- `run_reservoir(data, qubits, layers, reservoir, n_jobs)` – shards rows
  across `n_jobs` workers and returns the flattened Z expectations.

## Setup

Activate the Python environment and install dependencies:

```bash
source ve/bin/activate
pip install -r requirements.txt
```

## Run

Run the benchmark:

```bash
python python/worker.py --rows 3000 --cols 6
```

Sweep specific job counts and tune the circuit:

```bash
python python/worker.py --rows 3000 --cols 6 --qrc-layers 3 --jobs 1 4 8
```

Relevant CLI options:

- `--rows` – dataset rows, default `3000`.
- `--cols` – columns / qubits, default `6`.
- `--qrc-layers` – reservoir layers, default `2`.
- `--jobs` – one or more joblib worker counts to sweep, default `1 2 4 8 16`.

## Notes on Determinism and Performance

- Input generation uses a seeded NumPy generator (`QRC_SEED = 42`).
- BLAS/OpenMP thread counts are pinned to `1` before importing NumPy/Qiskit so
  parallel joblib workers do not oversubscribe CPU cores.
- Reservoir angles are derived from the fixed seed, and exact statevector
  expectations are independent of shard count.
- More workers reduce wall-clock time up to the point where process startup and
  simulator construction overhead dominates for a given dataset size.

## Direction for Reusable Utility Evolution

To evolve this into a general "shard rows across N Python workers" utility:

1. Make the per-row kernel pluggable instead of hard-coded to the QRC circuit.
2. Make the chunking strategy configurable (fixed rows, dynamic queue, strided).
3. Add a retry policy around worker failures.
4. Return or persist the result matrix instead of discarding it after timing.
5. Add optional worker pool warmup and lifecycle metrics.
