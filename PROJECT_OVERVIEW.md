# Project Overview: Parallel Quantum Reservoir Computing Benchmark

## Purpose

This project demonstrates how a quantum reservoir computing (QRC) simulation can be
parallelized across worker processes. A row-major dataset is generated in memory,
split into contiguous row shards, and each shard is processed by a
[joblib](https://joblib.readthedocs.io/) worker that runs a Qiskit Aer state-vector
simulation per row and returns a matrix of Z expectation values. The program
benchmarks how the total runtime scales with the number of parallel workers.

The entire implementation lives in a single Python file: `python/worker.py`.

---

## Architecture

```
+------------------------+        +------------------------+
|   Main process         |        |   joblib workers        |
| (dataset + benchmark)  |  fork  |  (loky backend)         |
|                        | -----> |  one Qiskit sim / shard |
+------------------------+        +------------------------+
```

* **Main process** (`main` in `python/worker.py`)
  * Parses CLI arguments (`--rows`, `--cols`, `--qrc-layers`, `--jobs`).
  * Generates a random dataset with NumPy (`default_rng`, seed `42`).
  * Builds deterministic reservoir rotation parameters from the fixed seed.
  * Sweeps the requested joblib job counts and times each run.

* **joblib workers** (`process_batch` in `python/worker.py`)
  * Each worker receives only its own contiguous slice of the dataset.
  * Builds a Qiskit `QuantumCircuit` per row and runs it on an `AerSimulator`
    (`statevector` method, seeded, single-threaded).
  * Returns the flattened Z expectations for its rows.

* **Determinism**
  * Reservoir angles and simulator output are seeded, so results are identical
    regardless of how many workers the rows are sharded across.

---

## Data Flow

1. **Dataset generation** – `np.random.default_rng(42)` produces a `rows x cols`
   matrix uniformly distributed in `[-1.0, 1.0]`.
2. **Reservoir parameters** – `build_reservoir_params` derives deterministic
   `(rx, rz)` rotation angles for every layer and qubit from the fixed seed.
3. **Row sharding** – `run_reservoir` splits the rows into `n_jobs` contiguous
   shards using integer bounds so each worker gets a near-equal slice.
4. **Quantum simulation** – Each worker builds a circuit per row via
   `build_circuit`: feature encoding (`RY`/`RZ`), a `CZ` entangling ring, then
   `--qrc-layers` reservoir layers (`RX`/`RZ` + `CZ`), followed by Z-basis
  and returns its statevector.
5. **Expectation values** – `compute_z_expectations` converts statevector
  probabilities into an exact per-qubit `<Z>` value in `[-1, 1]`.
6. **Aggregation** – joblib preserves dispatch order, so concatenating the shard
   outputs yields a row-major expectation matrix. The benchmark discards the
   result after timing each job count.

---

## Circuit Construction

For a row with `qubits = cols`:

* **Feature encoding** – For each qubit `q`, `theta = pi * tanh(row[q])`, then
  `RY(theta)` and `RZ(0.5 * theta)`.
* **Entangling ring** – `CZ(q, (q + 1) % qubits)` for each qubit (when `qubits > 1`).
* **Reservoir layers** – For each layer, apply `RX(base_rx + injection)` and
  `RZ(base_rz - 0.25 * injection)` per qubit, where `injection` mixes the feature
  angle back in, followed by another `CZ` ring.

---

## Benchmarking

The program sweeps the job counts passed via `--jobs` (default `{1, 2, 4, 8, 16}`),
clamping each to `min(jobs, rows)`. For each count it prints the wall-clock seconds
for the full run.

Typical output (truncated):

```
Dataset: 3000x6, qrc-layers=2

Runtime by joblib job count:
1	4.1234
2	2.2103
4	1.1876
8	0.7421
16	0.6210
```

---

## Running

### Prerequisites

* Python 3 with `numpy`, `qiskit`, `qiskit-aer`, and `joblib` installed.

### Setup

```bash
source ve/bin/activate
pip install -r requirements.txt
```

### Command-line usage

```bash
python python/worker.py --rows <NUM_ROWS> --cols <NUM_COLS>
```

* `--rows` – number of dataset rows, default `3000`.
* `--cols` – number of columns / qubits, default `6`.
* `--qrc-layers` – number of reservoir layers, default `2`.
* `--jobs` – one or more joblib worker counts to sweep, default `1 2 4 8 16`.

---

## Configuration Details

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rows` | Number of rows in the dataset. | `3000` |
| `cols` | Number of columns (also qubits). | `6` |
| `qrc-layers` | Reservoir layers in the circuit. | `2` |
| `jobs` | joblib worker counts to benchmark. | `1 2 4 8 16` |
| RNG seed | Fixed to `42` for reproducibility. | `42` |
| Data range | Uniform real distribution `[-1.0, 1.0]`. | – |

---

## Extending the Project

* **Change the kernel** – modify `build_circuit` to implement a different quantum
  or classical per-row computation.
* **Change sharding** – adjust `run_reservoir` to use a different chunking strategy
  or joblib backend.
* **Expose more options** – extend the `argparse` setup in `main`.
* **Add outputs** – return or persist the expectation matrix instead of discarding
  it after timing.

---

## License

This example code is provided under the MIT License. See the `LICENSE` file for details.
