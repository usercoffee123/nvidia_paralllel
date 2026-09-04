#!/usr/bin/env python3
"""Quantum reservoir computing benchmark using Qiskit Aer + joblib."""
import os

# Pin all threading to 1 — must be set BEFORE importing numpy/qiskit so that
# BLAS/OpenMP libraries pick up the values at load time.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import time

import numpy as np
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

try:
    import cudaq
except ImportError:  # CUDA-Q is optional for CPU-only installations.
    cudaq = None

QRC_SEED = 42


def build_reservoir_params(qubits: int, layers: int, seed: int):
    """Generate deterministic pseudo-random rotation angles for reservoir layers."""
    params = []
    for layer in range(layers):
        row = []
        for q in range(qubits):
            h = (seed + 101 * (layer + 1) + 1009 * (q + 1)) & 0xFFFFFFFF
            rx = ((h % 10007) / 10007.0) * (2.0 * math.pi)
            rz = (((h * 17 + 23) % 10009) / 10009.0) * (2.0 * math.pi)
            row.append((rx, rz))
        params.append(row)
    return params


def build_circuit(row_values, qubits: int, layers: int, reservoir):
    """Construct a Qiskit QuantumCircuit for one row of data."""
    qc = QuantumCircuit(qubits)

    def feature_angle(index: int) -> float:
        return math.pi * math.tanh(row_values[index % qubits])

    # Feature encoding: RY + RZ on each qubit
    for q in range(qubits):
        theta = feature_angle(q)
        qc.ry(theta, q)
        qc.rz(0.5 * theta, q)

    # Entangling layer: CZ ring
    if qubits > 1:
        for q in range(qubits):
            qc.cz(q, (q + 1) % qubits)

    # Reservoir layers
    for layer in range(layers):
        for q in range(qubits):
            base_rx, base_rz = reservoir[layer][q]
            injection = 0.35 * feature_angle(q + layer)
            qc.rx(base_rx + injection, q)
            qc.rz(base_rz - 0.25 * injection, q)
        if qubits > 1:
            for q in range(qubits):
                qc.cz(q, (q + 1) % qubits)

    return qc


def compute_z_expectations(statevector, qubits: int):
    """Compute exact per-qubit <Z> expectations from a statevector."""
    probabilities = np.abs(np.asarray(statevector)) ** 2
    expectations = [0.0] * qubits
    for q in range(qubits):
        physical_qubit = qubits - 1 - q
        expectation = 0.0
        for basis, probability in enumerate(probabilities):
            bit = (basis >> physical_qubit) & 1
            expectation += probability * (1.0 - 2.0 * bit)
        expectations[q] = float(expectation.real)
    return expectations


def process_batch(chunk, qubits: int, layers: int, reservoir):
    """Process one shard of rows and return their flattened Z expectations.

    Runs inside a joblib worker process. It receives only its own slice of the
    dataset and builds its own simulator; the fixed seed keeps results
    deterministic regardless of how the rows are sharded.
    """
    simulator = AerSimulator(method="statevector", max_parallel_threads=1)
    out = []
    for row in chunk:
        qc = build_circuit(row, qubits, layers, reservoir)
        qc.save_statevector()
        statevector = simulator.run(qc).result().get_statevector()
        out.extend(compute_z_expectations(statevector, qubits))
    return out


def process_batch_cudaq(chunk, qubits: int, layers: int, reservoir, target="nvidia"):
    """Process rows with CUDA-Q and return flattened exact Z expectations.

    ``target`` is injectable so tests can use CUDA-Q's CPU statevector target;
    production calls use the NVIDIA target by default.
    """
    if cudaq is None:
        raise RuntimeError("CUDA-Q is required when --gpu-percent is nonzero")

    cudaq.set_target(target)
    out = []
    for row in chunk:
        kernel = cudaq.make_kernel()
        qubit_register = kernel.qalloc(qubits)

        def feature_angle(index: int) -> float:
            return math.pi * math.tanh(row[index % qubits])

        for q in range(qubits):
            theta = feature_angle(q)
            kernel.ry(theta, qubit_register[q])
            kernel.rz(0.5 * theta, qubit_register[q])

        if qubits > 1:
            for q in range(qubits):
                kernel.cz(qubit_register[q], qubit_register[(q + 1) % qubits])

        for layer in range(layers):
            for q in range(qubits):
                base_rx, base_rz = reservoir[layer][q]
                injection = 0.35 * feature_angle(q + layer)
                kernel.rx(base_rx + injection, qubit_register[q])
                kernel.rz(base_rz - 0.25 * injection, qubit_register[q])
            if qubits > 1:
                for q in range(qubits):
                    kernel.cz(qubit_register[q], qubit_register[(q + 1) % qubits])

        # Output index q maps to physical qubit qubits-1-q (see compute_z_expectations).
        for q in range(qubits):
            result = cudaq.observe(
                kernel, cudaq.spin.z(qubits - 1 - q), shots_count=0
            )
            out.append(float(result.expectation()))
    return out


def run_reservoir(data, qubits: int, layers: int, reservoir, n_jobs: int):
    """Shard rows across n_jobs joblib workers and return flattened Z expectations.

    Each shard is pickled with only its own slice of the data so a worker never
    holds the whole dataset. joblib preserves dispatch order, so concatenating
    the returned batches yields row-major output.
    """
    rows = len(data)
    if rows == 0:
        return []
    n_jobs = max(1, min(n_jobs, rows))
    bounds = [(rows * i) // n_jobs for i in range(n_jobs + 1)]
    batches = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_batch)(data[bounds[i]:bounds[i + 1]], qubits, layers, reservoir)
        for i in range(n_jobs)
    )
    return [value for batch in batches for value in batch]


def run_hybrid(
    data,
    qubits: int,
    layers: int,
    reservoir,
    n_jobs: int,
    gpu_percent: float,
    cuda_target="nvidia",
):
    """Run a percentage of rows with CUDA-Q and the rest on CPU workers."""
    if not 0.0 <= gpu_percent <= 100.0:
        raise ValueError("gpu-percent must be between 0 and 100")
    if gpu_percent == 0.0:
        return run_reservoir(data, qubits, layers, reservoir, n_jobs)

    rows = len(data)
    gpu_rows = int(rows * gpu_percent / 100.0)
    if gpu_rows == 0:
        return run_reservoir(data, qubits, layers, reservoir, n_jobs)
    cpu_data = data[gpu_rows:]
    gpu_data = data[:gpu_rows]
    with ThreadPoolExecutor(max_workers=1) as executor:
        cpu_future = executor.submit(
            run_reservoir, cpu_data, qubits, layers, reservoir, n_jobs
        )
        gpu_result = process_batch_cudaq(
            gpu_data, qubits, layers, reservoir, target=cuda_target
        )
        cpu_result = cpu_future.result()
    return gpu_result + cpu_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--qrc-layers", type=int, default=2)
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--gpu-percent", type=float, default=0.0)
    args = parser.parse_args()

    if args.rows < 1 or args.cols < 1 or args.qrc_layers < 1:
        print("rows, cols and qrc-layers must be > 0")
        return 1
    if not 0.0 <= args.gpu_percent <= 100.0:
        print("gpu-percent must be between 0 and 100")
        return 1

    qubits = args.cols
    rng = np.random.default_rng(QRC_SEED)
    data = rng.uniform(-1.0, 1.0, size=(args.rows, qubits))
    reservoir = build_reservoir_params(qubits, args.qrc_layers, QRC_SEED)

    print(f"Dataset: {args.rows}x{qubits}, qrc-layers={args.qrc_layers}")
    print("\nRuntime by joblib job count:")
    for n_jobs in args.jobs:
        n = max(1, min(n_jobs, args.rows))
        t0 = time.perf_counter()
        run_hybrid(
            data, qubits, args.qrc_layers, reservoir, n, args.gpu_percent
        )
        print(f"{n_jobs}\t{time.perf_counter() - t0:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
