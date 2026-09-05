#!/usr/bin/env python3
"""CUDA-Q quantum reservoir computing benchmark."""
import argparse
from collections import deque
import math
import os
import subprocess
import sys
import time

for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS", "NUMBA_NUM_THREADS", "TBB_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import cudaq
import numpy as np

QRC_SEED = 42
NVIDIA_TARGET = "nvidia"
MIN_QUEUE_DEPTH = 2000
QUEUE_DEPTH_PER_GPU = 32


def build_reservoir_params(qubits: int, layers: int, seed: int):
    """Generate deterministic pseudo-random rotation angles."""
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


def compute_z_expectations(statevector, qubits: int):
    """Compute exact per-qubit Z expectations from a statevector."""
    probabilities = np.abs(np.asarray(statevector)) ** 2
    expectations = []
    for q in range(qubits):
        physical_qubit = qubits - 1 - q
        expectation = sum(
            probability * (1.0 - 2.0 * ((basis >> physical_qubit) & 1))
            for basis, probability in enumerate(probabilities)
        )
        expectations.append(float(expectation.real))
    return expectations


def _build_cudaq_kernel(qubits: int, layers: int, reservoir):
    kernel, *feature_angles = cudaq.make_kernel(*([float] * qubits))
    qubit_register = kernel.qalloc(qubits)
    for q in range(qubits):
        theta = feature_angles[q]
        kernel.ry(theta, qubit_register[q])
        kernel.rz(0.5 * theta, qubit_register[q])
    if qubits > 1:
        for q in range(qubits):
            kernel.cz(qubit_register[q], qubit_register[(q + 1) % qubits])
    for layer in range(layers):
        for q in range(qubits):
            base_rx, base_rz = reservoir[layer][q]
            injection = 0.35 * feature_angles[(q + layer) % qubits]
            kernel.rx(base_rx + injection, qubit_register[q])
            kernel.rz(base_rz - 0.25 * injection, qubit_register[q])
        if qubits > 1:
            for q in range(qubits):
                kernel.cz(qubit_register[q], qubit_register[(q + 1) % qubits])
    return kernel


def get_gpu_memory_mb(device=0):
    """Return free and total GPU memory from nvidia-smi, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device}", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        free_mb, total_mb = (int(value.strip()) for value in result.stdout.split(","))
        return free_mb, total_mb
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _configure_target(target: str) -> int:
    """Select a CUDA-Q target and return its available QPU count."""
    if target == NVIDIA_TARGET:
        cudaq.set_target(NVIDIA_TARGET, option="mqpu")
        num_gpus = cudaq.num_available_gpus()
        if num_gpus == 0:
            raise RuntimeError("CUDA-Q NVIDIA target is selected, but no GPU is available")
        return num_gpus

    cudaq.set_target(target)
    return 1


def _qpu_for_row(row_index: int, row_count: int, num_gpus: int) -> int:
    """Assign contiguous, balanced row batches to the available QPUs."""
    return min(row_index * num_gpus // row_count, num_gpus - 1)


def check_gpu_memory(qubits: int, num_gpus: int):
    """Check one statevector per GPU fits with 20% free-memory headroom."""
    if num_gpus < 1:
        return None
    memory = get_gpu_memory_mb()
    if memory is None:
        return None
    free_mb, total_mb = memory
    required_mb = (2**qubits * 8) / (1024**2)
    usable_mb = free_mb * 0.8 / num_gpus
    if required_mb > usable_mb:
        raise RuntimeError(
            f"GPU memory is insufficient for {qubits} qubits: need about {required_mb:.0f} MiB per GPU, "
            f"have {free_mb:.0f} MiB free of {total_mb:.0f} MiB ({num_gpus} GPU(s))"
        )
    return free_mb, total_mb, required_mb


def process_batch_cudaq(chunk, qubits: int, layers: int, reservoir,
                        target="nvidia", progress=False):
    """Run a row batch through CUDA-Q and return flat expectation values."""
    num_gpus = _configure_target(target)

    memory = check_gpu_memory(qubits, num_gpus) if target == NVIDIA_TARGET else None
    if progress:
        memory_text = ""
        if memory is not None:
            free_mb, total_mb, required_mb = memory
            memory_text = f"; {free_mb:.0f}/{total_mb:.0f} MiB free, ~{required_mb:.0f} MiB statevector"
        print(f"CUDA-Q: {len(chunk)} rows across {num_gpus} GPU(s){memory_text}", flush=True)
    if not len(chunk):
        return []

    kernel = _build_cudaq_kernel(qubits, layers, reservoir)
    z_terms = [cudaq.spin.z(qubits - 1 - q) for q in range(qubits)]
    composite = z_terms[0]
    for term in z_terms[1:]:
        composite = composite + term

    angles = math.pi * np.tanh(np.asarray(chunk, dtype=float))
    pending = deque()
    out = []
    next_row = 0
    row_index = 0
    queue_depth = max(MIN_QUEUE_DEPTH, QUEUE_DEPTH_PER_GPU * num_gpus)

    def submit(row_index):
        qpu_id = _qpu_for_row(row_index, len(angles), num_gpus)
        pending.append(cudaq.observe_async(
            kernel, composite, *(float(value) for value in angles[row_index]),
            qpu_id=qpu_id, shots_count=0,
        ))

    while next_row < min(queue_depth, len(angles)):
        submit(next_row)
        next_row += 1
    while pending:
        result = pending.popleft().get()
        out.extend(float(result.expectation(term)) for term in z_terms)
        row_index += 1
        if progress:
            print(f"CUDA-Q: {row_index}/{len(angles)} rows completed", file=sys.stderr, flush=True)
        if next_row < len(angles):
            submit(next_row)
            next_row += 1
    return out


def process_batch(chunk, qubits: int, layers: int, reservoir):
    """CUDA-Q CPU-target compatibility helper for unit tests."""
    return process_batch_cudaq(chunk, qubits, layers, reservoir, target="qpp-cpu")


def run_reservoir(data, qubits: int, layers: int, reservoir, target="nvidia"):
    """Run all rows, assigning successive rows round-robin to available GPUs."""
    return process_batch_cudaq(data, qubits, layers, reservoir,
                               target=target, progress=target == "nvidia")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--qrc-layers", type=int, default=2)
    args = parser.parse_args()
    if args.rows < 1 or args.cols < 1 or args.qrc_layers < 1:
        print("rows, cols and qrc-layers must be > 0")
        return 1

    qubits = args.cols
    rng = np.random.default_rng(QRC_SEED)
    data = rng.uniform(-1.0, 1.0, size=(args.rows, qubits))
    reservoir = build_reservoir_params(qubits, args.qrc_layers, QRC_SEED)
    print(f"Dataset: {args.rows}x{qubits}, qrc-layers={args.qrc_layers}")
    t0 = time.perf_counter()
    run_reservoir(data, qubits, args.qrc_layers, reservoir)
    print(f"CUDA-Q runtime: {time.perf_counter() - t0:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
