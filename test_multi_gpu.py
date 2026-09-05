#!/usr/bin/env python3
"""Minimal test: submit async work to all 8 QPUs and monitor allocation."""
import cudaq
import time

print("Setting target to 'nvidia' with option='mqpu'...")
cudaq.set_target("nvidia", option="mqpu")

n_qpus = cudaq.get_target().num_qpus()
print(f"QPU count: {n_qpus}")

if n_qpus != 8:
    raise RuntimeError(f"Expected 8 QPUs, got {n_qpus}")

@cudaq.kernel
def workload(theta: float):
    """28-qubit workload to trigger ~2 GiB allocation per GPU."""
    q = cudaq.qvector(28)

    h(q)

    # Heavy work across 100 layers
    for layer in range(100):
        for i in range(28):
            rz(theta, q[i])
            rx(0.37, q[i])

        for i in range(27):
            x.ctrl(q[i], q[i + 1])

    mz(q)

futures = []

print("\nSubmitting async work to all 8 QPUs...")
for gpu in range(8):
    print(f"  Submitting work to qpu_id={gpu}")
    f = cudaq.sample_async(
        workload,
        0.123 + gpu * 0.01,
        shots_count=1000,
        qpu_id=gpu,
    )
    futures.append(f)

print("\nAll 8 jobs submitted.")
print("Watch nvidia-smi now (use: watch -n 0.2 nvidia-smi)")
print("You should see ~2 GiB allocated and >0% utilization on all 8 GPUs.")
print("\nCollecting results...")

for i, f in enumerate(futures):
    print(f"  Waiting for qpu_id={i}...")
    result = f.get()
    print(f"  ✓ QPU {i} done")

print("\nAll done!")
