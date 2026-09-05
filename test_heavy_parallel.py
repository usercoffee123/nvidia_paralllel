#!/usr/bin/env python3
"""Test: 8 heavy 28-qubit jobs simultaneously, one per GPU."""
import cudaq
import time

print("Setting target to nvidia + mqpu...")
cudaq.set_target("nvidia", option="mqpu")

num_gpus = cudaq.num_available_gpus()
target = cudaq.get_target()
num_qpus = target.num_qpus()

print(f"Available GPUs: {num_gpus}")
print(f"CUDA-Q QPUs:    {num_qpus}")

if num_qpus != 8:
    raise RuntimeError(f"Expected 8 QPUs, got {num_qpus}")

@cudaq.kernel
def heavy_28q(theta: float):
    """28-qubit circuit with 200 layers (should allocate ~2 GiB)."""
    q = cudaq.qvector(28)
    h(q)

    for layer in range(200):
        for i in range(28):
            rz(theta, q[i])
            rx(0.37, q[i])

        for i in range(27):
            x.ctrl(q[i], q[i + 1])

    mz(q)

print("\n" + "=" * 60)
print("Submitting 8 heavy jobs (one per GPU)...")
print("=" * 60)

futures = []
start_submit = time.time()

for gpu_id in range(8):
    print(f"  Submitting job to qpu_id={gpu_id}")
    f = cudaq.sample_async(
        heavy_28q,
        0.1 + gpu_id * 0.01,
        shots_count=1000,
        qpu_id=gpu_id,
    )
    futures.append((gpu_id, f))

submit_time = time.time() - start_submit
print(f"\nAll 8 jobs queued in {submit_time:.3f}s")
print("⚠️  Watch nvidia-smi in another terminal now!")
print("   Expected: all 8 GPUs should show ~2-3 GiB allocation + compute activity")
print("\nWaiting for results...\n")

start_compute = time.time()

for gpu_id, future in futures:
    result = future.get()
    elapsed = time.time() - start_compute
    print(f"✓ GPU {gpu_id} done (elapsed: {elapsed:.1f}s)")

total_time = time.time() - start_submit
print(f"\nTotal time: {total_time:.1f}s")
