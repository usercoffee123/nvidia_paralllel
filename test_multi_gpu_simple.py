#!/usr/bin/env python3
"""Simpler test: small kernel across all 8 QPUs."""
import cudaq

print("Setting target to 'nvidia' with option='mqpu'...")
cudaq.set_target("nvidia", option="mqpu")

n_qpus = cudaq.get_target().num_qpus()
print(f"QPU count: {n_qpus}\n")

# Use observe_async like your worker does
@cudaq.kernel
def simple_kernel(theta: float):
    """Simpler kernel: 4 qubits, 1 layer."""
    q = cudaq.qvector(4)
    h(q)
    for i in range(4):
        rz(theta, q[i])

futures = []

print("Submitting observe_async work to all 8 QPUs...")
for gpu in range(8):
    print(f"  Submitting qpu_id={gpu}")
    # Try observe_async like your worker does
    f = cudaq.observe_async(
        simple_kernel,
        cudaq.spin.z(0),
        0.123 + gpu * 0.01,
        qpu_id=gpu,
        shots_count=0,
    )
    futures.append(f)

print("\nAll 8 jobs submitted.")
print("Collecting results...\n")

for i, f in enumerate(futures):
    try:
        result = f.get()
        print(f"✓ qpu_id={i}: {result.expectation(cudaq.spin.z(0)):.6f}")
    except Exception as e:
        print(f"✗ qpu_id={i}: ERROR: {e}")

print("\nDone!")
