#!/usr/bin/env python3
"""Sanity check: verify CUDA-Q mqpu setup and environment."""
import os
import cudaq

print("=" * 60)
print("ENVIRONMENT VARIABLES")
print("=" * 60)
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDAQ_MQPU_NGPUS      =", os.environ.get("CUDAQ_MQPU_NGPUS"))

print("\n" + "=" * 60)
print("CUDAQ STATE")
print("=" * 60)
print("cudaq.num_available_gpus() =", cudaq.num_available_gpus())

print("\nSetting target to 'nvidia' with option='mqpu'...")
cudaq.set_target("nvidia", option="mqpu")

target = cudaq.get_target()
print("target.name =", target.name)
print("target.num_qpus() =", target.num_qpus())

print("\n" + "=" * 60)
if target.num_qpus() == 8:
    print("✓ PASS: target.num_qpus() = 8")
else:
    print(f"✗ FAIL: target.num_qpus() = {target.num_qpus()} (expected 8)")
print("=" * 60)
