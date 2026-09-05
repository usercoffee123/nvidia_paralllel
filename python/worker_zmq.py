#!/usr/bin/env python3
"""ZMQ worker: single GPU, independent process."""
import argparse
import os
import sys

# =====================================================================
# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing cudaq
# This ensures the process only sees and uses the assigned GPU
# =====================================================================
def _parse_args_early():
    """Parse args to get gpu_id, then set env."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--router-url", type=str, required=True)
    parser.add_argument("--qubits", type=int, default=28)
    parser.add_argument("--layers", type=int, default=8)
    args, _ = parser.parse_known_args()
    return args

early_args = _parse_args_early()
os.environ["CUDA_VISIBLE_DEVICES"] = str(early_args.gpu_id)

# Single-threaded CPU mode
for var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS", "NUMBA_NUM_THREADS", "TBB_NUM_THREADS",
):
    os.environ[var] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

# Now safe to import cudaq (it will see only the assigned GPU)
import cudaq
import json
import zmq
import numpy as np

from worker import (
    _build_cudaq_kernel, build_reservoir_params, process_batch_cudaq,
    QRC_SEED,
)


def worker_main(gpu_id: int, router_url: str, qubits: int, layers: int):
    """
    Single-GPU worker process using DEALER socket.
    
    1. Connect to coordinator's ROUTER socket via DEALER (identity-based addressing)
    2. Receive row chunks
    3. Run them on assigned GPU (isolated via CUDA_VISIBLE_DEVICES)
    4. Send results back
    """
    # Verify GPU isolation
    num_gpus = cudaq.num_available_gpus()
    print(f"[GPU {gpu_id}] Initialized - CUDA_VISIBLE_DEVICES sees {num_gpus} GPU(s)", flush=True)
    if num_gpus != 1:
        print(f"[GPU {gpu_id}] WARNING: Expected 1 GPU but see {num_gpus}!", flush=True)
    
    context = zmq.Context()
    
    # DEALER: request-reply with coordinator (each worker gets own message stream)
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.RCVTIMEO, 60000)  # 60 second timeout on recv
    socket.setsockopt_string(zmq.IDENTITY, f"worker-gpu{gpu_id}")
    socket.connect(router_url)
    
    print(f"[GPU {gpu_id}] Connected to coordinator at {router_url}", flush=True)
    
    # Set target to single GPU (not mqpu - isolated use of one GPU)
    cudaq.set_target("nvidia")
    target = cudaq.get_target()
    print(f"[GPU {gpu_id}] Target: {target.name}, QPUs: {target.num_qpus()}", flush=True)
    
    tasks_completed = 0
    
    # Process work until no more messages (timeout after 60s of inactivity)
    while True:
        try:
            # Receive work from coordinator
            # DEALER returns [empty_delimiter, message_data]
            _, message_data = socket.recv_multipart()
            message = json.loads(message_data.decode())
            
            rows = np.array(message["rows"], dtype=np.float64)
            task_id = message.get("task_id", 0)
            
            print(f"[GPU {gpu_id}] Processing task {task_id} with {len(rows)} rows", flush=True)
            
            # Run on this GPU using CUDA-Q (single GPU mode)
            results = process_batch_cudaq(
                rows, qubits, layers, build_reservoir_params(qubits, layers, QRC_SEED),
                target="nvidia", progress=False
            )
            
            # Send results back via DEALER
            response = {
                "task_id": task_id,
                "gpu_id": gpu_id,
                "results": results,  # List of floats
            }
            
            socket.send_multipart([b"", json.dumps(response).encode()])
            tasks_completed += 1
            print(f"[GPU {gpu_id}] Task {task_id} done - sent {len(results)} results", flush=True)
            
        except zmq.error.Again:
            # Timeout - no more work
            print(f"[GPU {gpu_id}] Receive timeout - no more work", flush=True)
            break
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {type(e).__name__}: {e}", flush=True)
            break
    
    print(f"[GPU {gpu_id}] Completed {tasks_completed} tasks", flush=True)
    
    socket.close()
    context.term()
    print(f"[GPU {gpu_id}] Worker shutdown", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--router-url", type=str, required=True)
    parser.add_argument("--qubits", type=int, default=28)
    parser.add_argument("--layers", type=int, default=8)
    args = parser.parse_args()
    
    worker_main(args.gpu_id, args.router_url, args.qubits, args.layers)
