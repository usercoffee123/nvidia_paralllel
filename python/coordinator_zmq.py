#!/usr/bin/env python3
"""ZMQ coordinator: spawns workers, distributes work, collects results."""
import argparse
import json
import os
import subprocess
import sys
import time
import zmq
import numpy as np


def get_visible_gpus():
    """Parse CUDA_VISIBLE_DEVICES or default to 0-7."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    return [int(g.strip()) for g in cuda_visible.split(",")]


def run_distributed(rows, qubits: int, layers: int, num_gpus: int = None):
    """
    Distribute rows across GPUs using ZMQ workers with ROUTER/DEALER pattern.
    Each worker gets its own GPU and runs independently.
    
    Returns: flat array of expectation values (same format as process_batch_cudaq)
    """
    if num_gpus is None:
        num_gpus = len(get_visible_gpus())
    
    visible_gpus = get_visible_gpus()[:num_gpus]
    
    print(f"Coordinator: Using {num_gpus} GPU(s): {visible_gpus}", flush=True)
    
    # ZMQ context
    context = zmq.Context()
    
    # ROUTER socket: receives connections from DEALER workers
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind("tcp://127.0.0.1:0")
    router_url = router_socket.getsockopt_string(zmq.LAST_ENDPOINT)
    
    print(f"Coordinator ROUTER: {router_url}", flush=True)
    
    # Spawn worker processes
    worker_procs = []
    for gpu_id in visible_gpus:
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "worker_zmq.py"),
            "--gpu-id", str(gpu_id),
            "--router-url", router_url,
            "--qubits", str(qubits),
            "--layers", str(layers),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        worker_procs.append((gpu_id, proc))
        print(f"Spawned worker for GPU {gpu_id} (PID {proc.pid})", flush=True)
    
    # Give workers time to connect to ROUTER
    time.sleep(2.0)
    
    # Divide rows into chunks (one per GPU)
    rows = np.asarray(rows, dtype=np.float64)
    chunk_size = (len(rows) + num_gpus - 1) // num_gpus
    chunks = []
    
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, len(rows))
        chunks.append(rows[start:end].tolist())
    
    print(f"Divided {len(rows)} rows into {num_gpus} chunks of ~{chunk_size} rows each", flush=True)
    
    # Send work to each worker via ROUTER
    # ROUTER addresses workers by identity: b"worker-gpu0", b"worker-gpu1", etc.
    print("Sending work to all workers...", flush=True)
    for i, gpu_id in enumerate(visible_gpus):
        worker_id = f"worker-gpu{gpu_id}".encode()
        message = {
            "task_id": i,
            "rows": chunks[i],
        }
        router_socket.send_multipart([worker_id, b"", json.dumps(message).encode()])
    
    # Collect results from workers via ROUTER
    print("Collecting results...", flush=True)
    results_by_chunk = [None] * num_gpus
    
    for _ in range(num_gpus):
        # ROUTER returns [identity, empty_delimiter, response_data]
        worker_id, _, response_data = router_socket.recv_multipart()
        response = json.loads(response_data.decode())
        task_id = response["task_id"]
        results = response["results"]
        print(f"Got results from task {task_id}: {len(results)} values", flush=True)
        results_by_chunk[task_id] = results
    
    # Clean up sockets
    router_socket.close()
    context.term()
    
    # Wait for workers to finish and collect their output
    print("Waiting for workers to finish...", flush=True)
    for gpu_id, proc in worker_procs:
        try:
            proc.wait(timeout=30)
            stdout_data = proc.stdout.read() if proc.stdout else ""
            for line in stdout_data.split("\n"):
                if line.strip():
                    print(f"  [GPU {gpu_id}] {line.strip()}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"Worker GPU {gpu_id} timed out, terminating...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    
    # Flatten results in order
    flat_results = []
    for chunk_results in results_by_chunk:
        if chunk_results:
            flat_results.extend(chunk_results)
    
    print(f"\nAll workers done. Returned {len(flat_results)} results.", flush=True)
    return flat_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--qubits", type=int, default=28)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    
    # Dummy data for testing
    test_rows = np.random.randn(args.rows, args.qubits).tolist()
    
    results = run_distributed(test_rows, args.qubits, args.layers, args.gpus)
    print(f"Got {len(results)} results", flush=True)
