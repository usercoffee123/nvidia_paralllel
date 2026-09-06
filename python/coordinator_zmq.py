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
    router_socket.setsockopt(zmq.LINGER, 0)  # never block on close
    router_socket.setsockopt(zmq.RCVTIMEO, 5000)  # poll in 5s slices so we can check worker liveness
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
    
    def fail_dead_workers():
        """Raise if any worker process exited; their messages would never arrive."""
        for g, p in worker_procs:
            if p.poll() is not None:
                out = p.stdout.read() if p.stdout else ""
                raise RuntimeError(
                    f"Worker GPU {g} died (exit {p.returncode}). Output:\n{out}")
    
    # Ready handshake: ROUTER silently drops messages to identities that have not
    # connected yet (slow-joiner race: cudaq import can take many seconds), so
    # never send work until every worker has announced itself.
    print("Waiting for ready handshake from all workers...", flush=True)
    ready = set()
    deadline = time.monotonic() + 120
    while len(ready) < num_gpus:
        fail_dead_workers()
        if time.monotonic() > deadline:
            raise RuntimeError(f"Only {len(ready)}/{num_gpus} workers ready after 120s: {ready}")
        try:
            frames = router_socket.recv_multipart()
        except zmq.error.Again:
            continue
        meta = json.loads(frames[-1].decode())
        if meta.get("status") == "ready":
            ready.add(frames[0].decode())
            print(f"Ready {len(ready)}/{num_gpus}: {frames[0].decode()}", flush=True)
    
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
    # Payload is binary float64 (JSON for 100k+ rows takes minutes to encode/parse)
    print("Sending work to all workers...", flush=True)
    for i, gpu_id in enumerate(visible_gpus):
        worker_id = f"worker-gpu{gpu_id}".encode()
        chunk = np.ascontiguousarray(chunks[i], dtype=np.float64)
        meta = {"task_id": i, "num_rows": chunk.shape[0], "num_cols": qubits}
        router_socket.send_multipart(
            [worker_id, b"", json.dumps(meta).encode(), chunk.tobytes()])
    
    # Collect results from workers via ROUTER
    print("Collecting results...", flush=True)
    results_by_chunk = [None] * num_gpus
    collected = 0
    
    while collected < num_gpus:
        fail_dead_workers()
        try:
            # ROUTER returns [identity, empty_delimiter, meta_json, results_bytes]
            worker_id, _, meta_data, results_data = router_socket.recv_multipart()
        except zmq.error.Again:
            continue
        meta = json.loads(meta_data.decode())
        task_id = meta["task_id"]
        results = np.frombuffer(results_data, dtype=np.float64)
        print(f"Got results from task {task_id}: {len(results)} values", flush=True)
        results_by_chunk[task_id] = results
        collected += 1
        # Poison pill: tell this worker to exit now instead of waiting for recv timeout
        router_socket.send_multipart([worker_id, b"", b'{"command": "kill"}'])
    
    # Clean up sockets
    router_socket.close()
    context.term()
    
    # Wait for workers to finish and collect their output
    print("Waiting for workers to finish...", flush=True)
    for gpu_id, proc in worker_procs:
        try:
            proc.wait(timeout=10)
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
        if chunk_results is not None and len(chunk_results):
            flat_results.extend(chunk_results.tolist())
    
    print(f"\nAll workers done. Returned {len(flat_results)} results.", flush=True)
    return flat_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--qubits", type=int, default=28)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Output CSV file path")
    args = parser.parse_args()
    
    # Dummy data for testing
    test_rows = np.random.randn(args.rows, args.qubits).tolist()
    
    results = run_distributed(test_rows, args.qubits, args.layers, args.gpus)
    print(f"Got {len(results)} results", flush=True)
    
    # Write results to CSV: one row per input row, one column per qubit expectation
    results_matrix = np.asarray(results, dtype=np.float64).reshape(-1, args.qubits)
    header = ",".join(f"z_q{q}" for q in range(args.qubits))
    np.savetxt(args.output, results_matrix, delimiter=",", header=header, comments="")
    print(f"Wrote {results_matrix.shape[0]} rows x {results_matrix.shape[1]} columns to {args.output}", flush=True)
