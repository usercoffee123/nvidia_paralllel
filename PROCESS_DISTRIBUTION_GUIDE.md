# Process Distribution Guide

## Overview

The updated coordinator now supports specifying the number of worker **processes** separately from the number of **GPUs**, allowing you to distribute processes evenly across available GPUs.

## How It Works

1. **Specify processes and GPUs** via CLI arguments
2. **Processes are distributed evenly** across GPUs using round-robin assignment
3. **Each process gets a unique identifier** (proc_id) for ZMQ communication
4. **Rows are divided equally** among all processes (not just GPUs)
5. **Each process runs on its assigned GPU** via CUDA_VISIBLE_DEVICES

## Usage Examples

### Example 1: 4 processes, 4 GPUs (1 process per GPU)
```bash
python coordinator_zmq.py --rows 1000 --processes 4 --gpus 4
```
- Process 0 → GPU 0
- Process 1 → GPU 1
- Process 2 → GPU 2
- Process 3 → GPU 3
- Result: Each GPU gets 1 process, 250 rows each

### Example 2: 10 processes, 2 GPUs (5 processes per GPU)
```bash
python coordinator_zmq.py --rows 1000 --processes 10 --gpus 2
```
- Processes 0, 2, 4, 6, 8 → GPU 0
- Processes 1, 3, 5, 7, 9 → GPU 1
- Result: Each GPU gets 5 processes, 100 rows each

### Example 3: 10 processes, 1 GPU (all on same GPU)
```bash
python coordinator_zmq.py --rows 1000 --processes 10 --gpus 1
```
- Processes 0-9 → GPU 0
- Result: 1 GPU gets 10 processes hammering it with 100 rows each

### Example 4: Default behavior (processes = GPUs)
```bash
python coordinator_zmq.py --rows 1000
```
- If you have 4 GPUs available: 4 processes created, 1 per GPU
- If you have 8 GPUs available: 8 processes created, 1 per GPU

### Example 5: Custom parameters
```bash
python coordinator_zmq.py \
    --rows 5000 \
    --qubits 28 \
    --layers 8 \
    --processes 16 \
    --gpus 4
```
- 16 processes across 4 GPUs = 4 processes per GPU
- 5000 rows divided into 16 chunks = ~312 rows per process

## CLI Arguments

```
--rows N              Number of input rows (default: 80)
--qubits N            Number of qubits (default: 28)
--layers N            Number of QRC layers (default: 8)
--gpus N              Number of GPUs to use (default: all available)
--processes N         Number of worker processes (default: number of GPUs)
```

## Load Distribution Formula

For **P processes** and **G GPUs**:

1. **Assignment**: Process i → GPU (i mod G)
2. **Load per GPU**: ceil(P / G) processes per GPU
3. **Rows per process**: floor(total_rows / P) rows per process

### Examples:
- 4 processes, 4 GPUs: 1 process per GPU
- 5 processes, 4 GPUs: GPU 0 & 1 get 2 processes each, GPU 2 & 3 get 1 each
- 10 processes, 2 GPUs: 5 processes per GPU
- 10 processes, 1 GPU: 10 processes on the same GPU

## Worker Process Information

Each worker logs:
- `[Proc N, GPU M]` - Process ID and assigned GPU
- Initialization confirmation
- Task processing status
- Results sent back to coordinator

Example output:
```
[Proc 0, GPU 0] Initialized - CUDA_VISIBLE_DEVICES sees 1 GPU(s)
[Proc 0, GPU 0] Connected to coordinator at tcp://127.0.0.1:12345
[Proc 0, GPU 0] Processing task 0 with 312 rows
[Proc 0, GPU 0] Task 0 done - sent 8736 results
```

## Implementation Details

### coordinator_zmq.py
- Parses `--processes` and `--gpus` arguments
- Calculates process-to-GPU mapping (round-robin)
- Spawns worker processes with unique proc_id
- Divides rows into process-count chunks
- Uses worker identity `worker-proc{N}` for ZMQ routing

### worker_zmq.py
- Accepts `--proc-id` argument
- Uses ZMQ DEALER identity `worker-proc{N}`
- Receives rows assigned to that process
- Runs on GPU specified by CUDA_VISIBLE_DEVICES
- Returns results with task_id, gpu_id, and proc_id

## Notes

- Multiple processes **can** share the same GPU for parallel quantum workload
- Each process still gets its own CUDA context (via CUDA_VISIBLE_DEVICES)
- Rows are always divided **equally** among all processes
- ZMQ ROUTER/DEALER pattern ensures independent message streams per process
