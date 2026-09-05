# GPU Row Distribution

The worker uses CUDA-Q's `mqpu` target for independent circuit evaluations:

```python
cudaq.set_target("nvidia", option="mqpu")
num_gpus = cudaq.num_available_gpus()
```

`mqpu` creates one logical QPU for each visible NVIDIA GPU. Each row is submitted asynchronously with an explicit QPU ID, and all queued futures are submitted before results are collected.

```python
qpu_id = min(row_index * num_gpus // row_count, num_gpus - 1)
future = cudaq.observe_async(
    kernel,
    observable,
    *parameters,
    qpu_id=qpu_id,
    shots_count=0,
)
```

For eight GPUs, eight rows are assigned one per GPU. For larger datasets, contiguous balanced batches are assigned to each GPU. Results are collected with `future.get()` in submission order, so output rows remain ordered even though execution is asynchronous.

## Target Modes

- `mqpu`: many independent circuits run concurrently on separate GPUs. This is the mode used by this project.
- `mgpu`: one large statevector simulation uses memory and computation from multiple GPUs. This is a different workload and is not used here.

## Verification

Run a workload with all GPUs visible:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /root/nvidia_paralllel/run.sh 8 28
```

CUDA-Q should report `8 rows across 8 GPU(s)`. Monitor utilization while the process is active:

```bash
watch -n 0.5 nvidia-smi
```

Memory usage may be higher on GPU 0 because CUDA-Q keeps additional runtime and compilation workspace there. GPU utilization and comparable single-GPU timings are better indicators of distribution than a single memory snapshot.
