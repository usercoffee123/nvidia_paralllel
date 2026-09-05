# Project Overview

The benchmark generates a seeded row-major dataset and evaluates every row with a CUDA-Q quantum reservoir kernel. The NVIDIA `mqpu` target exposes one logical QPU per available GPU. Rows are submitted asynchronously to balanced QPU batches, so the same code handles one GPU or a multi-GPU system without process-level workers.

## Data Flow

1. Generate `rows x cols` inputs with NumPy and seed `42`.
2. Build deterministic reservoir rotation parameters.
3. Build one parameterized CUDA-Q kernel.
4. Select CUDA-Q's NVIDIA `mqpu` target and query `num_available_gpus()`.
5. Assign contiguous balanced row batches with explicit `qpu_id` values.
6. Submit asynchronous futures, then collect them in row order and return per-qubit Z expectations.

The CPU test fixture uses CUDA-Q's `qpp-cpu` target; production runs use the NVIDIA `mqpu` target.

`mqpu` distributes independent circuits across GPUs. It should not be confused with `mgpu`, which distributes one large statevector across multiple GPUs.

## Running

```bash
python python/worker.py --rows 3000 --cols 6 --qrc-layers 2
```

```bash
pytest
```
