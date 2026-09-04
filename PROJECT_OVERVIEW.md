# Project Overview

The benchmark generates a seeded row-major dataset and evaluates every row with a CUDA-Q quantum reservoir kernel. The NVIDIA `mqpu` target exposes one logical QPU per available GPU. Rows are submitted asynchronously with round-robin `qpu_id` values, so the same code handles one GPU or a multi-GPU system without process-level workers.

## Data Flow

1. Generate `rows x cols` inputs with NumPy and seed `42`.
2. Build deterministic reservoir rotation parameters.
3. Build one parameterized CUDA-Q kernel.
4. Select CUDA-Q's NVIDIA `mqpu` target and query `num_qpus()`.
5. Submit rows with `qpu_id=row_index % num_qpus()`.
6. Collect futures in submission order and return per-qubit Z expectations.

The CPU test fixture uses CUDA-Q's `qpp-cpu` target; production runs use the NVIDIA `mqpu` target.

## Running

```bash
python python/worker.py --rows 3000 --cols 6 --qrc-layers 2
```

```bash
pytest
```
