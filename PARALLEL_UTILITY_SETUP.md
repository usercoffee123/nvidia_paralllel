# CUDA-Q setup

This project uses CUDA-Q for all circuit execution. Install the runtime and development dependencies into the bundled environment:

```bash
source ve/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The NVIDIA target must see the system GPUs. CUDA-Q's `mqpu` target exposes the available GPU count, and the worker assigns submitted rows round-robin across those logical QPUs. A single-GPU system therefore uses one QPU automatically.

Run the benchmark with:

```bash
python python/worker.py --rows 3000 --cols 6 --qrc-layers 2
```
