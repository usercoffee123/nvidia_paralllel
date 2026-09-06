#!/bin/bash
set -euo pipefail

ROWS=${1:-40}
QUBITS=${2:-24}
LAYERS=${3:-10}
OUTPUT=${4:-results.csv}
GPU_LIST=${CUDA_VISIBLE_DEVICES:-0}
PY=/root/nvidia_paralllel/ve/bin/python
COORDINATOR=/root/nvidia_paralllel/python/coordinator_zmq.py

# Count number of GPUs
NUM_GPUS=$(echo "${GPU_LIST}" | tr ',' '\n' | wc -l)

echo "=== Dataset: ${ROWS} rows x ${QUBITS} qubits, ${LAYERS} layers ==="
echo "=== Visible GPUs: ${GPU_LIST} (${NUM_GPUS} total) ==="
echo "=== Output CSV: ${OUTPUT} ==="
echo "=== ZMQ Multi-GPU run (independent processes per GPU) ==="
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
	"${PY}" -u "${COORDINATOR}" \
	--rows "${ROWS}" --qubits "${QUBITS}" --layers "${LAYERS}" --gpus "${NUM_GPUS}" \
	--output "${OUTPUT}"
