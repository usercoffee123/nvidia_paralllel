#!/bin/bash
set -euo pipefail

# Usage: ./run.sh [rows] [qubits] [layers] [devices] [output.csv]
#   devices: comma-separated GPU ids, e.g. 0,1,2,3 (one worker per device)
ROWS=${1:-40}
QUBITS=${2:-21}
LAYERS=${3:-40}
DEVICES=${4:-${CUDA_VISIBLE_DEVICES:-0}}
OUTPUT=${5:-results.csv}
PY=/root/nvidia_paralllel/ve/bin/python
COORDINATOR=/root/nvidia_paralllel/python/coordinator_zmq.py

# Count number of GPUs
NUM_GPUS=$(echo "${DEVICES}" | tr ',' '\n' | wc -l)

echo "=== Dataset: ${ROWS} rows x ${QUBITS} qubits, ${LAYERS} layers ==="
echo "=== Devices: ${DEVICES} (${NUM_GPUS} GPUs, one worker each) ==="
echo "=== Output CSV: ${OUTPUT} ==="
CUDA_VISIBLE_DEVICES="${DEVICES}" \
	"${PY}" -u "${COORDINATOR}" \
	--rows "${ROWS}" --qubits "${QUBITS}" --layers "${LAYERS}" --gpus "${NUM_GPUS}" \
	--output "${OUTPUT}"
