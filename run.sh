set -euo pipefail

ROWS=${1:-500}
QUBITS=${2:-28}
LAYERS=${3:-80}
GPU_LIST=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
PY=/root/nvidia_paralllel/ve/bin/python
COORDINATOR=/root/nvidia_paralllel/python/coordinator_zmq.py

# Count number of GPUs
NUM_GPUS=$(echo "${GPU_LIST}" | tr ',' '\n' | wc -l)

echo "=== Dataset: ${ROWS} rows x ${QUBITS} qubits, ${LAYERS} layers ==="
echo "=== Visible GPUs: ${GPU_LIST} (${NUM_GPUS} total) ==="
echo "=== ZMQ Multi-GPU run (independent processes per GPU) ==="
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
	"${PY}" -u "${COORDINATOR}" \
	--rows "${ROWS}" --qubits "${QUBITS}" --layers "${LAYERS}" --gpus "${NUM_GPUS}"
