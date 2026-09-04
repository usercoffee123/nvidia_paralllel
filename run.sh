ROWS=${1:-50}
COLS=${2:-32}
JOBS=${3:-1}
PY=/root/nvidia_paralllel/ve/bin/python
WORKER=/root/nvidia_paralllel/python/worker.py

echo "=== Dataset: ${ROWS} rows x ${COLS} cols ==="
echo "=== CUDA-Q NVIDIA GPU run ==="
$PY -u $WORKER --rows $ROWS --cols $COLS --qrc-layers 8 
