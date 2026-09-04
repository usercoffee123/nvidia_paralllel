ROWS=${1:-10000}
COLS=${2:-17}
JOBS=${3:-128}
/root/nvidia_paralllel/ve/bin/python /root/nvidia_paralllel/python/worker.py --rows $ROWS --cols $COLS --qrc-layers 40 --jobs $JOBS
