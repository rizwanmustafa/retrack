#!/usr/bin/env bash
# Installs the dependencies and runs the whole Retrack pipeline.
#   bash colab/run.sh <quick|full> <results_dir> [extra retrack_pipeline.py args]
set -euo pipefail
MODE=${1:-full}
RESULTS=${2:-/content/drive/MyDrive/retrack_results}
shift $(( $# >= 2 ? 2 : $# ))
REPO=$(cd "$(dirname "$0")/.." && pwd)

nvidia-smi --query-gpu=name,memory.total --format=csv || echo "WARNING: no GPU visible (Runtime > Change runtime type > A100)"
echo "installing dependencies..."
pip install -q -r "$REPO/requirements-colab.txt"
python -c "import torch, numpy, lightning, kornia; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| numpy', numpy.__version__, '| lightning', lightning.__version__)"

cd "$REPO/WorldTrack"
export PYTHONUNBUFFERED=1
mkdir -p "$RESULTS"
python retrack_pipeline.py --mode "$MODE" --results "$RESULTS" "$@" 2>&1 | tee -a "$RESULTS/pipeline_$MODE.log"
