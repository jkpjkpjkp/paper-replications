#!/usr/bin/env bash
# Prepare before renting GPU time. Use the image's preinstalled CUDA torch.
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 MPLBACKEND=Agg
python -m pip install numpy==1.26.4 pandas==2.2.2 scipy==1.14.1 scikit-learn==1.5.2 matplotlib==3.10.0 joblib==1.4.2 tqdm==4.67.1
python -u reproduction/prepare_cache.py --pools 40 50 > reproduction/cache_build_remote.log 2>&1
python -m pip freeze > reproduction/cpu_environment_freeze.txt
sha256sum asset_data/full_dataset.csv > reproduction/remote_data.sha256
