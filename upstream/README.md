# Risk-Aware Utility Re-Ranking for Financial Asset Recommendation

This repository contains the **official implementation** of RURA submitted to **WSDM 2026**.

The results from the paper can be easily reproduced by following the steps below.

---

## 1. Download the Data
Download the data from the provided link and place it in the **top-level directory** of this repository.
```
https://drive.google.com/drive/folders/1jl9YmSnbkQoSaZ2rMWKILBfHjOCnQtTA?usp=sharing
```
---

## 2. Build the Docker Image
```bash
sudo DOCKER_BUILDKIT=1 docker build -f Dockerfile.gpu -t rura:gpu .
```

## 3. Run RURA
```bash
sudo docker run --rm -it --gpus all \
  -v "$PWD":/workspace:ro \
  -v "$PWD/features":/data/features:ro \
  -v "$PWD/candidates":/data/candidates:ro \
  -v "$PWD/profiles_riskmap_cat":/data/profiles:ro \
  -v "$PWD/splits":/data/splits:ro \
  -v "$PWD/runs_rura_full_riskmap_050":/app/runs \
  -v "$PWD/logs":/app/logs \
  --workdir /workspace \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PYTHONHASHSEED=2025 -e PYTHONPATH=/workspace \
  rura:gpu \
  python3 /workspace/run_rura.py \
    --features_dir /data/features \
    --candidates_dir /data/candidates \
    --profiles_dir /data/profiles \
    --slices_file /data/splits/slices.jsonl \
    --out_dir /app/runs \
    --method rura_la \
    --use_cov false \
    --L 30 \
    --lambda_div 0.50 \
    --device auto \
    --num_workers 16 \
    --chunk_rows 200000 \
    --seed 2025 \
    --progress_estimate_total true \
    --progress_every_users 1 \
    --progress_time_sec 2 \
    --on_user_error raise
```

## 4. Evaluate RURA with nDCG and ROI
```bash
sudo docker run --rm -it --gpus all \
  -v "$PWD":/workspace:ro \
  -v "$PWD/features":/data/features:ro \
  -v "$PWD/candidates":/data/candidates:ro \
  -v "$PWD/profiles_riskmap_cat":/data/profiles:ro \
  -v "$PWD/clean":/data/clean:ro \
  -v "$PWD/splits":/data/splits:ro \
  -v "$PWD/runs_rura_full_riskmap_050":/data/runs:ro \
  -v "$PWD/metrics_rura_full_riskmap_050":/app/metrics \
  -v "$PWD/logs":/app/logs \
  --workdir /workspace \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PYTHONHASHSEED=2025 -e PYTHONPATH=/workspace \
  rura:gpu \
  bash -lc '\
    set -euo pipefail; TIMEFORMAT="[time] metrics %3lR"; \
    time python3 /workspace/make_metrics_fast.py \
      --runs_dir /data/runs \
      --features_dir /data/features \
      --candidates_dir /data/candidates \
      --profiles_dir /data/profiles \
      --slices_file /data/splits/slices.jsonl \
      --clean_dir /data/clean \
      --out_dir /app/metrics \
      --methods rura_la \
      --eval_mode realized_ts \
      --cvar_mode normal \
      --alpha_cvar 0.95 \
      --overall_weight users \
      --fees_bps_buy 10 --fees_bps_sell 10 --slippage_bps 10 \
      --device auto \
      --num_workers 16 \
      --chunk_rows 200000 \
      --seed 2025 \
      --progress_estimate_total true \
      --progress_every_users 1 \
      --progress_time_sec 2
  '
```

## 5. Evaluate RURA Utility
```bash
sudo docker run --rm -it \
  -v "$PWD":/workspace:ro \
  -v "$PWD/runs_rura_full_riskmap_050":/data/runs:ro \
  -v "$PWD/profiles_riskmap_cat":/data/profiles:ro \
  -v "$PWD/splits":/data/splits:ro \
  -v "$PWD/candidates":/data/candidates:ro \
  -v "$PWD/utility_rura_full_riskmap_050":/app/utility \
  -v "$PWD/logs":/app/logs \
  --workdir /workspace \
  rura:gpu \
  bash -lc '\
    set -euo pipefail; TIMEFORMAT="[time] utility %3lR"; \
    time python3 /workspace/tools/compute_utility_fast.py \
      --runs_dir /data/runs \
      --profiles_dir /data/profiles \
      --slices_file /data/splits/slices.jsonl \
      --out_dir /app/utility \
      --user_category_csv /workspace/configs/user_categories.csv \
      --rho_scale_map "Conservative=1.20,Income=1.10,Balanced=1.00,Aggressive=0.85" \
      --candidates_dir /data/candidates \
      --features_dir "" \
      --mu_col_candidates "score_blend,score_tx,score_px,p_i,score,pred" \
      --log /app/logs/utility_all_rura.log \
      --progress_every_users 500 --progress_time_sec 20
  '
```
