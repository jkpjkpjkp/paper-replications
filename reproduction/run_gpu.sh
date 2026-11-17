#!/usr/bin/env bash
# Run from the extracted SIT directory, after CPU/data preparation.
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 MPLBACKEND=Agg
mkdir -p reproduction/logs
python -u reproduction/benchmark.py --device cuda --pool 50 --batch 64 > reproduction/logs/gpu_benchmark_50.log 2>&1
python -u reproduction/benchmark.py --device cuda --pool 40 --batch 64 > reproduction/logs/gpu_benchmark_40.log 2>&1
python -m pip freeze > reproduction/environment_freeze.txt
nvidia-smi > reproduction/gpu_environment.txt
date -Is > reproduction/gpu_started.txt
common=(--is_training 1 --root_path ./asset_data/ --data_path full_dataset.csv --model Signature --data FULL --window_size 60 --horizon 20 --d_model 8 --num_layers 2 --sig_input_dim 2 --cross_sig_dim 1 --des Exp --itr 3 --num_workers 10 --train_epochs 30 --patience 3 --batch_size 64)
python -u run.py "${common[@]}" --model_id dp40 --data_pool 40 --n_heads 4 --hidden_c 8 --ff_dim 32 > reproduction/logs/fresh_dp40.log 2>&1
python -u run.py "${common[@]}" --model_id dp50 --data_pool 50 --n_heads 8 --hidden_c 64 --ff_dim 8 > reproduction/logs/fresh_dp50.log 2>&1
date -Is > reproduction/gpu_finished.txt
tar -czf sit_fresh_results.tar.gz results checkpoints reproduction
