#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 MPLBACKEND=Agg
mkdir -p reproduction/logs
python -m pip freeze > reproduction/environment_freeze.txt
nvidia-smi > reproduction/gpu_environment.txt
date -Is > reproduction/gpu_started.txt
CUDA_VISIBLE_DEVICES=1 python -u reproduction/benchmark.py --device cuda --pool 50 --batch 64 > reproduction/logs/gpu_benchmark_50.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u reproduction/benchmark.py --device cuda --pool 40 --batch 64 > reproduction/logs/gpu_benchmark_40.log 2>&1
common=(--is_training 1 --root_path ./asset_data/ --data_path full_dataset.csv --model Signature --data FULL --window_size 60 --horizon 20 --d_model 8 --num_layers 2 --sig_input_dim 2 --cross_sig_dim 1 --des Exp --itr 3 --num_workers 10 --train_epochs 30 --patience 3 --batch_size 64 --gpu 0)
CUDA_VISIBLE_DEVICES=0 python -u reproduction/launch_isolated.py "${common[@]}" --model_id dp40 --data_pool 40 --n_heads 4 --hidden_c 8 --ff_dim 32 > reproduction/logs/fresh_dp40.log 2>&1 &
pid40=$!
CUDA_VISIBLE_DEVICES=1 python -u reproduction/launch_isolated.py "${common[@]}" --model_id dp50 --data_pool 50 --n_heads 8 --hidden_c 64 --ff_dim 8 > reproduction/logs/fresh_dp50.log 2>&1 &
pid50=$!
trap 'kill -TERM "$pid40" "$pid50" 2>/dev/null || true' INT TERM
exit40=0; exit50=0
wait "$pid40" || exit40=$?
wait "$pid50" || exit50=$?
printf '{"pool40_exit": %s, "pool50_exit": %s}\n' "$exit40" "$exit50" > reproduction/run_exit_codes.json
date -Is > reproduction/gpu_finished.txt
tar -czf sit_fresh_results.tar.gz results checkpoints reproduction
test "$exit40" -eq 0 && test "$exit50" -eq 0
