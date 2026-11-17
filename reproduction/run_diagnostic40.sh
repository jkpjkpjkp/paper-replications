#!/usr/bin/env bash
set -euo pipefail
sit_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sit_python="$sit_root/reproduction/.venv/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 MPLBACKEND=Agg
test -f "$sit_root/reproduction/fresh_local/pool50_finished.txt"
common=(--is_training 1 --root_path "$sit_root/asset_data/" --data_path full_dataset.csv
        --precomp_root "$sit_root/signature_cache_6020" --model Signature --data FULL
        --window_size 60 --horizon 20 --d_model 8 --num_layers 2 --sig_input_dim 2
        --cross_sig_dim 1 --des Exp --num_workers 10 --batch_size 64
        --model_id dp40 --data_pool 40 --n_heads 4 --hidden_c 8 --ff_dim 32)
for sit_case in repeat_released longer_patience_first_run tf32_released; do
    sit_output="$sit_root/reproduction/discrepancy40/$sit_case"
    mkdir -p "$sit_output"
    test ! -f "$sit_output/started.txt"
    cd "$sit_output"
    sit_entry="$sit_root/run.py"
    case "$sit_case" in
        repeat_released) extra=(--itr 3 --train_epochs 30 --patience 3);;
        longer_patience_first_run) extra=(--itr 1 --train_epochs 100 --patience 10);;
        tf32_released) extra=(--itr 3 --train_epochs 30 --patience 3); sit_entry="$sit_root/reproduction/launch_tf32.py";;
    esac
    date -Is > started.txt
    printf '%q ' "$sit_python" "$sit_entry" "${common[@]}" "${extra[@]}" > command.txt
    printf '\n' >> command.txt
    PYTHONUNBUFFERED=1 "$sit_python" "$sit_entry" "${common[@]}" "${extra[@]}" > training.log 2>&1
    date -Is > finished.txt
done
