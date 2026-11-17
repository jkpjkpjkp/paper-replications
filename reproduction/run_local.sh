#!/usr/bin/env bash
set -euo pipefail
sit_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sit_pool=${1:?Specify 40 or 50}
sit_python="$sit_root/reproduction/.venv/bin/python"
sit_memory=${2:-native}
sit_launcher=("$sit_python")
if [[ "$sit_memory" == managed ]]; then
    sit_launcher+=("$sit_root/reproduction/launch_managed.py")
elif [[ "$sit_memory" != native ]]; then
    exit 2
fi
sit_output="$sit_root/reproduction/fresh_local"
mkdir -p "$sit_output" "$sit_root/reproduction/logs"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 MPLBACKEND=Agg
cd "$sit_output"
common=(--is_training 1 --root_path "$sit_root/asset_data/" --data_path full_dataset.csv
        --precomp_root "$sit_root/signature_cache_6020" --model Signature --data FULL
        --window_size 60 --horizon 20 --d_model 8 --num_layers 2 --sig_input_dim 2
        --cross_sig_dim 1 --des Exp --itr 3 --num_workers 10 --train_epochs 30
        --patience 3 --batch_size 64 --model_id "dp$sit_pool" --data_pool "$sit_pool")
case "$sit_pool" in
    40) extra=(--n_heads 4 --hidden_c 8 --ff_dim 32);;
    50) extra=(--n_heads 8 --hidden_c 64 --ff_dim 8);;
    *) exit 2;;
esac
date -Is > "pool${sit_pool}_started.txt"
printf '%q ' "${sit_launcher[@]}" "$sit_root/run.py" "${common[@]}" "${extra[@]}" > "pool${sit_pool}_command.txt"
printf '\n' >> "pool${sit_pool}_command.txt"
PYTHONUNBUFFERED=1 "${sit_launcher[@]}" "$sit_root/run.py" "${common[@]}" "${extra[@]}"
date -Is > "pool${sit_pool}_finished.txt"
