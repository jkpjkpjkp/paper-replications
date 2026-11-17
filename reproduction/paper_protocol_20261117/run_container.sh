#!/usr/bin/env bash
set -euo pipefail
sit_project=/home/jkp/data/SIT
sit_run="$sit_project/reproduction/paper_protocol_20261117"
sit_image=nvcr.io/nvidia/pytorch@sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb
sit_podman=(podman --root /mnt/data/home/jkp/sit-paper-runtime/containers
  --runroot /run/user/1000/sit-paper-containers --storage-driver overlay)
sit_devices=()
for sit_dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  sit_devices+=(--device "$sit_dev:$sit_dev")
done
sit_libs=()
for sit_lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
  sit_real=$(readlink -f "/lib64/$sit_lib")
  sit_libs+=(--volume "$sit_real:/opt/host-driver/$sit_lib:ro")
done
for sit_lib in /lib64/libnvidia-gpucomp.so.* /lib64/libnvidia-nvvm.so.*; do
  if [[ -f "$sit_lib" ]]; then
    sit_libs+=(--volume "$sit_lib:/opt/host-driver/$(basename "$sit_lib"):ro")
  fi
done
sit_options=(run --rm --name sit-paper-protocol-20261117
  --network host --security-opt label=disable --shm-size 1g
  --memory 12g --memory-swap 12g --pids-limit 256
  --volume "$sit_project:/workspace:ro"
  --volume "$sit_run:/workspace/reproduction/paper_protocol_20261117:rw"
  --volume /usr/bin/nvidia-smi:/usr/local/bin/nvidia-smi:ro
  --env LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu
  --env PYTHONPATH=/workspace/reproduction/paper_protocol_20261117/deps
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1
  --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=4
  --env MPLBACKEND=Agg --env MPLCONFIGDIR=/workspace/reproduction/paper_protocol_20261117/.mpl
  --workdir /workspace/reproduction/paper_protocol_20261117
  --entrypoint /bin/bash)
exec "${sit_podman[@]}" "${sit_options[@]}" "${sit_devices[@]}" "${sit_libs[@]}" "$sit_image" "$@"
