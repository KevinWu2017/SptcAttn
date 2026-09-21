#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
profile="${1:-quick}"
if [[ $# -gt 0 ]]; then shift; fi

case "${profile}" in
  quick)
    output_dir="${repo_root}/results/e2e/quick"
    profile_args=(
      --context_windows 10k
      --methods dense,minference
      --minference_backends v5
      --minference_block_sizes 32
      --warmup 1 --repeats 2
    )
    ;;
  paper)
    output_dir="${repo_root}/results/e2e/paper"
    profile_args=(
      --context_windows 10k,50k,100k,200k,300k,500k,1m
      --methods dense,a_shape,minference
      --minference_backends triton,cutlass,v5
      --minference_block_sizes 32,64
      --warmup 1 --repeats 10
    )
    ;;
  *)
    echo "Usage: $0 [quick|paper] [profile_e2e_sweep.py options...]" >&2
    exit 2
    ;;
esac

mkdir -p "${output_dir}"
cd "${repo_root}/baselines/minference"
exec "${python_bin}" experiments/benchmarks/profile_e2e_sweep.py \
  "${profile_args[@]}" \
  --output-dir "${output_dir}" \
  --run-name "${profile}" \
  "$@"
