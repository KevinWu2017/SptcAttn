#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
profile="${1:-quick}"
if [[ $# -gt 0 ]]; then shift; fi

case "${profile}" in
  quick)
    output_dir="${repo_root}/results/swa/quick"
    profile_args=(
      --rows 128:32:32
      --ctx-columns 32768
      --warmup 5 --iters 20 --seed 0
    )
    ;;
  paper)
    output_dir="${repo_root}/results/swa/paper"
    profile_args=(
      --rows 128:32:32,128:32:64
      --ctx-columns 32768,65536,131072,262144
      --warmup 20 --iters 100 --seed 0
    )
    ;;
  *)
    echo "Usage: $0 [quick|paper] [benchmark_sparse_sliding_window.py options...]" >&2
    exit 2
    ;;
esac

mkdir -p "${output_dir}"
export PYTHONPATH="${repo_root}/components/sglang/sgl-kernel/python:${repo_root}/baselines/minference${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" "${repo_root}/experiments/legacy/cmake_sparse_compare/benchmark_sparse_sliding_window.py" \
  "${profile_args[@]}" \
  --minference-root "${repo_root}/baselines/minference/minference" \
  --markdown-out "${output_dir}/latency.md" \
  --csv-out "${output_dir}/latency.csv" \
  "$@"
