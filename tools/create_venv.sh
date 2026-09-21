#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bootstrap_python="${PYTHON_BOOTSTRAP:-python3}"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"

if [[ "${venv_dir}" != /* ]]; then
  venv_dir="${repo_root}/${venv_dir}"
fi

if [[ -e "${venv_dir}" ]]; then
  if [[ -x "${venv_dir}/bin/python" ]]; then
    echo "Virtual environment already exists: ${venv_dir}"
    echo "Activate it with: source ${venv_dir}/bin/activate"
    exit 0
  fi
  echo "Refusing to overwrite non-venv path: ${venv_dir}" >&2
  exit 1
fi

"${bootstrap_python}" -m venv "${venv_dir}"
echo "Created virtual environment: ${venv_dir}"
echo "Next: source ${venv_dir}/bin/activate"
echo "Then install a CUDA-compatible PyTorch wheel before running make install-* or make build."
