#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"
if [[ "${venv_dir}" != /* ]]; then
  venv_dir="${repo_root}/${venv_dir}"
fi
python_bin="${venv_dir}/bin/python"
flash_attn_wheel="${repo_root}/third_party/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

if [[ ! -x "${python_bin}" ]]; then
  echo "Virtual environment not found: ${venv_dir}" >&2
  echo "Create it first: make venv PYTHON_BOOTSTRAP=/path/to/python3.12" >&2
  exit 1
fi
if [[ ! -f "${flash_attn_wheel}" ]]; then
  echo "Missing required local FlashAttention wheel: ${flash_attn_wheel}" >&2
  echo "Download the wheel documented in docs/ENVIRONMENT.md into third_party/." >&2
  exit 1
fi

"${python_bin}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Paper profile requires CPython 3.12; found {sys.version.split()[0]}")
PY

"${python_bin}" -m pip install -r "${repo_root}/requirements/paper-py312-cu12.txt"
"${python_bin}" -m pip install "${flash_attn_wheel}" --no-build-isolation --no-cache-dir
"${python_bin}" -m pip check
"${python_bin}" -c 'import sglang, flash_attn, vllm; print("paper dependencies: imports OK")'
