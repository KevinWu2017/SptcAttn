#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/components/sglang/sgl-kernel"
python_bin="${PYTHON:-python3}"
cuda_home="${SGL_KERNEL_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}"
arch_input="${SGL_KERNEL_CUDA_ARCHS:-80}"
build_jobs="${SGL_KERNEL_BUILD_JOBS:-16}"
compile_threads="${SGL_KERNEL_COMPILE_THREADS:-${build_jobs}}"
ccache_dir="${SGL_KERNEL_CCACHE_DIR:-${HOME}/.cache/sgl-kernel/ccache}"
extra_cmake_args="${SGL_KERNEL_EXTRA_CMAKE_ARGS:-}"
build_root="${repo_root}/.build/sgl-kernel"
build_dir="${build_root}/build"

if [[ ! -x "${python_bin}" && "${python_bin}" == */* ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi
if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
  echo "CUDA compiler not found: ${cuda_home}/bin/nvcc" >&2
  echo "Set SGL_KERNEL_CUDA_HOME=/path/to/cuda when invoking make." >&2
  exit 1
fi

torch_cmake_prefix="$(${python_bin} - <<'PY'
import scikit_build_core  # noqa: F401
import torch

print(torch.utils.cmake_prefix_path)
PY
)" || {
  echo "The selected Python needs both torch and scikit-build-core." >&2
  echo "Install the paper dependencies, or run: ${python_bin} -m pip install scikit-build-core" >&2
  exit 1
}

IFS=',' read -r -a raw_arches <<< "${arch_input//;/,}"
arches=()
flashinfer_arches=()
for raw_arch in "${raw_arches[@]}"; do
  arch="${raw_arch//[[:space:]]/}"
  if [[ "${arch}" =~ ^([0-9])\.([0-9])$ ]]; then
    arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
  fi
  if [[ ! "${arch}" =~ ^[0-9]+a?$ ]]; then
    echo "Invalid CUDA architecture '${raw_arch}'; use values such as 80, 89, or 90a." >&2
    exit 1
  fi
  arches+=("${arch}")
  numeric_arch="${arch%a}"
  flashinfer_arches+=("${numeric_arch:0:${#numeric_arch}-1}.${numeric_arch: -1}")
done
if [[ ${#arches[@]} -eq 0 ]]; then
  echo "At least one CUDA architecture is required." >&2
  exit 1
fi

cmake_arches="$(IFS=';'; echo "${arches[*]}")"
flashinfer_arch_list="$(IFS=';'; echo "${flashinfer_arches[*]}")"
cuda_version="$(${cuda_home}/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"

mkdir -p "${build_dir}" "${build_root}" "${ccache_dir}"
wheel_dir="$(mktemp -d "${build_root}/wheels.XXXXXX")"
export CUDA_HOME="${cuda_home}"
export CUDA_PATH="${cuda_home}"
export CUDA_VERSION="${SGL_KERNEL_CUDA_VERSION:-${cuda_version}}"
export FLASHINFER_CUDA_ARCH_LIST="${flashinfer_arch_list}"
export CMAKE_BUILD_PARALLEL_LEVEL="${build_jobs}"
export MAX_JOBS="${build_jobs}"
export CCACHE_DIR="${ccache_dir}"
export CCACHE_BASEDIR="${repo_root}"
if [[ -d "${cuda_home}/include/cccl" ]]; then
  export CPLUS_INCLUDE_PATH="${cuda_home}/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
  export C_INCLUDE_PATH="${cuda_home}/include/cccl${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
fi

if command -v ccache >/dev/null 2>&1; then
  ccache -s
else
  echo "ccache not found; continuing without compiler cache." >&2
fi

export CMAKE_ARGS="${extra_cmake_args} -DSGL_KERNEL_COMPILE_THREADS=${compile_threads} -DSGL_KERNEL_CUDA_ARCHS=${cmake_arches} -DSGL_KERNEL_CUTLASS_SOURCE_DIR=${repo_root}/third_party/cutlass -DSGL_FLASH_ATTENTION_SOURCE_DIR=${repo_root}/components/sgl-flash-attn -DCMAKE_CUDA_COMPILER=${cuda_home}/bin/nvcc -DCUDAToolkit_ROOT=${cuda_home} -DCMAKE_PREFIX_PATH=${torch_cmake_prefix} -DCUDA_VERSION=${CUDA_VERSION}"

echo "Building sgl-kernel with CUDA ${CUDA_VERSION}, SM ${cmake_arches}, jobs ${build_jobs}"
"${python_bin}" -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "${wheel_dir}" \
  --config-settings="build-dir=${build_dir}" \
  "${source_dir}"

wheel_path="$(find "${wheel_dir}" -maxdepth 1 -type f -name 'sgl_kernel-*.whl' -print -quit)"
if [[ -z "${wheel_path}" ]]; then
  echo "sgl-kernel wheel was not produced in ${wheel_dir}" >&2
  exit 1
fi
"${python_bin}" -m pip install --force-reinstall --no-deps "${wheel_path}"
"${python_bin}" -c 'import sgl_kernel, sgl_kernel.fa2_nosplit_ops; print("sgl-kernel: import OK")'
