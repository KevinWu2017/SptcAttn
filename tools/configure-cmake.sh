#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake_source="${repo_root}/experiments/legacy/cmake_sparse_compare"
cmake_build="${repo_root}/.build/cmake"

cmake -S "${cmake_source}" -B "${cmake_build}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}" \
  "$@"
