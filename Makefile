CMAKE_ARGS ?=
PYTHON ?= python3
PYTHON_BOOTSTRAP ?= python3
VENV_DIR ?= .venv
KERNEL_CSV ?= reports/figures/data/sparse_latency_matrix_a100.csv
SWA_CSV ?= results/swa/paper/latency.csv
E2E_CSV ?= baselines/minference/experiments/profile_e2e_results/gradientai_Llama-3-8B-Instruct-Gradient-1048k_20260520_105150.csv

.PHONY: help submodules venv install-paper-deps env-check install-flash-attn install-sgl-kernel install-minference configure build smoke kernel-quick kernel-paper swa-quick swa-paper e2e-quick e2e-paper plot-kernel plot-swa plot-e2e clean

help:
	@printf '%s\n' 'SparseAttn Unified targets:' \
	  '  make submodules Initialize and synchronize all source submodules' \
	  '  make venv       Create an isolated .venv without changing an existing environment' \
	  '  make install-paper-deps  Install the pinned CPython 3.12 paper environment into .venv' \
	  '  make env-check  Show Python, PyTorch, CUDA and GPU compatibility' \
	  '  make install-flash-attn  Build/install local FlashAttention for FA2 SWA baselines' \
	  '  make install-sgl-kernel  Build/install the local sgl_kernel extension' \
	  '  make install-minference  Build/install the local MInference extension' \
	  '  make configure  Configure the isolated CMake build in .build/cmake' \
	  '                  Pass CMAKE_ARGS="-DPython_EXECUTABLE=/path/to/python" when needed' \
	  '  make build      Build the v1-v5 sliding-window comparison executable' \
	  '  make smoke      Run a small v1-v5 sliding-window case (requires CUDA)' \
	  '  make kernel-quick / kernel-paper  Run kernel latency profiles into results/kernel/' \
	  '  make swa-quick / swa-paper  Run sliding-window profiles into results/swa/' \
	  '  make e2e-quick / e2e-paper  Run MInference E2E profiles into results/e2e/' \
	  '  make plot-kernel / plot-swa / plot-e2e  Render CSV input into results/plots/' \
	  '  make clean      Remove only this repository build directory'

submodules:
	@git submodule sync --recursive
	@git submodule update --init --recursive

venv:
	@VENV_DIR="$(VENV_DIR)" PYTHON_BOOTSTRAP="$(PYTHON_BOOTSTRAP)" tools/create_venv.sh

install-paper-deps:
	@VENV_DIR="$(VENV_DIR)" tools/install_paper_deps.sh

env-check:
	@$(PYTHON) tools/check_env.py

install-flash-attn:
	@$(PYTHON) -m pip install --no-build-isolation -e components/flash-attn

install-sgl-kernel:
	@$(PYTHON) -m pip install --no-build-isolation -e components/sglang/sgl-kernel

install-minference:
	@MINFERENCE_FORCE_BUILD=TRUE $(PYTHON) -m pip install --no-build-isolation -e baselines/minference

configure:
	@tools/configure-cmake.sh $(CMAKE_ARGS)

build: configure
	@cmake --build .build/cmake --target sparse_compare_v1_v5_sliding_window_demo -j

smoke: build
	@.build/cmake/sparse_compare_v1_v5_sliding_window_demo --ctx 256 --dim 64 --block-size-m 32 --block-size-n 32 --window-size 64 --warmup 2 --iters 5

kernel-quick:
	@PYTHON_BIN="$(PYTHON)" tools/run_kernel_latency.sh quick

kernel-paper:
	@PYTHON_BIN="$(PYTHON)" tools/run_kernel_latency.sh paper

swa-quick:
	@PYTHON_BIN="$(PYTHON)" tools/run_sliding_window.sh quick

swa-paper:
	@PYTHON_BIN="$(PYTHON)" tools/run_sliding_window.sh paper

e2e-quick:
	@PYTHON_BIN="$(PYTHON)" tools/run_e2e_sweep.sh quick

e2e-paper:
	@PYTHON_BIN="$(PYTHON)" tools/run_e2e_sweep.sh paper

plot-kernel:
	@$(PYTHON) reports/figures/plot_sparse_latency.py --input-csv "$(KERNEL_CSV)"

plot-swa:
	@$(PYTHON) reports/figures/plot_sliding_window_latency.py --input-csv "$(SWA_CSV)"

plot-e2e:
	@$(PYTHON) reports/figures/plot_e2e_profile_latency.py --input-csv "$(E2E_CSV)"

clean:
	@rm -rf .build
