PYTHON ?= python3
PYTHON_BOOTSTRAP ?= python3
VENV_DIR ?= .venv
SGL_KERNEL_CUDA_HOME ?= $(CUDA_HOME)
SGL_KERNEL_CUDA_ARCHS ?= 80
SGL_KERNEL_BUILD_JOBS ?= 16
SGL_KERNEL_COMPILE_THREADS ?= $(SGL_KERNEL_BUILD_JOBS)
SGL_KERNEL_CCACHE_DIR ?= $(CURDIR)/.build/ccache
MINFERENCE_CUDA_ARCHS ?= 8.0
CMAKE_ARGS ?=

.PHONY: help submodules venv env-check install-sgl-kernel install-minference

help:
	@printf '%s\n' \
	  'make submodules        Initialize source and dependency submodules' \
	  'make venv              Create an isolated Python environment' \
	  'make env-check         Show Python, PyTorch, CUDA and GPU versions' \
	  'make install-sgl-kernel Build and install the SPTC operator' \
	  'make install-minference Install the MInference integration'

submodules:
	@git submodule sync --recursive
	@git submodule update --init --recursive

venv:
	@VENV_DIR="$(VENV_DIR)" PYTHON_BOOTSTRAP="$(PYTHON_BOOTSTRAP)" tools/create_venv.sh

env-check:
	@$(PYTHON) tools/check_env.py

install-sgl-kernel:
	@PYTHON="$(PYTHON)" SGL_KERNEL_CUDA_HOME="$(SGL_KERNEL_CUDA_HOME)" SGL_KERNEL_CUDA_ARCHS="$(SGL_KERNEL_CUDA_ARCHS)" SGL_KERNEL_BUILD_JOBS="$(SGL_KERNEL_BUILD_JOBS)" SGL_KERNEL_COMPILE_THREADS="$(SGL_KERNEL_COMPILE_THREADS)" SGL_KERNEL_CCACHE_DIR="$(SGL_KERNEL_CCACHE_DIR)" SGL_KERNEL_EXTRA_CMAKE_ARGS="$(CMAKE_ARGS)" tools/build_sgl_kernel.sh

install-minference:
	@MINFERENCE_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="$(MINFERENCE_CUDA_ARCHS)" $(PYTHON) -m pip install --no-build-isolation -e integrations/minference
