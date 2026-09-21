# 论文复现实验环境

## 目标 profile

| 项目 | 约束 |
| --- | --- |
| Python | CPython 3.12.12 |
| CUDA toolkit | CUDA 12.x |
| SGLang | `sglang==0.4.8` |
| vLLM | `vllm==0.9.0` |
| FlashAttention | `2.7.4.post1+cu12torch2.7cxx11abiFALSE`，CPython 3.12 wheel |
| TileLang | `0.1.14` |
| Accelerate | `1.15.0` |
| Transformers | `4.52.3` |

环境创建和安装命令如下。所有命令必须在本仓库根目录执行，并且使用 `.venv/bin/python`，以免把包安装到系统 Python 或旧工作区环境。

```bash
make venv PYTHON_BOOTSTRAP=/home/cpwu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12
make install-paper-deps
```

`make install-paper-deps` 使用 `requirements/paper-py312-cu12.txt` 安装 SGLang、vLLM、TileLang、Accelerate、Transformers，并安装 `third_party/` 下的本地 FlashAttention wheel。它要求 `.venv` 为 CPython 3.12，且在结束时运行 `pip check` 与 `sglang`、`flash_attn`、`vllm` import 验证。

FlashAttention wheel 位于 `third_party/`，来源为 Dao-AILab 的 [v2.7.4.post1 release](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl)。该 wheel 不写入通用 requirements 文件；`requirements/paper-py312-cu12.txt` 记录可从索引安装的目标依赖。

## 本机核验记录（2026-09-21）

在 `SparseAttnUnified/.venv` 中观测到：

```text
Python      3.12.12
torch       2.7.0
triton      3.3.0
sglang      0.4.8
flash_attn  2.7.4.post1
vllm        0.9.0
tilelang    0.1.14
accelerate  1.15.0
transformers 4.52.3
```

`pip check` 通过，且 `sglang` 与 `flash_attn` 已可导入。`sgl-kernel` 尚未安装；在运行本仓库的 metadata converter、SWA 或 local sparse kernel 路径前，必须从本仓库编译该定制扩展。该 Make target 对应原来的 `sglang/sgl-kernel/cmd-build.sh`，但使用 `pip wheel`，无需 `uv`：

```bash
make install-sgl-kernel PYTHON="$PWD/.venv/bin/python"
.venv/bin/python -m pip show sgl-kernel
```

默认目标是 SM80（A100）。可显式选择 CUDA toolkit、一个或多个 GPU 架构、并行度与 ccache 目录：

```bash
make install-sgl-kernel PYTHON="$PWD/.venv/bin/python" \
  SGL_KERNEL_CUDA_HOME=/usr/local/cuda-12.8 \
  SGL_KERNEL_CUDA_ARCHS=80,89 \
  SGL_KERNEL_BUILD_JOBS=16 \
  SGL_KERNEL_CCACHE_DIR="$HOME/.cache/sgl-kernel/ccache"
```

生成的 wheel 与 CMake build tree 位于 `.build/sgl-kernel/`。该构建直接引用 `components/sgl-flash-attn` 的 FA2/SWA 源码；`make install-flash-attn` 的完整 editable FlashAttention 安装不是 SWA no-split/tuned 流程的必需步骤。

### vLLM / Transformers 兼容性

`vllm==0.9.0` 与 Transformers 5.x 会因 `aimv2` 重复注册而无法导入。因此本 profile 固定 `transformers==4.52.3`。在本机 `.venv` 中，`pip check` 与 `import vllm` 均已通过；这只验证依赖导入，不代表 GPU/E2E 已完成验证。

最后运行：

```bash
make env-check PYTHON="$PWD/.venv/bin/python"
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > results/environment/pip-freeze.txt
```
