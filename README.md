# SparseAttn Unified

这是 SparseAttn 的 orchestration repo。顶层维护环境、构建、实验与文档；核心源码保留为 Git submodule，以维持各自 upstream、提交历史和独立变更边界。旧工作区中的任何文件均未移动、删除或修改。

## 从这里开始

```bash
cd /home/cpwu/working/SparseAttn/SparseAttnUnified
git submodule update --init --recursive
make help
make venv PYTHON_BOOTSTRAP=/path/to/python3.12
source .venv/bin/activate
make env-check PYTHON=/path/to/python-with-torch
make install-sgl-kernel PYTHON=/path/to/python-with-torch
make configure PYTHON=/path/to/python-with-torch CMAKE_ARGS="-DPython_EXECUTABLE=/path/to/python-with-torch"
make build
```

`make build` 构建 v1-v5 sliding-window 对照程序；其构建目录为 `.build/cmake`，不会写入旧工作区。完整的源码准备、环境安装、编译、kernel/E2E 实验及画图步骤见 [使用指南](docs/USAGE.md)。

论文复现使用的 CPython 3.12 / CUDA 12 依赖 profile 与本机核验状态见 [环境记录](docs/ENVIRONMENT.md)。

## 当前范围

| 路径 | 职责 |
| --- | --- |
| `components/sgl-flash-attn` | Git submodule：稀疏 FlashAttention CUDA API、调度和 v1/v5 kernel |
| `components/sglang` | Git submodule：`sgl_kernel` Python/C++ binding 与 SGLang 集成 |
| `third_party/cutlass` | Git submodule：CUTLASS/CuTe 依赖 |
| `experiments/legacy` | Git submodule：`Kernels` demo、CMake 比较器和基准脚本 |
| `baselines/minference` | Git submodule：Triton 与端到端对照基线 |
| `reports/figures` | 可视化脚本；生成结果应写到 `results/` |
| `docs` | 架构、迁移边界和后续整理规则 |

## 支持策略

- 产品和新验证链只保留 **v1（基线）与 v5（优化目标）**。
- **v4 不再是新仓库的公开接口、测试基线或 benchmark 对象。** 初始快照中仍有少量 v4 源码，因为 v1/v5 CUDA 编译单元和历史 API 共用文件；它将在 v1↔v5 独立正确性回归建立后，以单独提交删除。
- v2/v3 已在旧扩展中不可用，后续会从公共 Python API 移除或改成明确的兼容错误。

详见 [使用指南](docs/USAGE.md)、[submodule 管理](docs/SUBMODULES.md)、[架构说明](docs/ARCHITECTURE.md)、[迁移清单](docs/MIGRATION.md)、[来源记录](docs/PROVENANCE.md) 和 [v4 退役计划](docs/V4_RETIREMENT.md)。
