# 使用新仓库

本仓库是可独立构建的 SparseAttn 工作入口。所有构建产物、实验 CSV、图和 profile 都写入本仓库的 `.build/` 或 `results/`；不会写入同级旧工作区。

## 1. 准备源码

克隆或进入 `SparseAttnUnified` 后先初始化 source submodule。`components/flash-attn`、`components/sglang`、`third_party/cutlass`、`experiments/legacy` 和 `baselines/minference` 各自保留独立 Git 历史与 upstream。

```bash
cd /path/to/SparseAttnUnified
make submodules
git status
make help
```

`SptcAttn-Paper` 是论文源码与已发表图表的独立归档；新的实验应从本仓库的脚本产生 CSV，再使用 `reports/figures/` 作图，而不应覆盖归档数据。

## 2. 建立环境

需要 Linux、NVIDIA GPU（v5 的 `mma.sp` 要求 SM80/Ampere 或更新）、可用的 `nvcc`、CMake >= 3.22，以及与本机 CUDA ABI 兼容的 PyTorch。论文记录的环境是 CUDA 12.8 / PyTorch 2.7；当前 `sgl-kernel` 上游快照的 README 则写的是 PyTorch 2.9.1。两者不能混用：选择一个与 CUDA 相容的 PyTorch 版本后，重新编译两个本地扩展，并把实际版本写入结果目录。

当前论文复现 profile 固定使用 CPython 3.12；安装命令、FlashAttention 本地 wheel 路径与核验规则见 [环境记录](ENVIRONMENT.md)。

```bash
make venv PYTHON_BOOTSTRAP=/path/to/python3.12
source .venv/bin/activate
python -m pip install --upgrade pip

# 先按 PyTorch 官方 CUDA wheel 索引安装与你机器匹配的 torch。
# 例如论文复现环境使用 PyTorch 2.7.x + CUDA 12.8 对应 wheel。
python -m pip install cmake ninja scikit-build-core setuptools wheel packaging pytest matplotlib

make env-check PYTHON="$(which python)"
```

若使用本仓库记录的论文复现 profile，在已下载 FlashAttention wheel 后可直接运行 `make install-paper-deps`；固定版本与 wheel 前置条件见 [环境记录](ENVIRONMENT.md)。

环境检查只读取状态。它应显示 `cuda ready: True` 与 `sm_80` 或更高。若 PyTorch 的 CUDA 版本与 `nvcc` / 驱动不兼容，先修复该问题再构建。

## 3. 构建本地扩展与 CMake demo

`sgl_kernel` 提供 metadata converter 和 Python 算子；MInference 提供 Triton/CUTLASS 对照路径。两者都必须由当前环境本地编译，避免误用旧工作区或预编译 wheel。

```bash
make install-flash-attn PYTHON="$(which python)"  # FA2/SWA baseline needed only for SWA
make install-sgl-kernel PYTHON="$(which python)"
make install-minference PYTHON="$(which python)"

# A100: 80；RTX 4090: 89；H100: 90。
make configure PYTHON="$(which python)" \
  CMAKE_ARGS="-DPython_EXECUTABLE=$(which python) -DCMAKE_CUDA_ARCHITECTURES=80 -DTORCH_CUDA_ARCH_LIST=8.0"
make build
make smoke
```

`make smoke` 仅运行一个小型 v1/v5 sliding-window case。它是构建检查，不是论文正确性或性能结论。

## 4. 跑 kernel 实验

先运行小样本，再跑论文的 kernel matrix。两个命令都使用固定种子、FP16、head dim 128、block 32；`paper` 配置覆盖 4K--256K、causal/non-causal、四个实现（v5、MInference CUTLASS、MInference Triton、dense）。

```bash
make kernel-quick PYTHON="$(which python)"
make kernel-paper PYTHON="$(which python)"
```

输出为：

```text
results/kernel/quick/{latency.csv,latency.md}
results/kernel/paper/{latency.csv,latency.md}
```

可把额外参数透传给 runner，例如仅跑非因果模式：

```bash
PYTHON_BIN="$(which python)" tools/run_kernel_latency.sh paper --causal-modes noncausal
```

运行前必须先通过独立的数值正确性测试；性能 CSV 不等价于 lossless 证明。

## 5. 跑端到端实验

E2E runner 使用 `gradientai/Llama-3-8B-Instruct-Gradient-1048k` 默认模型。它会通过 Hugging Face 下载受限模型/权重，运行前需要用户已具备访问权限、登录状态和足够显存；1M case 在 80GB GPU 上也可能 OOM。它的结果只写进 `results/e2e/`。

```bash
# 10K、dense vs v5、两次计时
make e2e-quick PYTHON="$(which python)"

# 10K--1M、dense/A-shape/Triton/CUTLASS/v5、block 32/64
make e2e-paper PYTHON="$(which python)"
```

完整 E2E 输出为 `results/e2e/paper/paper.csv`、`paper.json` 与 `paper_summary.tsv`。想更换模型、prompt 或启用 CPU KV cache 时，透传原脚本参数：

```bash
PYTHON_BIN="$(which python)" tools/run_e2e_sweep.sh paper \
  --model_name your-org/your-model --prompt_path /absolute/path/prompt.txt --kv_cache_cpu
```

NIAH、InfiniteBench 和 RULER 的原始入口在 `baselines/minference/experiments/`；它们需自行准备数据集和模型访问权限。运行时把 `--output_dir` 指向 `results/accuracy/<suite>`，不要写回源码目录。

## 6. 画图

Sliding-window 实验同样提供 quick 与论文规模配置；它需要先安装本地 FlashAttention，以提供 FA2 exact SWA 基线：

```bash
make swa-quick PYTHON="$(which python)"
make swa-paper PYTHON="$(which python)"
```

三个 plot 脚本都接受显式 CSV，默认将 PDF/PNG 写入 `results/plots/`。Kernel 和 E2E 可先用随源码保存的归档 CSV 验证绘图环境；SWA 则使用刚跑出的 CSV：

```bash
make plot-kernel PYTHON="$(which python)"
make plot-e2e PYTHON="$(which python)"
make plot-swa PYTHON="$(which python)" SWA_CSV=results/swa/paper/latency.csv
```

对新结果指定 CSV：

```bash
make plot-kernel PYTHON="$(which python)" KERNEL_CSV=results/kernel/paper/latency.csv
make plot-e2e PYTHON="$(which python)" E2E_CSV=results/e2e/paper/paper.csv
```

也可以直接调用脚本并指定图名：

```bash
python reports/figures/plot_sparse_latency.py \
  --input-csv results/kernel/paper/latency.csv \
  --output-pdf results/plots/kernel_paper.pdf \
  --output-png results/plots/kernel_paper.png
```

## 7. 可复现记录

每次准备论文图前，保存以下信息到相应 `results/<suite>/`：Git commit、`make env-check` 输出、GPU/驱动/CUDA、PyTorch/Triton/MInference 版本、完整命令、随机种子与原始 CSV。不要用不同 GPU、不同 causal mask 或不同 sparse pattern 的数据合并成同一个 speedup 结论。
