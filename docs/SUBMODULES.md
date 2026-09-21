# Submodule 管理

顶层仓库是 orchestration repo，不复制组件源码。首次 clone 后初始化所有组件：

```bash
git clone --recurse-submodules <SparseAttnUnified-remote>
# 或在已有 clone 中：
make submodules
```

组件的 URL 与固定 revision 位于 `.gitmodules` 和顶层 gitlink。查看当前状态：

```bash
git submodule status --recursive
git -C components/flash-attn status --short
git -C components/sglang status --short
git -C experiments/legacy status --short
git -C baselines/minference status --short
```

当前 `flash-attn`、`sglang`、`experiments/legacy`、`minference` 被有意保留为脏工作树，以便审查原有本地 SparseAttn 改动。不要使用 `git submodule update --checkout --force`，否则这些未提交改动会被清除。

确认要保留的改动后，应在对应 submodule 中创建独立提交并推送到其 fork；之后回到顶层提交更新后的 gitlink：

```bash
git -C components/flash-attn add <files>
git -C components/flash-attn commit -m "..."

git add components/flash-attn
git commit -m "chore: advance flash-attn submodule"
```

不要把 build、wheel、profile 和 benchmark 输出提交到 submodule；它们属于顶层 `results/` 或本地 `.build/`。
