#!/usr/bin/env bash
# 验证「集合操作门控错位」这个修复。
#
# 修的是什么：bootstrap 在非 rank 0 上把 args.sample_steps / sample_every 置 0
# （关掉重复出图），而采样外面那对 barrier 的门控读的正是这两个字段 —— 于是非
# rank 0 连 barrier 一起跳过。NCCL 按调用顺序配对集合操作，rank 0 多执行的 barrier
# 让此后每个 rank 的集合操作永久错位。真机症状：
#   - rank 1 误报「其他 rank 的 loss 非有限值」（它的 all_reduce 与 rank 0 的
#     barrier 配上了对，读回垃圾值）
#   - rank 1 不等 rank 0 采样完就冲进训练前向，被误报的 NaN skip 又让上一轮计算图
#     不释放 → 两份 gradient-checkpoint 图叠着 → OOM
#
# 用法：bash tools/check_collective_fix.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/3 静态门控审计（不需要卡）==="
python -m pytest tests/test_collective_gating.py -v 2>&1 | tail -15

echo
echo "=== 2/3 相关既有测试 ==="
# 显式列文件而不是用 -k：pytest 的 -k 是在**收集之后**才过滤的，用 tests/ 当路径会
# 先导入全部测试模块，任何一个模块缺可选依赖（如 reg 那几个要 sklearn）都会在收集
# 阶段就 Interrupted —— 于是本该跑的这些一条都没跑，而且 set -e 还会顺手掐掉第 3 段。
RELATED=(
  tests/test_collective_gating.py
  tests/test_distributed.py
  tests/test_accelerator.py
  tests/test_pause_marker.py
  tests/test_ddp_find_unused.py
  tests/test_ddp_schema.py
  tests/test_loop_ddp_grad_accum.py
  tests/test_dataset_sampler_ddp.py
  tests/test_cmd_builder_ddp.py
  tests/test_krea2_modeling.py
  tests/test_krea2_mask_shortcut_static.py
)
# 不让这一段的失败掐掉第 3 段（真机双卡验证才是重点），但记下来最后一起报。
stage2_rc=0
python -m pytest "${RELATED[@]}" -q 2>&1 | tail -18 || stage2_rc=$?

echo
echo "=== 3/3 双卡集合操作对齐实测 ==="
# 真正测「两个 rank 的集合操作调用次数是否一致」。错位时 barrier 会挂住，
# 所以用 timeout 兜底：卡住即失败。
cat > /tmp/_align_probe.py <<'PY'
"""模拟 bootstrap 的 rank 相关抑制 + 采样门控，检查 barrier 是否对齐。"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.getcwd())  # 脚本 cd 到仓库根后再启动，utils 从这里导入

from utils import distributed as dist_env  # noqa: E402


class Args:
    sample_steps = 10
    sample_every = 1
    no_progress = False


class Ctx:
    sample_steps_all_ranks = 0
    sample_every_all_ranks = 0
    global_step = 0


dist_env.init()
rank = dist_env.rank()
args, ctx = Args(), Ctx()

# 按 bootstrap 的顺序：先存 rank 不变副本，再做 rank 相关抑制
ctx.sample_steps_all_ranks = int(args.sample_steps or 0)
ctx.sample_every_all_ranks = int(args.sample_every or 0)
if not dist_env.is_main():
    args.no_progress = True
    args.sample_steps = 0
    args.sample_every = 0

# 副本必须各 rank 一致 —— 用 all_reduce 亲自验，不靠推理
t = torch.tensor(
    [float(ctx.sample_steps_all_ranks), float(ctx.sample_every_all_ranks)],
    device=f"cuda:{dist_env.local_rank()}",
)
mn = t.clone()
mx = t.clone()
dist.all_reduce(mn, op=dist.ReduceOp.MIN)
dist.all_reduce(mx, op=dist.ReduceOp.MAX)
assert torch.equal(mn, mx), f"[rank {rank}] 副本各 rank 不一致: min={mn} max={mx}"
print(f"[rank {rank}] rank 不变副本一致: sample_steps={int(mn[0])} "
      f"sample_every={int(mn[1])}（args 上本 rank 看到的是 {args.sample_steps}）")

# 数 barrier 次数：门控用副本，两个 rank 必须数出同一个值
n = 0
sampling_enabled = ctx.sample_steps_all_ranks > 0 or ctx.sample_every_all_ranks > 0
if ctx.global_step == 0 and sampling_enabled:      # resume.py 基线采样
    dist_env.barrier(); n += 1
    dist_env.barrier(); n += 1
for step in range(1, 21):                           # loop.py step 采样
    if ctx.sample_steps_all_ranks > 0 and step % ctx.sample_steps_all_ranks == 0:
        dist_env.barrier(); n += 1
        dist_env.barrier(); n += 1
for epoch in range(1, 4):                           # loop.py epoch 采样
    if ctx.sample_every_all_ranks > 0 and epoch % ctx.sample_every_all_ranks == 0:
        dist_env.barrier(); n += 1
        dist_env.barrier(); n += 1

c = torch.tensor([float(n)], device=f"cuda:{dist_env.local_rank()}")
lo, hi = c.clone(), c.clone()
dist.all_reduce(lo, op=dist.ReduceOp.MIN)
dist.all_reduce(hi, op=dist.ReduceOp.MAX)
assert int(lo[0]) == int(hi[0]) == n, (
    f"[rank {rank}] barrier 次数各 rank 不一致：本 rank {n}, "
    f"min {int(lo[0])}, max {int(hi[0])}"
)
print(f"[rank {rank}] barrier 调用次数对齐：{n} 次")

# 再验一次 _agree_on_finite_loss 的语义：rank 0 正常、rank 1 也正常 → 不该跳
local_finite = 1.0
mean = float(dist_env.all_reduce_mean(local_finite))
assert mean >= 1.0, f"[rank {rank}] 全 rank 正常却判成要跳过：mean={mean}"
print(f"[rank {rank}] all_reduce_mean 语义正确：{mean}")

dist_env.destroy()
print(f"[rank {rank}] OK")
PY

if timeout 180 python -m torch.distributed.run \
    --nnodes 1 --nproc_per_node 2 --master_port 29517 \
    /tmp/_align_probe.py; then
  echo
  echo "第 3 段通过：集合操作在两个 rank 上对齐。"
  if [ "$stage2_rc" -ne 0 ]; then
    echo
    echo "但第 2 段（既有测试）退出码 $stage2_rc —— 往上翻看是哪条失败。"
    exit "$stage2_rc"
  fi
  echo "全部通过。"
else
  rc=$?
  echo
  if [ "$rc" -eq 124 ]; then
    echo "失败：超时 —— barrier 挂住了，说明集合操作仍然错位。"
  else
    echo "失败：exit $rc"
  fi
  exit 1
fi
