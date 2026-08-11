#!/usr/bin/env bash
# 验证「LoKr 走 bypass_forward_diff 而非 rebuild」的数值等价性 + 性能改进。
#
# 背景：lycoris ac2616f (2025-10-04) 把 rebuild forward 从一次 matmul 改成两次
# （基座 + 稠密 delta），每个注入层约 2× FLOPs。bypass 走 Kronecker 恒等式，
# 第二次 matmul 只需 1/factor 的量（factor=8 时注入层 2.0 → 1.125 单位）。
# 本仓已给 LoRA 开过（issue #182），LoKr 数学也是等价的，只是 upstream 一直
# 未从 FLOPs 角度看过它。
#
# 修了什么：utils/lycoris_adapter.py:181 的条件加上 lokr；外加 rank_dropout
# 与 DoRA 两个 guard（bypass 对它们会静默失效），配上两条 AST 守卫测试。
#
# 用法：bash tools/check_lokr_bypass.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/3 bypass 等价性测试 + guard 守卫测试 ==="
python -m pytest tests/test_lycoris_bypass.py tests/test_lycoris_tlora.py::test_tlora_inject_never_sets_bypass_mode tests/test_lycoris_tlora.py::test_lokr_bypass_is_gated_on_rank_dropout -v 2>&1 | grep -E "(PASSED|FAILED|ERROR|test_)" | tail -20

echo
echo "=== 2/3 本机实测数值等价（默认 Krea2 MLP 尺寸）==="
python tools/measure_lokr_bypass.py --tokens 512 --iters 5

echo
echo "=== 3/3 full matrix 分支（用户生产配置）==="
python tools/measure_lokr_bypass.py --tokens 512 --iters 5 --full-matrix
