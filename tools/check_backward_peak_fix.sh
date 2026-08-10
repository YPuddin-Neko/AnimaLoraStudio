#!/usr/bin/env bash
# 验证「分块注意力在反向里也真的省显存」这个修复。
#
# 修的是什么：外层 checkpoint 的首次前向在 no_grad 下跑，每块的分数矩阵是临时量，
# 分块有效。但反向里 recompute_fn 带 grad 重跑，每次 SDPA 都为自己的反向保留
# attention weights —— S_q/chunk 份同时活着 = 不分块的整块。真机 chunk=256 时
# 61 块 × 1.43 GiB = 87.1 GiB（不分块 87.0 GiB），反向重算到第 19 块就 OOM。
# 修法：每块再套一层 checkpoint，峰值回到「一块」，代价是该块 attention 多算一遍。
#
# 用法：bash tools/check_backward_peak_fix.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/2 单测（含两条新增：调用次数 + 梯度一致）==="
python -m pytest tests/test_krea2_modeling.py -q 2>&1 | tail -8

echo
echo "=== 2/2 真卡实测 forward / backward 峰值 ==="
echo "要看的是：forward 峰值随 chunk 变化，backward 峰值**也**随 chunk 变化。"
echo "修复前 backward 那一列会几乎持平且接近不分块 —— 那就是分块在反向里失效。"
echo
python tools/measure_attn_backward_peak.py

echo
echo "完成。"
