#!/usr/bin/env bash
# 验证「带 mask 的注意力按空闲显存自适应选块」这条改动。
#
# 1) 跑 Krea2 modeling 的全部单测（含新增的选块用例）
# 2) 在真卡上打印当前余量下会选到多大的块，以及那一块要吃多少显存
#
# 用法：bash tools/check_adaptive_chunk.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/2 单测 ==="
python -m pytest tests/test_krea2_modeling.py -q

echo
echo "=== 2/2 真卡选块 ==="
python - <<'PY'
import torch

from modeling.krea2.krea2_modeling import (
    _CHUNK_FREE_VRAM_FRACTION,
    _MASKED_ATTN_QUERY_CHUNK,
    _MATH_SDPA_OVERHEAD,
    _pick_query_chunk,
)

GIB = 1024 ** 3
print(f"torch {torch.__version__}  hip={torch.version.hip}  cuda={torch.version.cuda}")
print(f"常数：fallback={_MASKED_ATTN_QUERY_CHUNK} "
      f"overhead={_MATH_SDPA_OVERHEAD} fraction={_CHUNK_FREE_VRAM_FRACTION}")

if not torch.cuda.is_available():
    raise SystemExit("没有可用加速卡，跳过真卡部分")

# 真机 2048px 桶：1728x2432 -> VAE/8 -> patch2 -> 16928 image token
CASES = [
    ("2048px 桶 bs=1", 1, 48, 16928),
    ("2048px 桶 bs=2", 2, 48, 16928),
    ("1024px 桶 bs=2", 2, 48, 4224),
]
for idx in range(torch.cuda.device_count()):
    dev = torch.device(f"cuda:{idx}")
    free, total = torch.cuda.mem_get_info(dev)
    print(f"\ncuda:{idx}  空闲 {free / GIB:.2f} / {total / GIB:.2f} GiB")
    for name, b, h, s_k in CASES:
        # 只要 shape/device/is_cuda，用极小张量占位即可
        q = torch.empty((b, h, 8, 64), device=dev, dtype=torch.bfloat16)
        chunk = _pick_query_chunk(q, s_k)
        need = b * h * chunk * s_k * 4 * _MATH_SDPA_OVERHEAD
        print(f"  {name:16} S_k={s_k:>6} -> chunk={chunk:>5}  "
              f"该块约需 {need / GIB:5.2f} GiB")
        del q
PY
echo
echo "全部通过。"
