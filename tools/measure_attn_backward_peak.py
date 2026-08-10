"""实测：按 query 分块能压住 forward 峰值，但**压不住 backward 的重算峰值**。

结论如果成立，会看到 forward 峰值随 chunk 线性下降，而 backward 峰值几乎不随 chunk
变化、且接近不分块的整块大小。

机制：``checkpoint(use_reentrant=False)`` 的首次前向在 ``no_grad`` 下跑，每块的分数
矩阵是临时量 —— 峰值是「一块」，分块有效。但 backward 里 ``recompute_fn`` 会**带
grad** 重跑同一段代码，于是每次 SDPA 调用都要为自己的反向保留 attention weights，
61 次调用的 61 份同时活着，加起来正好等于不分块的整块。

用法（单卡即可，不需要双卡）：
    python tools/measure_attn_backward_peak.py
显存不够跑满配置时它会自动降 S_k，仍能看出趋势。
"""

from __future__ import annotations

import sys

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, ".")

from modeling.krea2.krea2_modeling import _chunked_masked_attention  # noqa: E402

GIB = 1024 ** 3


def measure(b: int, h: int, s: int, d: int, chunk: int | None, device) -> tuple[float, float]:
    """返回 ``(forward 峰值 GiB, backward 峰值 GiB)``。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    q = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(b, 1, 1, s, dtype=torch.bool, device=device)
    mask[0, ..., -7:] = False          # 制造真实 padding，逼 SDPA 走 math 后端
    base = torch.cuda.max_memory_allocated(device)

    # mask 走参数而不是闭包捕获：末尾的 del 会让闭包捕获的名字变得可疑（ruff F821
    # 会报，而且真要是被 del 到就是运行时错误）。
    def fn(qq, kk, vv, mm):
        if chunk is None:
            return torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, attn_mask=mm, dropout_p=0.0, is_causal=False,
            )
        return _chunked_masked_attention(qq, kk, vv, mm, chunk=chunk)

    torch.cuda.reset_peak_memory_stats(device)
    out = checkpoint(fn, q, k, v, mask, use_reentrant=False)
    torch.cuda.synchronize(device)
    fwd = (torch.cuda.max_memory_allocated(device) - base) / GIB

    torch.cuda.reset_peak_memory_stats(device)
    out.sum().backward()
    torch.cuda.synchronize(device)
    bwd = (torch.cuda.max_memory_allocated(device) - base) / GIB

    del q, k, v, out, mask
    torch.cuda.empty_cache()
    return fwd, bwd


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("需要加速卡")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    print(f"torch {torch.__version__}  hip={torch.version.hip}")
    print(f"cuda:0 空闲 {free / GIB:.1f} / {total / GIB:.1f} GiB\n")

    b, h, d = 2, 48, 128
    # 真机是 S=15600；整块要 87 GiB 装不下，所以缩到能同时跑「不分块」做对照的规模。
    # 趋势与绝对值无关 —— 要看的是 backward 峰值是否随 chunk 变化。
    s = 2048
    while b * h * s * s * 4 > free * 0.25 and s > 256:
        s //= 2
    print(f"实测规模：B={b} H={h} S={s} D={d}")
    print(f"不分块的整块分数矩阵 = B*H*S*S*4 = {b * h * s * s * 4 / GIB:.2f} GiB\n")

    print(f"{'chunk':>8} {'块数':>5} {'forward 峰值':>14} {'backward 峰值':>15}")
    print("-" * 48)
    rows = []
    for chunk in (128, 256, 512, 1024, None):
        try:
            fwd, bwd = measure(b, h, s, d, chunk, device)
        except torch.OutOfMemoryError:
            print(f"{str(chunk):>8} {'-':>5} {'OOM':>14} {'OOM':>15}")
            continue
        n = "-" if chunk is None else str(-(-s // chunk))
        label = "不分块" if chunk is None else str(chunk)
        print(f"{label:>8} {n:>5} {fwd:>11.2f} GiB {bwd:>12.2f} GiB")
        rows.append((label, fwd, bwd))

    if len(rows) >= 2:
        fwd_span = max(r[1] for r in rows) / max(min(r[1] for r in rows), 1e-9)
        bwd_span = max(r[2] for r in rows) / max(min(r[2] for r in rows), 1e-9)
        print(f"\nforward 峰值最大/最小 = {fwd_span:.1f}x   "
              f"backward 峰值最大/最小 = {bwd_span:.1f}x")
        print("若 forward 明显分散而 backward 几乎持平 → 分块只压得住前向，"
              "backward 的重算峰值与不分块相同。")


if __name__ == "__main__":
    main()
