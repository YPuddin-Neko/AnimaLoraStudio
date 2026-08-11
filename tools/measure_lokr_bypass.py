#!/usr/bin/env python
"""实测 LoKr 的 rebuild 与 bypass 两条 forward 路径：数值是否一致 + 各自多快。

背景
----
lycoris 的 `_rebuild_forward` 把 ΔW 物化成 (out,in) 稠密矩阵，再对它多跑一次全量
matmul —— 每个注入层约 2× FLOPs：

    base  = org_forward(x)          matmul #1
    delta = op(x, 稠密ΔW)            matmul #2
    return base + delta

`bypass_forward_diff` 走 Kronecker 恒等式 (A⊗B)vec(X) = vec(B X Aᵀ)，不物化 ΔW，
第二次 matmul 只需 1/factor 的量。理论上注入层从 2.0 单位降到 1 + 1/factor
（factor=8 → 1.125），即约 1.78× —— 但那是 FLOPs 账，真实收益取决于本机对
`(T, f, in/f) × (out/f, in/f)` 这种带 batch 维小矩阵乘的效率。所以要实测。

用法
----
    python tools/measure_lokr_bypass.py                     # 默认 Krea2 的 MLP up 层
    python tools/measure_lokr_bypass.py --factor 4 --tokens 4096
    python tools/measure_lokr_bypass.py --full-matrix       # 用户生产配置那一支
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn as nn


def build(bypass: bool, *, in_dim: int, out_dim: int, factor: int,
          full_matrix: bool, device: str, dtype: torch.dtype, seed: int = 0):
    """造一个注入好的 LokrModule。返回 (被 patch 的 linear, lokr 模块)。"""
    from lycoris.modules.lokr import LokrModule

    torch.manual_seed(seed)
    linear = nn.Linear(in_dim, out_dim, bias=False).to(device=device, dtype=dtype)
    mod = LokrModule(
        lora_name="bench",
        org_module=linear,
        multiplier=1.0,
        # 超大 lora_dim 触发 use_w2（第二块不分解）—— 生产配置是 1145141919
        lora_dim=10**9 if full_matrix else 32,
        alpha=1 if full_matrix else 32,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        factor=factor,
        bypass_mode=bypass,
    )
    mod.apply_to()
    mod.to(device=device)
    return linear, mod


def lokr_param_names(mod) -> list[str]:
    return [
        n for n in ("lokr_w1", "lokr_w1_a", "lokr_w1_b",
                    "lokr_w2", "lokr_w2_a", "lokr_w2_b")
        if getattr(mod, n, None) is not None
    ]


def sync(src, dst, lin_src, lin_dst, std: float = 0.02) -> None:
    """两个模块的 base 权重与全部 lokr 参数对齐，并搅活零初始化的那块。

    LoKr 默认零初始化 w2（或 w2_b）；不搅动的话 ΔW≡0，两条路径平凡相等，
    等价性检查就是空过的。
    """
    with torch.no_grad():
        lin_dst.weight.copy_(lin_src.weight)
        for n in lokr_param_names(src):
            p = getattr(src, n)
            p.normal_(std=std)
            getattr(dst, n).copy_(p)


def timed(fn, *, warmup: int, iters: int, device: str) -> float:
    """返回单次平均耗时（毫秒）。"""
    for _ in range(warmup):
        fn()
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device != "cpu":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    # 默认取 Krea2 的 MLP up 层（OOM 日志里那一层）
    ap.add_argument("--in-dim", type=int, default=36864)
    ap.add_argument("--out-dim", type=int, default=6912)
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=2048,
                    help="序列长度。生产是 15600（2624x1472），默认取小值以便快速跑完")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--full-matrix", action="store_true",
                    help="触发 use_w2（生产配置那一支）")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip-backward", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    if device == "cpu":
        print("警告：没有可用加速卡，CPU 上的相对耗时没有参考价值\n")

    print(f"层    : ({args.out_dim}, {args.in_dim})   factor={args.factor}   "
          f"full_matrix={args.full_matrix}")
    print(f"输入  : ({args.batch}, {args.tokens}, {args.in_dim})   dtype={args.dtype}   device={device}")

    common = dict(in_dim=args.in_dim, out_dim=args.out_dim, factor=args.factor,
                  full_matrix=args.full_matrix, device=device, dtype=dtype)
    lin_bypass, mod_bypass = build(True, **common)
    lin_rebuild, mod_rebuild = build(False, **common)

    got_w2 = bool(getattr(mod_bypass, "use_w2", False))
    if got_w2 != args.full_matrix:
        print(f"\n注意：--full-matrix={args.full_matrix} 但实际 use_w2={got_w2}"
              f"（lycoris 按 rank 与分解后维度自行决定），测的是 use_w2={got_w2} 那一支")

    names = lokr_param_names(mod_bypass)
    print(f"参数  : {', '.join(names)}")
    for n in names:
        print(f"        {n:12} {tuple(getattr(mod_bypass, n).shape)}")

    sync(mod_bypass, mod_rebuild, lin_bypass, lin_rebuild)

    x = torch.randn(args.batch, args.tokens, args.in_dim, device=device, dtype=dtype)

    # ── 1. 数值等价 ────────────────────────────────────────────────────────
    print("\n=== 1/3 数值等价（eval，无 dropout）===")
    mod_bypass.eval()
    mod_rebuild.eval()
    with torch.no_grad():
        out_b = lin_bypass(x)
        out_r = lin_rebuild(x)
    # bf16 下容差必须放宽：两条路径累加长度不同（bypass 分两步、rebuild 一次），
    # 差异来自浮点累加顺序而非算法。fp32 下应到 1e-5。
    atol = 1e-5 if dtype == torch.float32 else 2e-2
    diff = (out_b.float() - out_r.float()).abs()
    rel = diff.max().item() / max(out_r.float().abs().max().item(), 1e-12)
    print(f"max|Δ| = {diff.max().item():.3e}   相对 = {rel:.3e}   容差 = {atol:.0e}")
    if out_b.float().abs().max().item() == 0:
        print("!! 输出恒零，等价性检查无意义（ΔW 没搅活）")
        return 1
    ok_fwd = torch.allclose(out_b.float(), out_r.float(), atol=atol, rtol=atol)
    print("forward 等价 " + ("OK" if ok_fwd else "**不一致**"))

    # ── 2. 梯度等价 ────────────────────────────────────────────────────────
    ok_bwd = True
    if not args.skip_backward:
        print("\n=== 2/3 梯度等价 ===")
        mod_bypass.train()
        mod_rebuild.train()
        target = torch.randn_like(out_r)
        lin_bypass(x).sub(target).pow(2).mean().backward()
        lin_rebuild(x).sub(target).pow(2).mean().backward()
        for n in names:
            gb = getattr(mod_bypass, n).grad
            gr = getattr(mod_rebuild, n).grad
            if gb is None or gr is None:
                print(f"  {n:12} 缺梯度 bypass={gb is not None} rebuild={gr is not None}")
                ok_bwd = False
                continue
            if gb.abs().sum().item() == 0:
                print(f"  {n:12} 梯度恒零，这一项没测到东西")
                continue
            d = (gb.float() - gr.float()).abs().max().item()
            scale = max(gr.float().abs().max().item(), 1e-12)
            same = torch.allclose(gb.float(), gr.float(), atol=atol, rtol=atol)
            print(f"  {n:12} max|Δgrad| = {d:.3e}  相对 = {d / scale:.3e}  "
                  + ("OK" if same else "**不一致**"))
            ok_bwd = ok_bwd and same
        for m in (mod_bypass, mod_rebuild):
            for n in names:
                getattr(m, n).grad = None
    else:
        print("\n=== 2/3 梯度等价：--skip-backward 跳过 ===")

    # ── 3. 耗时 ───────────────────────────────────────────────────────────
    print(f"\n=== 3/3 耗时（warmup={args.warmup} iters={args.iters}）===")
    mod_bypass.eval()
    mod_rebuild.eval()

    def fwd(lin):
        def run():
            with torch.no_grad():
                lin(x)
        return run

    ms_r = timed(fwd(lin_rebuild), warmup=args.warmup, iters=args.iters, device=device)
    ms_b = timed(fwd(lin_bypass), warmup=args.warmup, iters=args.iters, device=device)
    print(f"  forward  rebuild {ms_r:8.2f} ms   bypass {ms_b:8.2f} ms   "
          f"加速 {ms_r / ms_b:.2f}x")

    if not args.skip_backward:
        mod_bypass.train()
        mod_rebuild.train()

        def fwd_bwd(lin, mod):
            def run():
                for n in names:
                    getattr(mod, n).grad = None
                lin(x).pow(2).mean().backward()
            return run

        ms_r2 = timed(fwd_bwd(lin_rebuild, mod_rebuild),
                      warmup=args.warmup, iters=args.iters, device=device)
        ms_b2 = timed(fwd_bwd(lin_bypass, mod_bypass),
                      warmup=args.warmup, iters=args.iters, device=device)
        print(f"  fwd+bwd  rebuild {ms_r2:8.2f} ms   bypass {ms_b2:8.2f} ms   "
              f"加速 {ms_r2 / ms_b2:.2f}x")

    # ── 理论对照 ──────────────────────────────────────────────────────────
    f = args.factor
    if args.in_dim % f == 0 and args.out_dim % f == 0:
        dense = args.out_dim * args.in_dim
        kron = (args.in_dim // f) * (args.out_dim // f) * f + (args.out_dim // f) * f * f
        print(f"\n理论（逐 token MAC）：稠密 delta {dense:,} / kron delta {kron:,} "
              f"= {dense / kron:.2f}x")
        print(f"注入层整体 rebuild 2.0 单位 → bypass {1 + 1 / f:.3f} 单位 "
              f"= {2.0 / (1 + 1 / f):.2f}x 上限（含省不掉的基座 matmul）")

    print()
    if ok_fwd and ok_bwd:
        print("数值等价通过 —— 切 bypass 不改变产出。")
        return 0
    print("**数值不等价** —— 不要切 bypass，先查原因。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
