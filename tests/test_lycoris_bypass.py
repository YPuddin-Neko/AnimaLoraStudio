"""lora / lokr 默认走 bypass_mode 的等价性 + DoRA / rank_dropout / LoHa 的 rebuild guard。

背景：lycoris 的 rebuild forward 把 ΔW 物化成 (out,in) 稠密矩阵，再对它多跑一次
全量 matmul —— 即每个注入层 ~2× FLOPs：

    _rebuild_forward:  base  = org_forward(x)        ← matmul #1
                       delta = op(x, 稠密ΔW)          ← matmul #2
                       return base + delta

不是一直如此：lycoris ac2616f (2025-10-04) 之前 rebuild 是合并权重后**一次** matmul，
那次为 stacked wrapper 的重构把它改成了两次。upstream 未把它当性能问题 ——
Network-Args.md 把 bypass_mode 定位成量化功能（"Designed for bnb 8bit/4bit linear layer"），
只在 FP8 / QuantLinears / 非 Linear 类时自动开（base.py:227-243）。普通 LoRA 的同一问题
有人报过（issue #182），LoKr 没有。

三种 algo 的 bypass 实现差别很大，这决定了本仓给谁开：
  lora: org_forward(x) + lora_up(lora_down(x)) * scale，低秩两次小 matmul。→ 开
  lokr: Kronecker 恒等式 (A⊗B)vec(X)=vec(B X Aᵀ)，不物化 ΔW，第二次 matmul 只需
        1/factor 的量（factor=8 时注入层 2.0 → 1.125 单位）。→ 开
  loha: bypass_forward_diff 内部照样调 get_weight() 物化稠密 ΔW（loha.py:296），
        只省掉合并进基座那一步，开了没收益。→ 不开

本文件验证：
1)  lora algo (LoCon) 下，bypass=True 与 bypass=False 的 forward / backward 数值等价
1b) lokr 下同样的等价性，低秩与 full matrix 两个分支各测一遍，外加
    「bypass 不得调 get_weight()」的性能属性守卫
2)  AnimaLycorisAdapter(algo='lora' / 'lokr') 默认 bypass_mode=True
3)  DoRA(weight_decompose=True) 强制 bypass_mode=False，避免 lycoris bypass 路径
    不走 wd 分支导致的静默失效
4)  lokr + rank_dropout 强制 rebuild —— bypass 不施加 rank_drop（base.py:255-258），
    会静默失效；lora 不受此限（LoConModule 的 bypass 走 Brank_drop(AX)）
5)  LoHa 保持 bypass_mode=False（默认 rebuild，行为不变）
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from utils.lycoris_adapter import AnimaLycorisAdapter
from training.families.anima.preset import ANIMA_PRESET
from training.families.krea2.preset import KREA2_PRESET
from training.families.krea2.quant_fp8 import patch_fp8_linears

pytest.importorskip("lycoris")


class MockDiT(nn.Module):
    """对齐 ANIMA_PRESET 的 target_name（*q_proj/*k_proj/*v_proj/*output_proj/*mlp.layer1/2）"""

    def __init__(self, d: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.output_proj = nn.Linear(d, d, bias=False)


def _bypass_modes(adapter: AnimaLycorisAdapter) -> list[bool]:
    return [bool(getattr(m, "bypass_mode", False)) for m in adapter.network.loras]


# ─── (1) LoCon 数值等价：bypass=True vs bypass=False ──────────────────────────


def _build_lora_module(bypass: bool, seed: int = 0):
    """直接造一个 LoConModule，避开 LycorisNetwork 的 preset/target 匹配"""
    from lycoris.modules.locon import LoConModule

    torch.manual_seed(seed)
    linear = nn.Linear(64, 64, bias=False)
    mod = LoConModule(
        lora_name="test",
        org_module=linear,
        multiplier=1.0,
        lora_dim=8,
        alpha=8,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        bypass_mode=bypass,
    )
    mod.apply_to()
    return linear, mod


def _copy_lora_weights(src, dst) -> None:
    """把 src 的 lora_up/down 权重塞进 dst（同 shape）"""
    dst.lora_up.weight.data.copy_(src.lora_up.weight.data)
    dst.lora_down.weight.data.copy_(src.lora_down.weight.data)


def test_locon_bypass_vs_rebuild_forward_equivalent() -> None:
    """同样权重、同样输入：bypass 路径与 rebuild 路径 forward 输出数值一致。

    LoRA paper 的 W'X = WX + BAX，bypass 是直接算右式；rebuild 是先 W'=W+BA 再算 W'X。
    fp32 下应严格一致到 ~1e-5 量级。
    """
    linear_a, mod_bypass = _build_lora_module(bypass=True, seed=0)
    linear_b, mod_rebuild = _build_lora_module(bypass=False, seed=0)
    # base linear 权重已经因为 seed=0 一致；同步 lora 部分
    _copy_lora_weights(mod_bypass, mod_rebuild)
    # 让 lora_up 不为 0（默认 init 是 0）才能真正测到 lora 路径
    with torch.no_grad():
        mod_bypass.lora_up.weight.normal_(std=0.1)
        mod_rebuild.lora_up.weight.copy_(mod_bypass.lora_up.weight)

    # 同步 base linear 权重以防 seed 之外有差异
    with torch.no_grad():
        linear_b.weight.copy_(linear_a.weight)

    mod_bypass.eval()
    mod_rebuild.eval()
    x = torch.randn(2, 16, 64)
    out_bypass = linear_a(x)
    out_rebuild = linear_b(x)
    assert torch.allclose(out_bypass, out_rebuild, atol=1e-5, rtol=1e-5)


def test_locon_bypass_vs_rebuild_backward_equivalent() -> None:
    """同样 loss：两条路径在 lora_up/lora_down 上的梯度数值一致。"""
    linear_a, mod_bypass = _build_lora_module(bypass=True, seed=1)
    linear_b, mod_rebuild = _build_lora_module(bypass=False, seed=1)
    _copy_lora_weights(mod_bypass, mod_rebuild)
    with torch.no_grad():
        mod_bypass.lora_up.weight.normal_(std=0.1)
        mod_rebuild.lora_up.weight.copy_(mod_bypass.lora_up.weight)
        linear_b.weight.copy_(linear_a.weight)

    mod_bypass.train()
    mod_rebuild.train()
    x = torch.randn(2, 16, 64, requires_grad=False)
    target = torch.randn(2, 16, 64)

    loss_bypass = (linear_a(x) - target).pow(2).mean()
    loss_rebuild = (linear_b(x) - target).pow(2).mean()
    assert torch.allclose(loss_bypass, loss_rebuild, atol=1e-5)

    loss_bypass.backward()
    loss_rebuild.backward()

    assert torch.allclose(
        mod_bypass.lora_up.weight.grad,
        mod_rebuild.lora_up.weight.grad,
        atol=1e-5, rtol=1e-5,
    )
    assert torch.allclose(
        mod_bypass.lora_down.weight.grad,
        mod_rebuild.lora_down.weight.grad,
        atol=1e-5, rtol=1e-5,
    )


# ─── (1b) LoKr 数值等价：bypass=True vs bypass=False ─────────────────────────
#
# 这几条是「切 bypass 不改变产出质量」这个结论的**唯一**依据 —— 不是推导，是断言。
# LoKr 的 bypass 走 Kronecker 恒等式而非低秩 up/down，所以必须单独测，
# 不能靠上面 LoCon 那两条覆盖。


def _build_lokr_module(bypass: bool, seed: int = 0, *, full_matrix: bool = False,
                       in_dim: int = 64, out_dim: int = 64, factor: int = 4):
    """直接造一个 LokrModule，避开 LycorisNetwork 的 preset/target 匹配。

    full_matrix=True 用超大 lora_dim 触发 use_w2（第二块不分解）—— 这正是用户
    生产配置（lora_dim=1145141919）走的分支，与低秩分支是 bypass 里的两条不同代码路径
    （lokr.py:476 的 if self.use_w2），必须分别测。
    """
    from lycoris.modules.lokr import LokrModule

    torch.manual_seed(seed)
    linear = nn.Linear(in_dim, out_dim, bias=False)
    mod = LokrModule(
        lora_name="test",
        org_module=linear,
        multiplier=1.0,
        lora_dim=10**9 if full_matrix else 4,
        alpha=1 if full_matrix else 4,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        factor=factor,
        bypass_mode=bypass,
    )
    mod.apply_to()
    return linear, mod


def _lokr_param_names(mod) -> list[str]:
    """LoKr 的参数名随 use_w1/use_w2 变，取实际存在的那些。"""
    return [
        n for n in ("lokr_w1", "lokr_w1_a", "lokr_w1_b",
                    "lokr_w2", "lokr_w2_a", "lokr_w2_b")
        if getattr(mod, n, None) is not None
    ]


def _sync_lokr(src, dst, linear_src, linear_dst, std: float = 0.1) -> None:
    """把两个模块的 base 权重与全部 LoKr 参数对齐，并把零初始化的那块搅活。

    LoKr 默认把 w2（或 w2_b）零初始化，不搅动的话 ΔW≡0，两条路径会平凡相等 ——
    测试就会在真正有分歧时也通过。
    """
    names = _lokr_param_names(src)
    assert names, "LokrModule 应至少有一组 lokr_w* 参数"
    with torch.no_grad():
        linear_dst.weight.copy_(linear_src.weight)
        for n in names:
            p = getattr(src, n)
            p.normal_(std=std)
            getattr(dst, n).copy_(p)


@pytest.mark.parametrize("full_matrix", [False, True], ids=["low_rank", "full_matrix"])
def test_lokr_bypass_vs_rebuild_forward_equivalent(full_matrix: bool) -> None:
    """同权重同输入：Kronecker 结构化 forward 与稠密 rebuild forward 数值一致。

    数学上是同一个求和式的两种算法：
      rebuild:  y = (W + kron(w1,w2)) x
      bypass:   y = W x + vec(w2 · X · w1ᵀ)
    展开后逐元素相同（Kronecker 定义），差异只来自浮点累加顺序。
    """
    linear_a, mod_bypass = _build_lokr_module(bypass=True, seed=0, full_matrix=full_matrix)
    linear_b, mod_rebuild = _build_lokr_module(bypass=False, seed=0, full_matrix=full_matrix)
    assert bool(getattr(mod_bypass, "use_w2", False)) is full_matrix, (
        "full_matrix 开关没能命中 use_w2 分支，测试就没测到目标代码路径"
    )
    _sync_lokr(mod_bypass, mod_rebuild, linear_a, linear_b)

    mod_bypass.eval()
    mod_rebuild.eval()
    x = torch.randn(2, 16, 64)
    out_bypass = linear_a(x)
    out_rebuild = linear_b(x)
    # ΔW 必须真的非零，否则这条测试是空过的
    assert not torch.allclose(out_bypass, linear_a.weight.new_zeros(out_bypass.shape))
    assert torch.allclose(out_bypass, out_rebuild, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("full_matrix", [False, True], ids=["low_rank", "full_matrix"])
def test_lokr_bypass_vs_rebuild_backward_equivalent(full_matrix: bool) -> None:
    """同 loss：两条路径在全部 lokr_w* 上的梯度数值一致。

    forward 等价不足以保证训练等价 —— 梯度才是真正决定权重更新的东西。
    """
    linear_a, mod_bypass = _build_lokr_module(bypass=True, seed=1, full_matrix=full_matrix)
    linear_b, mod_rebuild = _build_lokr_module(bypass=False, seed=1, full_matrix=full_matrix)
    _sync_lokr(mod_bypass, mod_rebuild, linear_a, linear_b)

    mod_bypass.train()
    mod_rebuild.train()
    x = torch.randn(2, 16, 64)
    target = torch.randn(2, 16, 64)

    loss_bypass = (linear_a(x) - target).pow(2).mean()
    loss_rebuild = (linear_b(x) - target).pow(2).mean()
    assert torch.allclose(loss_bypass, loss_rebuild, atol=1e-5)

    loss_bypass.backward()
    loss_rebuild.backward()

    names = _lokr_param_names(mod_bypass)
    for n in names:
        g_bypass = getattr(mod_bypass, n).grad
        g_rebuild = getattr(mod_rebuild, n).grad
        assert g_bypass is not None, f"{n} 在 bypass 路径上没拿到梯度"
        assert g_rebuild is not None, f"{n} 在 rebuild 路径上没拿到梯度"
        # 梯度不能恒零，否则等价性是平凡的
        assert g_bypass.abs().sum() > 0, f"{n} 的梯度恒零，测试没测到东西"
        assert torch.allclose(g_bypass, g_rebuild, atol=1e-5, rtol=1e-5), (
            f"{n} 的梯度在两条路径间不一致"
        )


def test_lokr_bypass_scale_and_multiplier_carried() -> None:
    """alpha/rank 缩放与 multiplier 在 bypass 路径上同样施加。

    bypass 的缩放在 lokr.py:548-553（h * self.scale * scale * self.scalar），
    rebuild 的在 make_kron 内部 + 561 行的 * self.scalar —— 两处独立实现，
    漏一个会让 LoRA 强度整体偏移，且 forward 等价测试用默认 multiplier=1 抓不到。
    """
    linear_a, mod_bypass = _build_lokr_module(bypass=True, seed=2)
    linear_b, mod_rebuild = _build_lokr_module(bypass=False, seed=2)
    _sync_lokr(mod_bypass, mod_rebuild, linear_a, linear_b)
    mod_bypass.multiplier = 2.5
    mod_rebuild.multiplier = 2.5
    mod_bypass.eval()
    mod_rebuild.eval()

    x = torch.randn(2, 16, 64)
    assert torch.allclose(linear_a(x), linear_b(x), atol=1e-5, rtol=1e-5)


def test_lokr_bypass_does_not_materialize_dense_delta() -> None:
    """bypass 路径不得调用 get_weight() —— 那是物化稠密 ΔW 的入口。

    这条测的是**性能属性**而非数值：等价性测试无法区分「用 Kronecker 结构算」和
    「物化 ΔW 后再算」，而后者正是 LoHa 的 bypass 实现（loha.py:296 调 get_weight），
    开了没有收益。哪天 LoKr 的 bypass 被改成同样的写法，这条会失败。
    """
    linear, mod = _build_lokr_module(bypass=True, seed=3, full_matrix=True)
    calls = []
    original = mod.get_weight
    mod.get_weight = lambda *a, **k: (calls.append(1), original(*a, **k))[1]  # type: ignore[method-assign]

    mod.eval()
    linear(torch.randn(2, 4, 64))
    assert not calls, (
        "bypass forward 调用了 get_weight()，说明它在物化稠密 ΔW —— "
        "factor 倍的 FLOPs 收益已经没了"
    )


# ─── (2-4) AnimaLycorisAdapter 按 algo + DoRA 自动选 bypass_mode ──────────────


def test_adapter_lora_defaults_to_bypass_mode() -> None:
    """algo='lora' 不开 DoRA → 全部模块 bypass_mode=True（issue #182 默认快路径）"""
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lora", rank=8, alpha=8)
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes, "preset 应该至少匹配一个 q/k/v/output_proj"
    assert all(modes), f"lora algo 全部模块应走 bypass，但得到 {modes}"


def test_adapter_lora_with_dora_forces_rebuild() -> None:
    """algo='lora' + lora_dora=True：DoRA 数学上必须 rebuild，guard 不能让 bypass 静默吞掉 wd 分支"""
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, 
        algo="lora", rank=8, alpha=8, weight_decompose=True,
    )
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"DoRA 必须走 rebuild，但 bypass_mode={modes}"


def test_adapter_lokr_defaults_to_bypass_mode() -> None:
    """algo='lokr' 不开 DoRA / rank_dropout → 全部模块 bypass_mode=True。

    LoKr 的 bypass 是 Kronecker 恒等式 (A⊗B)vec(X)=vec(B X Aᵀ)，不物化稠密 ΔW，
    第二次 matmul 只需 1/factor 的量。等价性由本文件的 forward/backward 测试锁住。
    """
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr", rank=8, alpha=8, factor=8)
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes, "preset 应该至少匹配一个 q/k/v/output_proj"
    assert all(modes), f"lokr 应走 bypass，但 bypass_mode={modes}"


def test_adapter_lokr_with_dora_forces_rebuild() -> None:
    """lokr + DoRA：bypass forward 不走 wd 分支，开了会让 DoRA 静默失效。

    lycoris docs/Network-Args.md "Weight Decompose" 也写明它强制 bypass_mode=False。
    """
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(
        preset=ANIMA_PRESET, algo="lokr", rank=8, alpha=8, factor=8,
        weight_decompose=True,
    )
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"lokr + DoRA 必须走 rebuild，但 bypass_mode={modes}"


def test_adapter_lokr_with_rank_dropout_forces_rebuild() -> None:
    """lokr + rank_dropout：两条路径语义不同，bypass 会让 rank_dropout 静默失效。

    lycoris base.py:255-258 自己写着：
        g(x) = WX + drop(ΔWX)         非 LoCon, bypass    ← 不施加 rank_drop
        g(x) = (W + rank_drop(ΔW))X   非 LoCon, rebuild   ← 施加
    bypass_forward_diff 只调 self.drop、不调 self.rank_drop，也不走 get_weight()
    （rank_dropout 逻辑在 lokr.py:375-380 那里面）。所以开了必须留在 rebuild。

    这是「静默失效」类 bug 的守卫：错了不报错、不 OOM，只是正则化没生效。
    """
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(
        preset=ANIMA_PRESET, algo="lokr", rank=8, alpha=8, factor=8,
        rank_dropout=0.1,
    )
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"lokr + rank_dropout 必须走 rebuild，但 bypass_mode={modes}"


def test_adapter_lora_with_rank_dropout_still_bypasses() -> None:
    """lora + rank_dropout 仍走 bypass —— LoConModule 的 bypass 走 Brank_drop(AX)，
    rank_drop 照样施加（base.py:255 第一条），所以上面那条限制只对 lokr 成立。
    漏掉这条会让 rank_dropout 的 guard 顺手把 lora 也拖回慢路径。
    """
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(
        preset=ANIMA_PRESET, algo="lora", rank=8, alpha=8, rank_dropout=0.1,
    )
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert all(modes), f"lora + rank_dropout 应仍走 bypass，但 bypass_mode={modes}"


def test_adapter_lokr_fp8_base_forces_bypass_and_trains() -> None:
    """Monkeypatched FP8 nn.Linear uses bypass and full-precision LoKr params."""
    torch.manual_seed(0)
    model = MockDiT(d=16)
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.requires_grad_(False)
            module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
            scales[name] = torch.tensor(0.5)
    patch_fp8_linears(model, scales)

    adapter = AnimaLycorisAdapter(
        preset=KREA2_PRESET, algo="lokr", rank=4, alpha=4, factor=4,
    )
    adapter.inject(model)

    modes = _bypass_modes(adapter)
    assert modes and all(modes)
    params = adapter.get_params()
    assert params and all(p.dtype == torch.float32 for p in params)

    output = model.q_proj(torch.randn(2, 3, 16))
    output.square().mean().backward()
    grads = [p.grad for p in params if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_adapter_loha_keeps_rebuild() -> None:
    """algo='loha' 行为不变（LoHa bypass 内部仍 rebuild，开了反而慢；lycoris changelog 原话）"""
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="loha", rank=8, alpha=8)
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"loha 应保持 rebuild，但 bypass_mode={modes}"
