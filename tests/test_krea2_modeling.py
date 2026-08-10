"""Krea2 modeling structure, forward contract, and ComfyUI key compatibility."""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

import modeling.krea2.krea2_modeling as krea2_modeling
from modeling.krea2 import KREA2_CONFIG, Attention, Krea2Config, SingleStreamDiT


def _tiny_config() -> Krea2Config:
    return Krea2Config(
        features=64,
        tdim=16,
        txtdim=32,
        heads=4,
        kvheads=2,
        multiplier=2,
        layers=2,
        patch=2,
        channels=4,
        txtlayers=3,
        txtheads=4,
        txtkvheads=2,
    )


def _tiny_inputs():
    x = torch.randn(2, 4, 1, 5, 7)
    timesteps = torch.tensor([0.2, 0.8])
    context = torch.randn(2, 5, 3, 32)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    return x, timesteps, context, mask


def test_public_config_matches_krea2_checkpoint_architecture() -> None:
    assert KREA2_CONFIG == Krea2Config()
    assert KREA2_CONFIG.features == 6144
    assert KREA2_CONFIG.heads == 48
    assert KREA2_CONFIG.kvheads == 12
    assert KREA2_CONFIG.layers == 28
    assert KREA2_CONFIG.txtlayers == 12
    assert KREA2_CONFIG.patch == 2
    assert KREA2_CONFIG.channels == 16


def test_full_meta_model_has_expected_size_linear_count_and_gqa_shapes() -> None:
    with torch.device("meta"):
        model = SingleStreamDiT()

    assert sum(parameter.numel() for parameter in model.parameters()) == 12_820_073_036
    assert sum(isinstance(module, torch.nn.Linear) for module in model.modules()) == 264
    state = model.state_dict()
    assert len(state) == 430
    assert state["blocks.0.attn.wq.weight"].shape == (6144, 6144)
    assert state["blocks.0.attn.wk.weight"].shape == (1536, 6144)
    assert state["blocks.0.attn.wv.weight"].shape == (1536, 6144)
    assert state["blocks.0.attn.gate.weight"].shape == (6144, 6144)


def test_parameter_paths_flatten_to_comfyui_kohya_lora_keys() -> None:
    with torch.device("meta"):
        state_keys = set(SingleStreamDiT().state_dict())

    parameter_paths = [
        "blocks.0.attn.wq.weight",
        "blocks.0.attn.gate.weight",
        "blocks.0.mlp.up.weight",
        "txtfusion.refiner_blocks.0.attn.wo.weight",
        "txtmlp.1.weight",
    ]
    for path in parameter_paths:
        assert path in state_keys

    flattened = {
        "lora_unet_" + path.removesuffix(".weight").replace(".", "_")
        for path in parameter_paths
    }
    assert flattened == {
        "lora_unet_blocks_0_attn_wq",
        "lora_unet_blocks_0_attn_gate",
        "lora_unet_blocks_0_mlp_up",
        "lora_unet_txtfusion_refiner_blocks_0_attn_wo",
        "lora_unet_txtmlp_1",
    }


def test_tiny_forward_preserves_5d_shape_and_crops_patch_padding() -> None:
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    with torch.no_grad():
        output = model(x, timesteps, context, mask)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_batched_forward_matches_per_sample_forward() -> None:
    """Modulation is broadcast per sample, never across the token axis.

    ``tproj`` emits (B, 1, 6*features); a missing token axis would broadcast
    (B, features) against (B, L, features) and silently pass only when B == 1.
    """
    torch.manual_seed(11)
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    with torch.no_grad():
        batched = model(x, timesteps, context, mask)
        per_sample = torch.cat(
            [
                model(
                    x[i : i + 1],
                    timesteps[i : i + 1],
                    context[i : i + 1],
                    mask[i : i + 1],
                )
                for i in range(x.shape[0])
            ]
        )
    torch.testing.assert_close(batched, per_sample, rtol=1e-5, atol=1e-5)


def test_flattened_and_layered_text_context_are_equivalent() -> None:
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    flattened = context.flatten(2)
    with torch.no_grad():
        layered_out = model(x, timesteps, context, mask)
        flattened_out = model(x, timesteps, flattened, mask)
    torch.testing.assert_close(layered_out, flattened_out)


def test_padding_mask_prevents_padded_text_from_affecting_image_output() -> None:
    torch.manual_seed(7)
    model = SingleStreamDiT(_tiny_config()).eval()
    for block in model.blocks:
        block.mod.lin.data.fill_(0.1)

    x, timesteps, context, mask = _tiny_inputs()
    changed = context.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1000
    with torch.no_grad():
        original_out = model(x, timesteps, context, mask)
        changed_out = model(x, timesteps, changed, mask)
    torch.testing.assert_close(original_out, changed_out, rtol=1e-5, atol=1e-5)


def test_chunked_masked_attention_matches_unchunked() -> None:
    """按 query 分块的注意力必须与不分块**逐值相同**。

    分块的理由是显存：带 attn_mask 时 SDPA 只能走 math 后端（flash 不支持任意
    mask、mem-efficient 在海光 DTK 上没编译），而 math 会 materialize
    ``[B, H, S_q, S_k]`` —— Krea2 在 2048px 桶上 S≈16.9k、48 heads，bs=2 就是
    102 GiB，64GB 卡必 OOM（bs=1 也要 51 GiB，同样装不下）。

    分块在**数学上**精确：softmax 沿 key 维归一化，每个 query 行的输出只依赖该行
    自己的分数向量，query 之间无耦合，不像 flash 那样需要 online-softmax rescale。

    但**实现上不是 bit-exact**。真机（BW1000 / DTK / torch 2.5.1）实测最大绝对差
    4.17e-07 ≈ 3.5 个 fp32 eps —— SDPA 对不同 ``s_q`` 会选不同的 kernel tile /
    累加顺序，而浮点加法不满足结合律。所以容差取 fp32 量级而非 0。

    容差仍然足够严：真正的切法错误误差是 O(0.1~1)，比这里的阈值大四五个数量级
        沿 key 切  → softmax 分母错   → O(1)
        块边界漏算 → 整行为 0/未初始化 → O(1)
        mask 对不上 → padding 参与注意力 → O(0.1~1)
    ``atol`` 兜住接近 0 的元素（那里 rtol 无意义 —— 实测 rel diff 2.9e-04 就出现在
    这种元素上），``rtol`` 兜住大值。
    """
    from modeling.krea2.krea2_modeling import _chunked_masked_attention

    torch.manual_seed(3)
    b, h, s_q, s_k, d = 2, 4, 37, 29, 16
    q = torch.randn(b, h, s_q, d)
    k = torch.randn(b, h, s_k, d)
    v = torch.randn(b, h, s_k, d)
    # [B, 1, 1, S_k] key-padding mask，含 False（否则测不到 mask 复用是否正确）
    mask = torch.ones(b, 1, 1, s_k, dtype=torch.bool)
    mask[0, ..., -7:] = False
    mask[1, ..., -3:] = False

    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
    )
    # 多个块大小都要等价，含 s_q 整除 / 有余数 / 大于 s_q 三种情形
    for chunk in (5, 8, 16, 37, 64):
        got = _chunked_masked_attention(q, k, v, mask, chunk=chunk)
        assert got.shape == reference.shape, f"chunk={chunk} 形状不对"
        torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-5)


def test_chunked_attention_rows_are_independent_of_chunk_placement() -> None:
    """同一 query 行落在块首 / 块中 / 块尾都得到同一结果（fp32 量级内）。

    这条比「与不分块比较」更能抓切法错误。若实现误把块内位置当成了绝对位置
    （典型：mask 也跟着切、或 RoPE 偏移按块内 index 算），那么同一行在不同块布局下
    的输出就会不同 —— 而与不分块的整体比较可能因为误差被平均而看不出来。

    做法：用互质的块大小（3 / 7 / 11）让每一行在三次运行里落到不同的块内位置。
    """
    from modeling.krea2.krea2_modeling import _chunked_masked_attention

    torch.manual_seed(13)
    b, h, s_q, s_k, d = 1, 2, 23, 17, 8
    q = torch.randn(b, h, s_q, d)
    k = torch.randn(b, h, s_k, d)
    v = torch.randn(b, h, s_k, d)
    mask = torch.ones(b, 1, 1, s_k, dtype=torch.bool)
    mask[..., -5:] = False

    outs = [_chunked_masked_attention(q, k, v, mask, chunk=c) for c in (3, 7, 11)]
    for i in range(1, len(outs)):
        torch.testing.assert_close(outs[0], outs[i], rtol=1e-5, atol=1e-5)


def test_chunked_attention_shortcircuits_when_small() -> None:
    """query 数不超过 chunk 时直接走原路，不该产生分块开销。

    断言方式是行为等价 + 只调一次 SDPA：分块版对小输入必须与单次调用完全一致。
    """
    from modeling.krea2 import krea2_modeling as m

    torch.manual_seed(5)
    q = torch.randn(1, 2, 10, 8)
    k = torch.randn(1, 2, 10, 8)
    v = torch.randn(1, 2, 10, 8)
    mask = torch.ones(1, 1, 1, 10, dtype=torch.bool)
    mask[..., -2:] = False

    calls = {"n": 0}
    real = torch.nn.functional.scaled_dot_product_attention

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    orig = m.F.scaled_dot_product_attention
    m.F.scaled_dot_product_attention = counting
    try:
        out = m._chunked_masked_attention(q, k, v, mask, chunk=1024)
    finally:
        m.F.scaled_dot_product_attention = orig

    assert calls["n"] == 1, f"小输入不该分块，实际调了 {calls['n']} 次 SDPA"
    # 这里可以要求 bit-exact：短路路径就是原封不动的单次调用，同一 kernel、
    # 同一累加顺序，不存在分块那条路的浮点重排。
    torch.testing.assert_close(out, real(q, k, v, attn_mask=mask), rtol=0, atol=0)


def test_pick_query_chunk_falls_back_off_cuda() -> None:
    """CPU 张量拿不到显存读数，必须回落到保守常数而不是抛异常。

    选块只是个优化决策，任何一步失败都不该让前向崩掉。
    """
    from modeling.krea2 import krea2_modeling as m

    q = torch.randn(1, 4, 8, 16)
    assert not q.is_cuda
    assert m._pick_query_chunk(q, 8) == m._MASKED_ATTN_QUERY_CHUNK


GIB = 1024 ** 3


class _FakeCudaQuery:
    """自称在 cuda 上的 query 替身，配合 monkeypatch 伪造显存读数。"""

    is_cuda = True
    ndim = 4
    device = "cuda:0"

    def __init__(self, batch: int, heads: int) -> None:
        self.shape = (batch, heads, 4096, 128)


def _pick_with_free(
    monkeypatch: pytest.MonkeyPatch,
    free_bytes: int | None,
    batch: int,
    heads: int,
    s_k: int,
) -> int:
    """在伪造的空闲显存读数下跑一次选块；``free_bytes=None`` 表示读数抛异常。"""
    from modeling.krea2 import krea2_modeling as m

    def fake_mem_get_info(_device):
        if free_bytes is None:
            raise RuntimeError("hipErrorNoDevice")
        return (free_bytes, free_bytes * 2)

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)
    return m._pick_query_chunk(_FakeCudaQuery(batch, heads), s_k)


def test_pick_query_chunk_falls_back_when_mem_query_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mem_get_info`` 抛异常（驱动/后端差异）时回落到保守常数，不向上传播。

    选块只是优化决策，任何一步失败都不该让前向崩掉。
    """
    from modeling.krea2 import krea2_modeling as m

    got = _pick_with_free(monkeypatch, None, 2, 48, 16928)
    assert got == m._MASKED_ATTN_QUERY_CHUNK


def test_pick_query_chunk_scales_with_free_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """块大小随余量走，且选出的块确实装得进预算。

    这条锁住自适应的**动机**：真机上 full matrix 配置 OOM 时只剩 2.8 GiB，而常规
    低秩配置有 20 GiB 以上 —— 写死一个常数必然在一侧错（小了白丢吞吐，大了 OOM）。
    """
    from modeling.krea2 import krea2_modeling as m

    b, h, s_k = 2, 48, 16928  # 真机 2048px 桶
    tight = _pick_with_free(monkeypatch, 8 * GIB, b, h, s_k)
    loose = _pick_with_free(monkeypatch, 32 * GIB, b, h, s_k)
    assert tight < loose, f"余量大 4 倍却没选更大的块：{tight} vs {loose}"

    for free_gib, chunk in ((8, tight), (32, loose)):
        need = b * h * chunk * s_k * 4 * m._MATH_SDPA_OVERHEAD
        budget = free_gib * GIB * m._CHUNK_FREE_VRAM_FRACTION
        assert need <= budget, (
            f"free={free_gib}G 选了 chunk={chunk}，需 {need / GIB:.2f}G "
            f"超出预算 {budget / GIB:.2f}G"
        )


def test_pick_query_chunk_floor_may_exceed_fraction_but_not_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """余量紧到连一个最小块都超预算时，128 的下限优先于 fraction —— 但仍远小于余量。

    真机 full matrix 的 rank 1：OOM 那一刻只剩 2.8 GiB，按 0.5 的 fraction 算预算
    只有 1.40 GiB，而 128 块要 1.55 GiB。fraction 是安全边际不是硬上限，此时让下限
    赢是对的：1.55 < 2.80，仍然装得下。真装不下时该由 allocator 报 OOM，不该由选块
    函数返回 0 去做除零。
    """
    from modeling.krea2 import krea2_modeling as m

    b, h, s_k = 2, 48, 16928
    free = int(2.8 * GIB)
    chunk = _pick_with_free(monkeypatch, free, b, h, s_k)
    assert chunk == 128, f"紧余量下没落到下限：{chunk}"

    need = b * h * chunk * s_k * 4 * m._MATH_SDPA_OVERHEAD
    assert need > free * m._CHUNK_FREE_VRAM_FRACTION, "本用例要求下限压过 fraction"
    assert need < free, f"下限块要 {need / GIB:.2f}G，超过余量 {free / GIB:.2f}G"


def test_pick_query_chunk_accounts_for_batch_and_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """batch / head / S_k 任一翻倍，块大小相应减半 —— 漏掉任一维都会高估预算。

    分数矩阵是 ``[B, H, chunk, S_k]``。早先的公式漏了 batch，bs=2 时预算刚好高估
    一倍，正落在 OOM 那一侧。

    参数取成 2 的幂（per_query = 2 MiB，base = 2048），这样 128 对齐不会引入
    截断，减半关系可以用严格相等来断言。
    """
    base = _pick_with_free(monkeypatch, 8 * GIB, 1, 32, 8192)
    assert base == 2048, f"基准算错了，后面的减半断言无意义：{base}"
    assert _pick_with_free(monkeypatch, 8 * GIB, 2, 32, 8192) == base // 2
    assert _pick_with_free(monkeypatch, 8 * GIB, 1, 64, 8192) == base // 2
    assert _pick_with_free(monkeypatch, 8 * GIB, 1, 32, 16384) == base // 2


def test_pick_query_chunk_is_clamped_and_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """结果永远是 128 的倍数并夹在 ``[128, 4096]``。

    上限的理由：一次吃几 GiB 以上的分数矩阵没有意义 —— 余量真有那么多说明这一层
    本该走 flash 而不是 math。
    """
    starved = _pick_with_free(monkeypatch, 1024, 2, 48, 16928)   # 1 KiB 余量
    huge = _pick_with_free(monkeypatch, 4096 * GIB, 1, 1, 64)     # 荒谬的余量
    assert starved == 128, f"下限没夹住：{starved}"
    assert huge == 4096, f"上限没夹住：{huge}"
    for free_gib in (1, 2, 4, 7, 13, 20, 31):
        chunk = _pick_with_free(monkeypatch, free_gib * GIB, 2, 48, 16928)
        assert chunk % 128 == 0, f"free={free_gib}G 选了非 128 倍数：{chunk}"
        assert 128 <= chunk <= 4096


def test_chunked_masked_attention_defaults_to_adaptive_pick() -> None:
    """不传 ``chunk`` 时必须走 :func:`_pick_query_chunk`。

    防回归：helper 与选块函数分别测对了，但默认值若被改回写死常数，其余用例
    （都显式传 chunk）仍会全绿。
    """
    from modeling.krea2 import krea2_modeling as m

    torch.manual_seed(21)
    q = torch.randn(1, 2, 9, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)
    mask = torch.ones(1, 1, 1, 6, dtype=torch.bool)
    mask[..., -2:] = False

    seen: list[tuple[int, int]] = []
    orig = m._pick_query_chunk

    def spy(qq, s_k):
        seen.append((int(qq.shape[-3]), int(s_k)))
        return orig(qq, s_k)

    m._pick_query_chunk = spy
    try:
        out = m._chunked_masked_attention(q, k, v, mask)
    finally:
        m._pick_query_chunk = orig

    assert seen == [(2, 6)], f"默认路径没问选块函数，或参数不对：{seen}"
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
    )
    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-5)


def test_masked_forward_uses_chunked_path() -> None:
    """整模型前向在**有 padding** 时必须走分块路径。

    防回归：分块 helper 单独测对了，但调用点若被改回直接调 SDPA，上面两条仍会全绿。
    这条从模型级入口验证接线。
    """
    from modeling.krea2 import krea2_modeling as m

    torch.manual_seed(9)
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    assert not bool(mask.all()), "本用例需要含 False 的 mask"

    hits = {"n": 0}
    orig = m._chunked_masked_attention

    def spy(*a, **kw):
        hits["n"] += 1
        return orig(*a, **kw)

    m._chunked_masked_attention = spy
    try:
        with torch.no_grad():
            model(x, timesteps, context, mask)
    finally:
        m._chunked_masked_attention = orig

    assert hits["n"] > 0, "有 padding 的前向没有走分块路径 —— 调用点可能被改回裸 SDPA"


def test_all_true_mask_is_numerically_identical_to_none() -> None:
    """全 True 的 padding mask 与 ``mask=None`` 必须逐值相同。

    这是 forward 里「全 True 就置 None」那条捷径依赖的不变式。捷径的动机是显存：
    带 attn_mask 时 SDPA 的 flash 后端不接（不支持任意 mask），只能退到 math 后端
    显式 materialize 完整的 [B, H, S, S] 分数矩阵 —— 真机上（BW1000 64GB /
    Krea2 / bs=1）单次 attention 就试图分配 42.99 GiB 直接 OOM。

    而 bs=1 时 mask 恒为全 True：`pad_text_conditions` 按 batch 内最长 caption
    右填充，只有一条 caption 时 max_length 就是它自己的长度。所以这条捷径在
    最常见的配置上必然命中，它的正确性必须被钉住。

    用 rtol=0/atol=0 的严格相等：两条路径走的是**同一个** kernel（都是无 mask 的
    SDPA），不存在浮点重排，任何差异都意味着捷径的语义判断错了。
    """
    torch.manual_seed(11)
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, _ = _tiny_inputs()
    all_true = torch.ones(context.shape[0], context.shape[1], dtype=torch.bool)
    with torch.no_grad():
        with_mask = model(x, timesteps, context, all_true)
        without_mask = model(x, timesteps, context, None)
    torch.testing.assert_close(with_mask, without_mask, rtol=0, atol=0)


def test_partial_mask_still_differs_from_none() -> None:
    """含 False 的 mask **不能**被当成 None —— 否则 padding 会污染输出。

    与上一条互为对照：捷径只在全 True 时生效。这条防的是「为了省显存把判断放宽成
    无条件置 None」那种改法（它会让 test_padding_mask_prevents_padded_text... 失败，
    但那条测试的失败信息指向的是模型行为，不容易联想到是捷径写错了）。
    """
    torch.manual_seed(11)
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    assert not bool(mask.all()), "本用例需要含 False 的 mask"
    with torch.no_grad():
        masked = model(x, timesteps, context, mask)
        unmasked = model(x, timesteps, context, None)
    assert not torch.allclose(masked, unmasked, rtol=1e-4, atol=1e-4), (
        "含 padding 的 mask 被忽略了 —— padded 位置正在参与注意力"
    )


def test_gradient_checkpointing_path_backpropagates() -> None:
    model = SingleStreamDiT(_tiny_config()).train()
    model.enable_gradient_checkpointing()
    x, timesteps, context, mask = _tiny_inputs()
    x.requires_grad_(True)
    output = model(x, timesteps, context, mask)
    output.square().mean().backward()
    assert x.grad is not None
    assert model.first.weight.grad is not None
    assert model.last.linear.weight.grad is not None
    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing is False


def test_4d_latent_path_is_supported() -> None:
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    with torch.no_grad():
        output = model(x.squeeze(2), timesteps, context, mask)
    assert output.shape == (2, 4, 5, 7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"features": 63, "heads": 4},
        {"heads": 6, "kvheads": 4},
        {"txtdim": 31, "txtheads": 4},
        {"txtheads": 6, "txtkvheads": 4},
        {"tdim": 15},
        {"theta": 0},
    ],
)
def test_config_rejects_invalid_divisibility(kwargs) -> None:
    with pytest.raises(ValueError):
        Krea2Config(**kwargs)


def test_forward_rejects_invalid_temporal_and_conditioning_shapes() -> None:
    model = SingleStreamDiT(_tiny_config())
    _, timesteps, context, mask = _tiny_inputs()
    with pytest.raises(ValueError, match="T==1"):
        model(torch.randn(2, 4, 2, 4, 4), timesteps, context, mask)
    with pytest.raises(ValueError, match="context"):
        model(torch.randn(2, 4, 4, 4), timesteps, torch.randn(2, 5, 95), mask)
    with pytest.raises(ValueError, match="attention_mask"):
        model(
            torch.randn(2, 4, 4, 4),
            timesteps,
            context,
            torch.ones(2, 4, dtype=torch.bool),
        )


def test_attention_gqa_preserves_shape() -> None:
    attention = Attention(dim=64, heads=4, kvheads=2)
    x = torch.randn(2, 7, 64)
    mask = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    assert attention(x, mask=mask).shape == x.shape


def test_modeling_layer_only_imports_torch_einops_and_stdlib() -> None:
    tree = ast.parse(inspect.getsource(krea2_modeling))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert roots <= {"__future__", "dataclasses", "einops", "math", "torch"}


def test_chunked_attention_nests_checkpoint_per_chunk_when_grad_enabled() -> None:
    """带 grad 时每块必须各自套一层 checkpoint，否则分块在**反向**里等于没做。

    这条守的是一个真机上炸过、而且靠"看代码觉得对"完全看不出来的东西。

    外层 ``checkpoint(use_reentrant=False)`` 的首次前向在 ``no_grad`` 下跑 —— 每块的
    分数矩阵是临时量，峰值 = 一块，分块有效。但反向里 ``recompute_fn`` 会**带 grad**
    重跑同一段代码，此时每次 SDPA 调用都要为自己的反向保留 attention weights，
    ``S_q/chunk`` 份同时活着，加起来正好等于不分块的整块：

        真机 chunk=256、B=2、H=48、S_k=15600：
          单块 1.43 GiB × 61 块 = 87.1 GiB
          不分块 B*H*S*S*4      = 87.0 GiB   ← 一样

    也就是说没有这层嵌套，"按 query 分块"只压得住前向。真机反向重算到第 19 块
    (~26.9 GiB 激活) 就 OOM 了。

    测法：数 SDPA 的调用次数。带 grad 时每块会被 checkpoint 重算一次（前向一次 +
    该块反向时一次），所以反向跑完后的总次数应当是块数的两倍；不套嵌套则只有一倍。
    这比测显存稳（小张量上显存差异被 allocator 的块粒度吃掉，测不出来）。
    """
    from modeling.krea2 import krea2_modeling as m

    torch.manual_seed(31)
    b, h, s_q, s_k, d = 1, 2, 12, 8, 16
    chunk = 4
    n_chunks = -(-s_q // chunk)          # 3

    q = torch.randn(b, h, s_q, d, requires_grad=True)
    k = torch.randn(b, h, s_k, d, requires_grad=True)
    v = torch.randn(b, h, s_k, d, requires_grad=True)
    mask = torch.ones(b, 1, 1, s_k, dtype=torch.bool)
    mask[..., -2:] = False

    calls = {"n": 0}
    real = torch.nn.functional.scaled_dot_product_attention

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    orig = m.F.scaled_dot_product_attention
    m.F.scaled_dot_product_attention = counting
    try:
        out = m._chunked_masked_attention(q, k, v, mask, chunk=chunk)
        after_forward = calls["n"]
        out.sum().backward()
        after_backward = calls["n"]
    finally:
        m.F.scaled_dot_product_attention = orig

    assert after_forward == n_chunks, (
        f"前向该调 {n_chunks} 次 SDPA，实际 {after_forward} 次"
    )
    assert after_backward == 2 * n_chunks, (
        f"反向后总调用次数该是 {2 * n_chunks}（每块重算一次），实际 {after_backward}。"
        f"等于 {n_chunks} 说明每块没套 checkpoint —— 分块只压得住前向，反向峰值仍是"
        f"整块，真机上会 OOM。"
    )
    assert q.grad is not None and torch.isfinite(q.grad).all(), "梯度没算出来或含非有限值"


def test_chunked_attention_gradients_match_unchunked() -> None:
    """嵌套 checkpoint 不能改变梯度。

    上面那条只数了调用次数 —— 次数对但梯度错（比如重算时 mask 没跟上、或块边界的
    切片在反向里对不上）仍会绿。这条直接比梯度。

    容差取 fp32 量级而非 0：重算走的是另一次 kernel 调用，累加顺序可能不同，而浮点
    加法不满足结合律（同一原因见 test_chunked_masked_attention_matches_unchunked）。
    """
    from modeling.krea2 import krea2_modeling as m

    def grads(chunk: int | None) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(37)
        q = torch.randn(1, 2, 12, 16, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 2, 8, 16, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 2, 8, 16, dtype=torch.float64, requires_grad=True)
        mask = torch.ones(1, 1, 1, 8, dtype=torch.bool)
        mask[..., -2:] = False
        if chunk is None:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
            )
        else:
            out = m._chunked_masked_attention(q, k, v, mask, chunk=chunk)
        (out * torch.arange(1.0, out.numel() + 1, dtype=torch.float64).view(out.shape)).sum().backward()
        return q.grad.clone(), k.grad.clone(), v.grad.clone()

    reference = grads(None)
    for chunk in (4, 5, 8):
        got = grads(chunk)
        for name, a, b in zip(("q", "k", "v"), got, reference):
            torch.testing.assert_close(
                a, b, rtol=1e-9, atol=1e-9,
                msg=lambda s, n=name, c=chunk: f"chunk={c} 的 {n}.grad 与不分块不一致\n{s}",
            )
