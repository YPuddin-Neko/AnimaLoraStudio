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

    分块之所以**精确**而非近似：softmax 沿 key 维归一化，每个 query 行的输出只依赖
    该行自己的分数向量，query 之间无耦合。所以这里用 rtol=0/atol=0 —— 有任何差异
    就说明切法错了（切错维度、块边界漏算、mask 对不上），不是浮点误差。
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
        torch.testing.assert_close(got, reference, rtol=0, atol=0)


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
    torch.testing.assert_close(out, real(q, k, v, attn_mask=mask), rtol=0, atol=0)


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
