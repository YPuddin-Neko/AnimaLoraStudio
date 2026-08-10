"""WAN VAE 空间自注意力的显存兜底 —— 分块 + 连续布局。

背景（真机实测，BW1000 / DTK 26.04）：VAE 编码在 ``batch size >= 2`` 时抛

    RuntimeError: CUDA error: HIPBLAS_STATUS_INVALID_VALUE when calling
    `hipblasSgemmStridedBatched(...)`

而 bs=1 正常。不是显存不足（卡上 64GB），是 BLAS 参数类型溢出：

根因有两层，第一次只修了外层所以没生效：

**外层（必要但不充分）**：原实现把 ``.contiguous()`` 放在 ``.chunk()`` 之前，切出的
三片是同一块内存上的跨步视图（最后一维长度 c，倒数第二维 stride 仍是 3c），SDPA 的
快后端都要求最后两维连续，跨步视图会被拒。

**内层（真正的拦路者）**：这处 attention 的 head 数固定为 1，于是 head_dim = 通道数，
最深层 c = 512 —— 远超 flash 的 head_dim 上限（FA2 最多 256，ROCm/AOTriton 多为 128）。
所以 flash **无论布局如何都不可能接管**。只改 contiguous 之后仍然走 math，真机上同一
处继续报错（这是第一版修复失效的原因）。

math 后端显式 materialize ``[b*t, 1, S_q, S_k]``。1024² 图经 VAE 8× 下采样后
S = 128×128 = 16384，单个分数矩阵 16384² × 4B = 恰好 1.00 GiB；hipBLAS 用 signed
int32 算总输出字节：

    bs=1  1,073,741,824 B  塞得进 int32
    bs=2  2,147,483,648 B  正好 2^31，越界 1 字节

最终修法是**按 query 分块**（与 Krea2 那处同源，但那边的触发条件是 attn_mask 挡住
flash，这边是 head_dim 超限）。contiguous 仍然保留 —— 它让 mem-efficient 后端在支持
它的平台上可用（那个后端没有 head_dim 上限），DTK 上没编译 mem-efficient 才必须靠
分块兜底。

本文件不需要 torch：连续性与分块都是可以用算术精确推演的性质；另有 AST 检查钉住两处
修法不被改回去。真张量的数值等价性由 test_vae_tiled_decode.py 的既有覆盖 + 真机验证
保证。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = pathlib.Path("modeling/wan/vae2_1.py")


def _strides_after_permute_then_chunk(b_t: int, c: int, s: int):
    """复现 `reshape(b*t,1,3c,-1).permute(0,1,3,2)` 后再 chunk 的 stride。

    返回 (每片最后一维长度, 每片倒数第二维 stride)。两者相等 = 最后两维连续。
    """
    # reshape 后 [b_t, 1, 3c, S]，C-contiguous strides = (3c*S, 3c*S, S, 1)
    # permute(0,1,3,2) → [b_t, 1, S, 3c]，strides = (3c*S, 3c*S, 1, S)
    # 该视图**非**连续（最后一维 stride=S 而不是 1）
    # .contiguous() → strides = (S*3c, S*3c, 3c, 1)
    # .chunk(3, dim=-1) → 每片 [b_t, 1, S, c]，strides 不变 = (..., 3c, 1)
    return c, 3 * c


def test_stride_math_shows_chunk_before_contiguous_is_noncontiguous():
    """先 contiguous 再 chunk → 每片最后两维**不**连续。

    这条把「为什么会掉到 math 后端」变成可验证的算术，而不是靠记忆。
    """
    last_len, second_last_stride = _strides_after_permute_then_chunk(2, 384, 128 * 128)
    assert last_len == 384
    assert second_last_stride == 1152
    assert second_last_stride != last_len, (
        "若两者相等则布局连续、flash 可用 —— 那本 bug 就不存在了"
    )


@pytest.mark.parametrize("bs,overflows", [(1, False), (2, True), (3, True)])
def test_int32_boundary_matches_real_machine_behaviour(bs: int, overflows: bool):
    """int32 越界点必须落在 bs=2 —— 与真机「bs=1 好、bs=2 崩」精确对应。

    这条是本次诊断的核心证据。写成测试是为了让「为什么恰好是 2」这个反直觉的
    分界点留在代码库里：不是显存不够，是 1 GiB × bs 撞 2^31 字节。
    """
    s = 128 * 128
    total_bytes = bs * s * s * 4  # fp32 分数矩阵
    assert (total_bytes >= 2**31) is overflows
    if bs == 1:
        assert total_bytes == 1_073_741_824
    if bs == 2:
        assert total_bytes == 2_147_483_648  # 正好 2^31


def _attention_forward_src() -> str:
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "Attention" in node.name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return ast.unparse(item)
    pytest.fail("找不到 VAE Attention 类的 forward")


def test_contiguous_applied_per_chunk_not_before():
    """``.contiguous()`` 必须作用在**每一片**上，而不是 chunk 之前的整块。

    按 AST 结构判断而非文本位置：正确写法是 generator / 推导式
    ``(t.contiguous() for t in ....chunk(3, dim=-1))`` —— 其中 ``contiguous`` 在
    **文本上**先于 ``chunk``（它在 element 表达式里），但**执行上**在其后（每片被
    yield 时才调）。用 `src.find()` 比位置会得出相反结论，这也是这条测试最初写错的
    地方。

    防回归：顺序颠倒回去后数值仍完全正确（math 与 flash 数学等价），只有显存与
    hipBLAS 的 int32 限制会暴露 —— 那要在海光真机上跑 bs>=2 才复现。静态断言是
    CI 上唯一能抓住它的手段。
    """
    src = _attention_forward_src()
    tree = ast.parse(src)

    def _is_chunk_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "chunk"
        )

    def _calls_contiguous(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "contiguous"
            for n in ast.walk(node)
        )

    # 找 generator / listcomp，其迭代源含 .chunk()、element 含 .contiguous()
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            continue
        src_has_chunk = any(_is_chunk_call(n) for gen in node.generators
                            for n in ast.walk(gen.iter))
        if src_has_chunk and _calls_contiguous(node.elt):
            found = True
            break
    assert found, (
        "没找到「对 chunk 出的每一片各自 contiguous」的结构 —— "
        "三片可能仍共享 3c 的 stride，flash 会拒绝并回落 math 后端"
    )


def test_head_dim_exceeds_flash_limit_at_deepest_layer():
    """最深层的 head_dim 必须被认定为「flash 不可能接管」。

    这是第一版修复失效的原因，写成测试是为了让它留在代码库里：这处 attention 的
    head 数固定为 1（reshape 里那个字面量 1），所以 head_dim = 通道数。WAN VAE 的
    dim=128、最深 mult=4 → head_dim = 512，而 flash 的上限是 256（ROCm 上多为 128）。

    只修 contiguous 不够，必须有分块兜底。
    """
    dim, deepest_mult = 128, 4
    head_dim = dim * deepest_mult
    assert head_dim == 512
    assert head_dim > 256, (
        "若 head_dim 落回 flash 的上限内，分块就不再是必需的 —— 但那意味着模型结构"
        "变了，这条测试的前提需要重新核对"
    )


def test_chunked_attention_used_at_call_site():
    """attention 调用点必须走分块 helper，而不是裸 SDPA。

    防回归：裸 SDPA 在数值上完全正确（math 与分块数学等价），只有显存和 hipBLAS 的
    int32 限制会暴露 —— 那要在海光真机上跑 bs>=2 才复现。
    """
    src = _attention_forward_src()
    assert "_chunked_attention(" in src, (
        "forward 没走 _chunked_attention —— math 后端会 materialize 完整 [S, S] "
        "分数矩阵，bs>=2 时撞 hipBLAS 的 int32 上限"
    )
    assert "F.scaled_dot_product_attention(" not in src, (
        "forward 里还有裸 SDPA 调用 —— 应该全部经由 _chunked_attention"
    )


def test_chunk_size_keeps_common_batches_under_int32():
    """chunk 取值必须让常见 batch 都远离 int32 上限。

    钉住这个数值而不只是「存在分块」：chunk 调大到 8192 时 bs=2 就又是 1 GiB、
    bs=4 撞线，等于修了一半。
    """
    import re

    raw = _SRC.read_text(encoding="utf-8")
    m = re.search(r"_VAE_ATTN_QUERY_CHUNK\s*=\s*(\d+)", raw)
    assert m, "找不到 _VAE_ATTN_QUERY_CHUNK"
    chunk = int(m.group(1))
    s = 128 * 128
    for bs in (1, 2, 4, 8):
        total = bs * chunk * s * 4
        assert total < 2**31, (
            f"chunk={chunk} 时 bs={bs} 的分数矩阵 {total:,} B 撞 int32 上限"
        )


def test_reasoning_is_documented():
    """两处修法旁都必须留下 why。

    contiguous 看着像多余拷贝、分块看着像多余循环，都很容易被当性能负担删掉，而删掉
    的后果（bs>=2 时 hipBLAS INVALID_VALUE）只在海光真机上复现 —— NVIDIA 上永远不会
    暴露（cuBLAS 用 int64 算偏移）。
    """
    raw = _SRC.read_text(encoding="utf-8")

    # 分块 helper 旁：要写清 head_dim 超限这个根因 + int32 事实
    hidx = raw.find("_VAE_ATTN_QUERY_CHUNK = ")
    assert hidx > 0
    head_win = raw[max(0, hidx - 1500):hidx]
    assert "head_dim" in head_win, "分块常量旁没说明 head_dim 超限这个根因"
    assert "int32" in head_win, "分块常量旁没记录 hipBLAS int32 越界"

    # 调用点旁：要说清 contiguous 是必要不充分
    cidx = raw.find("_chunked_attention(q, k, v)")
    assert cidx > 0
    call_win = raw[max(0, cidx - 2500):cidx]
    assert "不充分" in call_win or "必要" in call_win, (
        "调用点旁没说明 contiguous 是必要但不充分条件 —— 下一个人可能又以为"
        "改 contiguous 就够了（第一版修复就是这么失效的）"
    )
    assert "2147483648" in call_win or "2^31" in call_win, "注释没写出越界的具体数值"
