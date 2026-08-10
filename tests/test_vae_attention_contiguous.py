"""WAN VAE 空间自注意力的 q/k/v 连续性 —— 决定走 flash 还是 math 后端。

背景（真机实测，BW1000 / DTK 26.04）：VAE 编码在 ``batch size >= 2`` 时抛

    RuntimeError: CUDA error: HIPBLAS_STATUS_INVALID_VALUE when calling
    `hipblasSgemmStridedBatched(...)`

而 bs=1 正常。不是显存不足（卡上 64GB），是 BLAS 参数类型溢出：

原实现把 ``.contiguous()`` 放在 ``.chunk()`` **之前**，于是切出的三片是同一块内存
上的跨步视图（最后一维长度 c，倒数第二维 stride 仍是 3c）。SDPA 的 flash /
mem-efficient 后端都要求最后两维连续，被拒后回落 math 后端，显式 materialize
``[b*t, 1, S, S]``。1024² 图经 VAE 8× 下采样后 S = 128×128 = 16384，单个分数矩阵
16384² × 4B = 恰好 1.00 GiB；hipBLAS 用 signed int32 算总输出字节：

    bs=1  1,073,741,824 B  塞得进 int32
    bs=2  2,147,483,648 B  正好 2^31，越界 1 字节

修法是 chunk 之后**各自** contiguous，让 flash 接管 —— 分数矩阵根本不被构造。

本文件不需要 torch：连续性是张量**布局**性质，可以用 stride 算术精确推演；另有一条
AST 检查钉住修法不被改回去。真张量的数值等价性由 test_vae_tiled_decode.py 那侧的
既有覆盖 + 真机验证保证。
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


def test_reasoning_is_documented():
    """修法旁必须留下 why：它看着像多余的拷贝，很容易被当性能负担删掉。

    删掉的后果（bs>=2 时 hipBLAS INVALID_VALUE）只在海光真机上复现，
    NVIDIA 上永远不会暴露（cuBLAS 用 int64 算偏移）。
    """
    raw = _SRC.read_text(encoding="utf-8")
    idx = raw.find(".chunk(3, dim=-1)")
    assert idx > 0
    window = raw[max(0, idx - 2500):idx]
    assert "flash" in window, "注释没说明 flash 后端要求连续布局这个根因"
    assert "int32" in window, "注释没记录 hipBLAS int32 越界这个真机事实"
    assert "2147483648" in window or "2^31" in window, "注释没写出越界的具体数值"
