"""Krea2 forward 里「全 True mask 置 None」捷径的**结构性**检查（不需要 torch）。

为什么要一个不依赖 torch 的版本：数值等价性由
`test_krea2_modeling.py::test_all_true_mask_is_numerically_identical_to_none` 保证，
但那条要真 torch 才能跑。而这条捷径的动机是**显存**（真机上差着 42.99 GiB 的
OOM），一旦被后来的重构悄悄改掉，在有 torch 的 CI 上数值测试仍会全绿 —— 因为两条
路径本来就数值等价，测试测不出「走的是慢路径」。

所以这里用 AST 钉住结构：捷径存在、且判定条件是「mask 全 True」。这是在没有显存
探针的情况下，能对「性能特性没被改掉」做的最直接断言。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = pathlib.Path("modeling/krea2/krea2_modeling.py")


@pytest.fixture(scope="module")
def forward_src() -> str:
    """取 SingleStreamDiT.forward 的源码文本。"""
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SingleStreamDiT":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return ast.unparse(item)
    pytest.fail("找不到 SingleStreamDiT.forward")


def test_shortcut_checks_mask_all_true(forward_src: str) -> None:
    """判定必须是 `attention_mask.all()` —— 不是别的启发式。

    曾经想过的错误写法：判 `batch == 1`（bs>1 且各 caption 等长时同样全 True，
    会漏掉）、或判 `attention_mask.any()`（语义反了，任何一个 True 就置 None，
    直接让 padding 参与注意力）。
    """
    assert ".all()" in forward_src, (
        "forward 里没有 .all() 判定 —— 全 True mask 的短路捷径可能被移除了。"
        "移除会让 SDPA 退到 math 后端 materialize [B,H,S,S] 分数矩阵，"
        "真机上（Krea2 / bs=1 / 64GB 卡）直接 OOM"
    )


def test_shortcut_leaves_masks_as_none(forward_src: str) -> None:
    """两个 mask 变量都要有 None 初值 —— 短路分支靠它们保持 None 生效。"""
    assert "text_mask = None" in forward_src
    assert "combined_mask = None" in forward_src


def test_image_mask_not_built_on_shortcut_path(forward_src: str) -> None:
    """全 True 路径上不该分配 image_mask、也不该 torch.cat。

    这两步在最常见的路径（bs=1）上纯浪费：image_mask 是个 [B, image_len] 的 bool
    张量（图像 token 数可达上万），cat 之后还要再切片，而结果紧接着被丢弃。
    结构上体现为 `torch.ones(` 与 `torch.cat(` 必须在 else 分支里，不在 if 之前。
    """
    tree = ast.parse(forward_src)
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)

    # 找到那个判 .all() 的 if
    target: ast.If | None = None
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and ".all()" in ast.unparse(node.test):
            target = node
            break
    assert target is not None, "找不到判 .all() 的 if 分支"

    body_src = "".join(ast.unparse(s) for s in target.body)
    else_src = "".join(ast.unparse(s) for s in target.orelse)

    assert "torch.ones(" not in body_src, "全 True 分支里不该分配 image_mask"
    assert "torch.cat(" not in body_src, "全 True 分支里不该做 torch.cat"
    assert "torch.ones(" in else_src, "有 padding 的分支必须分配 image_mask"
    assert "torch.cat(" in else_src, "有 padding 的分支必须 cat 出 combined_mask"


def test_shortcut_reasoning_is_documented(forward_src: str) -> None:
    """捷径旁必须留下为什么 —— 它是个反直觉的性能特化（看着像多余的判断）。

    注释在 ast.unparse 后会丢失，所以直接读源文件。这条测试的意义不是形式主义：
    没有注释的话，下一个读到这段的人很可能把它当冗余判断删掉，而删掉的后果
    （OOM）要在多卡真机上才复现得出来。
    """
    raw = _SRC.read_text(encoding="utf-8")
    idx = raw.find(".all()")
    assert idx > 0
    window = raw[max(0, idx - 2000):idx]
    assert "flash" in window, "捷径旁的注释没说明 flash 后端不接 attn_mask 这个根因"
    assert "OOM" in window or "GiB" in window, "注释没记录真机 OOM 的实测事实"
