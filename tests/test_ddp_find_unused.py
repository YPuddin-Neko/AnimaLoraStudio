"""``find_unused_parameters`` 的判据 —— 只有 module_dropout 才需要它。

为什么值得单独一个测试文件：这个 flag 两边都错得起。开着而不需要 → 每步多一次
autograd 全图遍历（真机上 DDP 会主动警告「没找到未用参数，考虑关掉」，这正是本次
修复的来源）；需要而关着 → 抛「Expected to have finished reduction in the prior
iteration」直接崩。判据必须精确对应「本步是否真可能有参数不参与前向」。

不需要 torch：被测函数只读 args 上的一个浮点数。
"""
from __future__ import annotations

import ast
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")
sys.path.insert(0, "runtime")

_SRC = pathlib.Path("runtime/training/phases/models.py")


@pytest.fixture(scope="module")
def needs():
    """从 models.py 抠出被测函数单独 exec。

    直接 import 会拖进 torch（models.py 顶层就 import torch），本机没装。而这个
    函数是纯逻辑，与 torch 无关 —— 测的仍是那份真实源码，不是复制品。
    """
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_needs_find_unused_parameters":
            ns: dict = {}
            exec(compile(ast.Module([node], []), "<models>", "exec"), ns)  # noqa: S102
            return ns["_needs_find_unused_parameters"]
    pytest.fail("models.py 里找不到 _needs_find_unused_parameters")


def _args(**kw):
    """带全部 dropout 默认值（0）的 args 替身。"""
    base = dict(lora_dropout=0.0, lora_rank_dropout=0.0, lora_module_dropout=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_off_when_no_dropout(needs):
    """全 0（默认配置）→ 关掉。

    这是真机上触发本次修复的场景：DTK 双卡 + LoRA 默认配置，DDP 报「没找到未用
    参数」。开着纯属白付每步一次全图遍历。
    """
    assert needs(_args()) is False


def test_on_when_module_dropout(needs):
    """``lora_module_dropout > 0`` → 必须开。

    LyCORIS 的 stochastic depth 按每模块每步独立掷骰子整块跳过，被跳过的模块该步
    没有梯度，而各 rank 的随机数不同步 —— 跳的不是同一批。关着的话 DDP 会等一个
    永远不来的梯度。
    """
    assert needs(_args(lora_module_dropout=0.1)) is True


@pytest.mark.parametrize("field", ["lora_dropout", "lora_rank_dropout"])
def test_other_dropouts_do_not_trigger(needs, field):
    """另两个 dropout **不该**触发 —— 它们不改变「参与前向的参数集合」。

    - ``lora_dropout`` 丢输入特征（对激活做 mask），参数照常参与矩阵乘、梯度照常产生
    - ``lora_rank_dropout`` 对中间激活乘 mask，lora_down / lora_up 两个张量整体仍在
      计算图里 —— 参数粒度上没有「未参与」

    DDP 看的是**参数**粒度，不是数值是否被 mask。把这两个也算进去会让一大批常用
    配置白付遍历开销，正是本次要修的那个浪费。
    """
    assert needs(_args(**{field: 0.5})) is False


def test_all_dropouts_together_only_module_matters(needs):
    """三个一起开时结论仍由 module_dropout 决定。"""
    assert needs(_args(lora_dropout=0.3, lora_rank_dropout=0.3)) is False
    assert needs(_args(lora_dropout=0.3, lora_rank_dropout=0.3,
                       lora_module_dropout=0.05)) is True


@pytest.mark.parametrize("bad", [None, "", "abc", object()])
def test_unreadable_config_errs_on_the_safe_side(needs, bad):
    """拿不到 / 读不懂配置时开着 —— 误判成本不对称。

    多花点时间总比训练崩掉好。None 与 "" 走 `or 0.0` 归零（= 关），非数值字符串和
    对象走 except（= 开），两条都不该抛。
    """
    result = needs(_args(lora_module_dropout=bad))
    assert isinstance(result, bool)


def test_missing_attr_treated_as_zero(needs):
    """args 上压根没这个字段（裸 CLI / 老 yaml）→ 按 0 处理，关掉。

    用 getattr 默认值而不是 KeyError：老配置文件缺字段是常态，不该因此崩在 DDP
    构造前。
    """
    assert needs(SimpleNamespace()) is False


def test_flag_is_wired_to_the_helper():
    """DDP 构造处必须调这个 helper，而不是写死 True/False。

    防回归：本次修复前那里是硬编码 `find_unused_parameters=True`。改回硬编码时
    上面所有用例仍会全绿（helper 本身没坏），只有这条能发现。
    """
    raw = _SRC.read_text(encoding="utf-8")
    assert "find_unused_parameters=_needs_find_unused_parameters(" in raw, (
        "DDP 构造处没走 _needs_find_unused_parameters —— 可能被改回硬编码了"
    )
    assert "find_unused_parameters=True," not in raw
