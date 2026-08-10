"""``bootstrap._check_ddp_prerequisites`` 的启动期互斥拦截。

这个函数此前**一条测试都没有** —— block swap / leap / SRA 三条拦截自写下就没被
验证过。补这个文件时才发现，所以四条一起测，不只测新加的那条。

为什么这些必须 fail-fast 而不是静默钉值：每一条猜错方向都会让用户以为「设置生效
了」。block swap 是为塞进小显存、多卡是为摊算力，替用户二选一都不对。

为什么 runtime 拦截与 schema 的 ``disable_when`` 声明并存（见
``tests/test_ddp_schema.py``）：schema 规则管 Studio UI 与配置读盘，而纯 CLI 启动
（``python runtime/anima_train.py --...``）压根不经过 pydantic。两层各守一条入口。
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest

_SRC = pathlib.Path(__file__).resolve().parent.parent / "runtime/training/phases/bootstrap.py"


def _load_check(dist_env_stub):
    """从 bootstrap.py 抠出被测函数单独 exec，注入 ``dist_env`` 替身。

    直接 import 会拖进 torch（bootstrap.py 顶层就 ``import torch``），本机没装 ——
    与 ``tests/test_ddp_find_unused.py`` 同款做法。这个函数是纯逻辑（只读 args 字段
    + 问 dist_env 几个布尔），与 torch 无关，所以测的仍是那份真实源码而非复制品。

    ``logger`` 也要给：函数末尾会对 SRA 单独开的情形发 warning。
    """
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_check_ddp_prerequisites":
            ns: dict = {
                "dist_env": dist_env_stub,
                "logger": types.SimpleNamespace(
                    warning=lambda *a, **k: None,
                    info=lambda *a, **k: None,
                ),
            }
            exec(compile(ast.Module([node], []), "<bootstrap>", "exec"), ns)  # noqa: S102
            return ns["_check_ddp_prerequisites"]
    pytest.fail("bootstrap.py 里找不到 _check_ddp_prerequisites")


def _args(**overrides):
    """造一个只有被测字段的 args 替身。

    用 ``SimpleNamespace`` 而不是真 ``TrainingConfig``：被测函数全程用
    ``getattr(args, name, default)`` 读值，且这里要能构造 schema 会拒绝的组合
    （多卡 + block swap 是 pydantic 校验失败的，见 test_ddp_schema）—— 正是
    纯 CLI 路径能造出来的那些。
    """
    base = dict(
        blocks_to_swap=0,
        leap_enabled=False,
        ppsf_fused_back_pass=False,
        sra_enabled=False,
        grad_clip_max_norm=0.0,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _dist_env(*, distributed: bool, world_size: int):
    return types.SimpleNamespace(
        is_distributed=lambda: distributed,
        world_size=lambda: world_size,
        rank=lambda: 0,
        local_rank=lambda: 0,
        is_main=lambda: True,
    )


@pytest.fixture(scope="module")
def check():
    """被测函数，``dist_env`` 伪装成 2 卡。"""
    return _load_check(_dist_env(distributed=True, world_size=2))


@pytest.fixture(scope="module")
def check_single():
    """同上，但伪装成单进程 —— 整个函数应当是 no-op。"""
    return _load_check(_dist_env(distributed=False, world_size=1))


# --------------------------------------------------------------- 单进程 no-op


@pytest.mark.parametrize("overrides", [
    {"blocks_to_swap": 28},
    {"leap_enabled": True},
    {"ppsf_fused_back_pass": True},
    {"sra_enabled": True, "grad_clip_max_norm": 1.0},
])
def test_single_process_never_blocks_anything(check_single, overrides) -> None:
    """单卡下这些特性全都合法 —— 拦截只针对多卡。

    这条锁住「零影响」这个前提：整条多卡改动的安全阀是单进程行为不变。
    """
    check_single(_args(**overrides))  # 不抛即通过


# ------------------------------------------------------------ 多卡逐条拦截


def test_clean_config_passes(check) -> None:
    """多卡 + 全部默认值 → 放行。防「拦得太宽」。"""
    check(_args())


def test_block_swap_is_rejected(check) -> None:
    """block swap：每 rank 各锁一份换出层内存，且换入换出与 DDP 抢 PCIe。"""
    with pytest.raises(RuntimeError, match="blocks_to_swap"):
        check(_args(blocks_to_swap=28))


def test_leap_is_rejected(check) -> None:
    """leap：一步多次前向，撞 DDP「一次 forward 对一次 backward」的硬约束。"""
    with pytest.raises(RuntimeError, match="leap_enabled"):
        check(_args(leap_enabled=True))


def test_ppsf_fused_back_pass_is_rejected(check) -> None:
    """PPSF fused backward：反向途中就地改参数 + 释放梯度，与 reducer 抢同一块存储。

    这条之前**没有任何拦截** —— 它是 False 纯粹因为配置默认值，不是因为有代码判断。
    而它是个 advanced 开关、描述写着「显存吃紧时开」，多卡 + full matrix（803M
    可训练参数）正是最容易去点它的处境。后果是静默的：不报错，各 rank 梯度不一致。
    """
    with pytest.raises(RuntimeError, match="ppsf_fused_back_pass"):
        check(_args(ppsf_fused_back_pass=True))


def test_sra_with_grad_clip_is_rejected(check) -> None:
    """SRA + 裁剪：未同步的 SRA 梯度混进全局范数 → 各 rank 裁剪系数不同。"""
    with pytest.raises(RuntimeError, match="sra_enabled"):
        check(_args(sra_enabled=True, grad_clip_max_norm=1.0))


def test_sra_without_grad_clip_passes(check) -> None:
    """SRA 单独开是允许的（退化成 per-rank 独立探针，只发警告）。

    锁住拦截条件里的 ``and`` —— 写成 ``or`` 会把合法配置也拦掉。
    """
    check(_args(sra_enabled=True, grad_clip_max_norm=0.0))


# ----------------------------------------------------------- 报错信息质量


def test_all_problems_reported_at_once(check) -> None:
    """多条同时踩中时一次全报，而不是修一条再撞下一条。

    ``problems`` 列表最后统一 raise 就是为了这个 —— 用户改一次配置就能全部解决。
    """
    with pytest.raises(RuntimeError) as excinfo:
        check(_args(
            blocks_to_swap=28,
            leap_enabled=True,
            ppsf_fused_back_pass=True,
            sra_enabled=True,
            grad_clip_max_norm=1.0,
        ))
    msg = str(excinfo.value)
    for field in (
        "blocks_to_swap", "leap_enabled", "ppsf_fused_back_pass", "sra_enabled",
    ):
        assert field in msg, f"{field} 没出现在报错里，用户得逐条试错"
    assert "world_size=2" in msg, "报错没说清是多卡导致的"


def test_error_names_the_field_verbatim(check) -> None:
    """报错里的字段名必须与配置里的键**逐字相同**。

    用户是拿这个名字去 config.yaml / UI 里搜的；写成「块交换」之类的意译会让人
    找不到该改哪一项。
    """
    with pytest.raises(RuntimeError) as excinfo:
        check(_args(ppsf_fused_back_pass=True))
    assert "ppsf_fused_back_pass" in str(excinfo.value)
