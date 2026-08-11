"""多卡下 SIGINT 必须**异步**（只置标志），不能像单卡那样立即退出。

真机故障（epoch 9 暂停）
----------------------
暂停信号在采样期间到达：rank 0 正在 ``is_main()`` 里出图，``handle_interrupt``
跑完就 ``sys.exit(0)``；rank 1 早已越过采样前的 barrier，此刻阻塞在采样后那个
barrier 上等永不到来的 rank 0。NCCL 的 barrier 阻塞在 C++ 里，Python 信号处理器
要等当前 C 调用返回才有机会跑，所以 rank 1 连自己的 SIGINT 都处理不了 ——
30 秒宽限期后被 torchrun SIGKILL：

    22:13:28  Received 2 death signal, shutting down workers
    22:13:58  Unable to shutdown process 942213 via 2, forcefully exiting via 9

被 SIGKILL 打断的 rank 死在集合操作中间，RCCL 通信器没有正常销毁。

supervisor 的文件标记通道本来就是为多卡设计的，但它的注释假设「worker 每步轮询」
—— 采样一次 3-7 分钟，那不是一个训练步，期间没有轮询点。所以光有文件通道不够，
信号这条路也必须改成异步。

本文件锁住的不变量
----------------
1. world_size > 1 → SIGINT 绑到 request_pause_from_signal（只置标志）
2. world_size == 1 → SIGINT 绑到 handle_interrupt（立即退出，CLI Ctrl+C 依赖即时性）
3. request_pause_from_signal 第一次只置标志、不退出；第二次强退
4. 训练循环的两处轮询点都必须同时看文件标记**和**信号标志

绕开 torch：用 AST 读真实源码，不 import runtime.training（那会拉起 torch）。
"""
from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, ".")

_REPO = Path(__file__).resolve().parent.parent
_RESUME = _REPO / "runtime" / "training" / "phases" / "resume.py"
_LOOP = _REPO / "runtime" / "training" / "loop.py"
_CONTEXT = _REPO / "runtime" / "training" / "context.py"


def _fn_source(path: Path, name: str) -> str:
    """抠出某个函数/方法的源码（第一个同名者）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            lines = path.read_text(encoding="utf-8").splitlines()
            return textwrap.dedent("\n".join(lines[node.lineno - 1: node.end_lineno]))
    raise AssertionError(f"{path.name} 里找不到 {name}()")


def _signal_calls(src: str) -> list[tuple[str, str]]:
    """找出 signal.signal(sig, handler) 调用，返回 [(信号名, handler 源码)]。"""
    out: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "signal"):
            continue
        if not (isinstance(f.value, ast.Name) and f.value.id == "signal"):
            continue
        if len(node.args) != 2:
            continue
        out.append((ast.unparse(node.args[0]), ast.unparse(node.args[1])))
    return out


def _enclosing_tests(src: str, needle_attr: str) -> list[str]:
    """找出包裹「调用了 signal.signal 且 handler 名含 needle_attr」的 if 条件。"""
    tree = ast.parse(src)
    hits: list[str] = []

    def walk(node: ast.AST, conds: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                t = ast.unparse(child.test)
                for s in child.body:
                    walk_stmt(s, [*conds, t])
                for s in child.orelse:
                    # else / elif 分支：不带 body 的 test
                    walk_stmt(s, [*conds, f"NOT({t})"])
            else:
                walk(child, conds)

    def walk_stmt(stmt: ast.AST, conds: list[str]) -> None:
        for (_sig, handler) in _signal_calls(ast.unparse(stmt)):
            if needle_attr in handler:
                hits.append(" and ".join(conds))
                return
        if isinstance(stmt, ast.If):
            walk(ast.Module(body=[stmt], type_ignores=[]), conds)
            return
        walk(stmt, conds)

    walk(tree, [])
    return hits


# ── 1 / 2：信号绑定按 world_size 分流 ────────────────────────────────────────


def _bindings_by_branch(src: str) -> list[tuple[str, str, str]]:
    """[(包裹条件, 信号名, handler 源码)]，条件取自真实 if/elif 链。"""
    out: list[tuple[str, str, str]] = []

    def walk(node: ast.AST, conds: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                t = ast.unparse(child.test)
                for s in child.body:
                    walk_stmt(s, [*conds, t])
                for s in child.orelse:
                    walk_stmt(s, [*conds, f"NOT({t})"])
            else:
                walk(child, conds)

    def walk_stmt(stmt: ast.AST, conds: list[str]) -> None:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Attribute) and f.attr == "signal"):
                continue
            if not (isinstance(f.value, ast.Name) and f.value.id == "signal"):
                continue
            if len(node.args) != 2:
                continue
            out.append((
                " and ".join(conds),
                ast.unparse(node.args[0]),
                ast.unparse(node.args[1]),
            ))
        if isinstance(stmt, ast.If):
            walk(ast.Module(body=[stmt], type_ignores=[]), conds)
            return
        walk(stmt, conds)

    walk(ast.parse(src), [])
    return out


def test_multigpu_branch_binds_only_async_handler() -> None:
    """world_size > 1 的分支里，**每一处**信号绑定都必须是 request_pause_from_signal。

    逐处检查而不是「存在任意一处」：只改 POSIX 那行、漏掉 Windows 的 SIGBREAK
    （或反过来）是最自然的回归方式，「有一处对」的断言抓不住它。
    """
    src = _fn_source(_RESUME, "run")
    bindings = _bindings_by_branch(src)
    assert bindings, "resume.run() 里找不到任何 signal.signal 绑定"

    multi = [
        (c, s, h) for c, s, h in bindings
        if "world_size" in c and not c.startswith("NOT(") and "NOT(dist_env.world_size" not in c
    ]
    assert multi, (
        f"找不到由 world_size 把门的多卡分支绑定。实际绑定：{bindings}"
    )
    for cond, sig, handler in multi:
        assert "request_pause_from_signal" in handler, (
            f"多卡分支（条件 {cond}）把 {sig} 绑到了 {handler} —— "
            f"多卡下立即退出会让其余 rank 挂在集合操作上被 SIGKILL。"
        )
        assert "handle_interrupt" not in handler, (
            f"多卡分支（条件 {cond}）仍在用 handle_interrupt：{handler}"
        )


def test_async_handler_is_gated_on_world_size() -> None:
    """异步处理器只能在 world_size > 1 时装 —— 单卡要保持即时退出。"""
    src = _fn_source(_RESUME, "run")
    conds = _enclosing_tests(src, "request_pause_from_signal")
    assert conds, "找不到包裹 request_pause_from_signal 绑定的 if 条件"
    for c in conds:
        assert "world_size" in c, (
            f"request_pause_from_signal 的绑定没有由 world_size 把门，"
            f"单卡也会走异步路径、Ctrl+C 不再即时生效。\n实际条件：{c}"
        )


def test_singlecard_still_binds_immediate_handler() -> None:
    """单卡分支必须仍然绑 handle_interrupt（立即退出）。"""
    src = _fn_source(_RESUME, "run")
    conds = _enclosing_tests(src, "ctx.handle_interrupt")
    assert conds, (
        "resume.run() 里找不到绑 handle_interrupt 的分支 —— "
        "单卡 CLI Ctrl+C 依赖它的即时性。"
    )
    assert any("NOT(" in c or "world_size" in c for c in conds), (
        f"handle_interrupt 的绑定不在 world_size 分流的任一支上：{conds}"
    )


# ── 3：request_pause_from_signal 的行为 ─────────────────────────────────────


def test_async_handler_sets_flag_and_does_not_exit_first_time() -> None:
    """第一次进来只置标志 + emit，不能有无条件的 sys.exit。"""
    src = _fn_source(_CONTEXT, "request_pause_from_signal")
    tree = ast.parse(src)
    assert "pause_signal_seen = True" in src.replace("self.", ""), (
        "request_pause_from_signal 没有置 pause_signal_seen 标志"
    )
    # 所有 sys.exit 都必须在 if 里（即重复触发那条路），不能在函数顶层语句序列上
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    for stmt in fn.body:
        for sub in ast.walk(stmt):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "exit"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "sys"
            ):
                assert isinstance(stmt, ast.If), (
                    "request_pause_from_signal 在非条件位置调了 sys.exit —— "
                    "第一次收到信号就退出，等于没改；其余 rank 仍会被 SIGKILL。"
                )


def test_async_handler_force_exits_on_second_signal() -> None:
    """重复按 → 立即强退（用户等不下去时的逃生口）。"""
    src = _fn_source(_CONTEXT, "request_pause_from_signal")
    assert "if self.pause_signal_seen" in src, (
        "没有「标志已置 → 强退」的分支，用户连按两次也只能干等"
    )


# ── 4：两处轮询点都要看信号标志 ─────────────────────────────────────────────


@pytest.mark.parametrize("channel", ["pause_requested", "pause_signal_seen"])
def test_loop_polls_both_pause_channels(channel: str) -> None:
    """训练循环里每处 handle_interrupt 调用点都必须同时看两条通道。

    只看文件标记 → CLI 里 Ctrl+C 起的多卡训练永远等不到标记文件。
    只看信号标志 → supervisor 的文件通道失效。
    """
    src = _LOOP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    sites: list[tuple[int, str]] = []

    def walk(node: ast.AST, conds: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                t = ast.unparse(child.test)
                for s in child.body:
                    walk_stmt(s, [*conds, t])
                for s in child.orelse:
                    walk_stmt(s, conds)
            else:
                walk(child, conds)

    def walk_stmt(stmt: ast.AST, conds: list[str]) -> None:
        for sub in ast.walk(stmt):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "handle_interrupt"
            ):
                sites.append((getattr(stmt, "lineno", -1), " and ".join(conds)))
                return
        if isinstance(stmt, ast.If):
            walk(ast.Module(body=[stmt], type_ignores=[]), conds)
            return
        walk(stmt, conds)

    walk(tree, [])
    assert sites, "loop.py 里找不到 handle_interrupt 调用点"
    for lineno, cond in sites:
        assert channel in cond, (
            f"loop.py:{lineno} 的 handle_interrupt 调用条件里没有 {channel}，"
            f"这条暂停通道在该位置失效。\n实际条件：{cond}"
        )


def test_epoch_end_pause_check_precedes_sampling() -> None:
    """epoch 末的暂停检查必须存在，且在采样之前。

    真机故障正是「rank 0 在采样中被打断、rank 1 挂在采样后的 barrier」，
    所以闸必须在采样门控之前。

    位置全部取自 AST 节点行号，不做文本搜索 —— 注释里也会出现
    handle_interrupt / sample_every_all_ranks 这些词，文本匹配会把注释
    当成代码，删掉真实调用后测试照样通过（本测试第一版就栽在这里）。
    """
    src = _LOOP.read_text(encoding="utf-8")
    tree = ast.parse(src)

    epoch_end_lines: list[int] = []
    interrupt_lines: list[int] = []
    sampling_lines: list[int] = []

    for node in ast.walk(tree):
        # epoch 末标志：ctx.current_epoch = epoch + 1
        if isinstance(node, ast.Assign):
            tgt = node.targets[0] if node.targets else None
            if (
                isinstance(tgt, ast.Attribute)
                and tgt.attr == "current_epoch"
                and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.Add)
            ):
                epoch_end_lines.append(node.lineno)
        # 真实的 handle_interrupt 调用
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "handle_interrupt"
        ):
            interrupt_lines.append(node.lineno)
        # 真实读取 ctx.sample_every_all_ranks
        if isinstance(node, ast.Attribute) and node.attr == "sample_every_all_ranks":
            sampling_lines.append(node.lineno)

    assert epoch_end_lines, "找不到 ctx.current_epoch = epoch + 1"
    epoch_end = min(epoch_end_lines)
    assert sampling_lines, "找不到 ctx.sample_every_all_ranks 的读取"
    sampling = min(ln for ln in sampling_lines if ln > epoch_end)

    after_epoch_end = [ln for ln in interrupt_lines if ln > epoch_end]
    assert after_epoch_end, (
        f"epoch 末（第 {epoch_end} 行）之后没有 handle_interrupt 调用 —— "
        "epoch 末的暂停闸被删了。标记在采样前出现时，rank 1 会撞进采样后"
        "那个没人应答的 barrier，30 秒后被 SIGKILL。"
    )
    pause_check = min(after_epoch_end)
    assert pause_check < sampling, (
        f"epoch 末的暂停检查（第 {pause_check} 行）在采样门控（第 {sampling} 行）"
        "之后 —— 起不到防挂的作用。"
    )
