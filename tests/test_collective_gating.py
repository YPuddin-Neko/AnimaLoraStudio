"""集合操作的门控条件必须是 rank 不变量。

这个文件守一类真机上炸过的 bug，而不是某一行代码。

背景：``bootstrap`` 会在非 rank 0 上把几个 args 字段改掉（关掉重复出图 / 进度条 /
状态文件），这个抑制本身是必要的。但 ``dist_env.barrier()`` 是**集合操作 —— 必须
所有 rank 都执行**。一旦某个 barrier 的外层 ``if`` 读了那些被 rank 改过的字段，
非 rank 0 就会连 barrier 一起跳过，后果是：

1. NCCL 按**调用顺序**配对集合操作。rank 0 多执行了 N 次 barrier，从此每个 rank 的
   集合操作永久错位 —— 真机上表现为 rank 1 的 ``_agree_on_finite_loss`` 与 rank 0
   的 barrier 配上对，读回垃圾值，误报「其他 rank 的 loss 非有限值」。
2. rank 0 采样时其余 rank 不再被挡，径直冲进下一步前向；被误报的 NaN skip 又让上一个
   micro-batch 的计算图活着不释放 → 两份 gradient-checkpoint 图叠在一起 → OOM。

真机日志（BW1000 ×2）里两条相隔 47ms 的行就是证据::

    00:36:14.222  [rank 1] step 0 ... 其他 rank 的 loss 非有限值   ← 已在训练前向
    00:36:14.269  [rank 0] [采样前] 显存 alloc=29.1GB            ← 才开始采样

为什么用静态分析而不是跑双卡：这类 bug 只在 world_size>1 且「rank 相关字段恰好把
条件翻成 False」时出现，单机 CI 复现不了；而它的形态是纯语法的 —— 门控读了哪些名字
在 AST 上一眼可见。所以直接在 AST 上立规矩。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

#: bootstrap 在非 rank 0 上改写的 ``args`` 字段。门控读到任何一个都是 bug。
#:
#: 与 ``runtime/training/phases/bootstrap.py`` 里那个 ``if not dist_env.is_main():``
#: 块一一对应。新增抑制字段时必须同步加到这里 —— 见
#: ``test_rank_mutated_fields_list_matches_bootstrap`` 会自动核对。
RANK_MUTATED_ARGS = {"no_progress", "sample_steps", "sample_every"}

#: bootstrap 在非 rank 0 上置空的 ``ctx`` 字段，同理。
RANK_MUTATED_CTX = {"monitor_server", "wandb_monitor"}

#: 含集合操作的文件。
FILES = (
    "runtime/training/loop.py",
    "runtime/training/phases/resume.py",
)

#: 集合操作的方法名。``all_reduce_mean`` 也算 —— 它内部是 ``dist.all_reduce``。
COLLECTIVE_CALLS = {"barrier", "all_reduce_mean"}

REPO = pathlib.Path(__file__).resolve().parent.parent


def _referenced_names(node: ast.AST) -> set[str]:
    """节点里出现的所有名字，``a.b`` 形态归一成 ``"a.b"``。"""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            out.add(f"{sub.value.id}.{sub.attr}")
        elif isinstance(sub, ast.Name):
            out.add(sub.id)
    return out


def _local_assignments(tree: ast.AST) -> dict[str, set[str]]:
    """局部变量名 → 它被赋值时引用到的名字集合（并集，覆盖多处赋值）。

    门控经常写成 ``_sample_steps = ctx.sample_steps_all_ranks`` 再 ``if
    _sample_steps > 0``，所以要顺着局部变量再查一层，否则「换个变量名」就能绕过
    这条测试。
    """
    table: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        if node.value is None:
            continue
        refs = _referenced_names(node.value)
        for name in targets:
            table.setdefault(name, set()).update(refs)
    return table


def _resolve(names: set[str], locals_table: dict[str, set[str]]) -> set[str]:
    """把局部变量展开成它们引用的名字（带环保护，最多展开 8 层）。"""
    seen = set(names)
    frontier = set(names)
    for _ in range(8):
        nxt: set[str] = set()
        for name in frontier:
            if "." in name:
                continue
            nxt |= locals_table.get(name, set()) - seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def _risky(names: set[str]) -> set[str]:
    """名字集合里踩到 rank 相关字段的那些。"""
    bad = set()
    for name in names:
        if "." not in name:
            continue
        obj, attr = name.rsplit(".", 1)
        if obj == "args" and attr in RANK_MUTATED_ARGS:
            bad.add(name)
        elif obj == "ctx" and attr in RANK_MUTATED_CTX:
            bad.add(name)
    return bad


def _collective_gates(path: str) -> list[tuple[int, str, set[str]]]:
    """``(行号, 方法名, 外层所有 if 条件引用到的名字)``，局部变量已展开。"""
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    locals_table = _local_assignments(tree)

    found: list[tuple[int, str, set[str]]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in COLLECTIVE_CALLS
        ):
            continue
        used: set[str] = set()
        cur: ast.AST = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.If):
                used |= _referenced_names(cur.test)
            elif isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
        found.append((node.lineno, node.func.attr, _resolve(used, locals_table)))
    return found


@pytest.mark.parametrize("path", FILES)
def test_collective_gates_use_rank_invariant_conditions(path: str) -> None:
    """每个集合操作的外层条件都不能读 bootstrap 按 rank 改写的字段。"""
    offenders = [
        (lineno, call, sorted(bad))
        for lineno, call, used in _collective_gates(path)
        if (bad := _risky(used))
    ]
    assert not offenders, (
        f"{path} 有集合操作的门控读了 rank 相关字段，非 rank 0 会连它一起跳过 → "
        f"NCCL 集合操作永久错位：\n"
        + "\n".join(
            f"  第 {ln} 行 {call}() 的外层条件用到 {bad}"
            for ln, call, bad in offenders
        )
        + "\n改成读 ctx.sample_steps_all_ranks / ctx.sample_every_all_ranks 那类 "
        "rank 不变副本。"
    )


@pytest.mark.parametrize("path", FILES)
def test_every_file_actually_has_collectives(path: str) -> None:
    """确认上面那条测的是真东西 —— 文件里得真有集合操作。

    防的是「文件被重构、集合操作搬走了，而上面那条因为找不到目标而空过」。
    """
    found = _collective_gates(path)
    assert found, f"{path} 里没找到任何集合操作调用，上面那条测试等于没跑"


def test_rank_mutated_fields_list_matches_bootstrap() -> None:
    """``RANK_MUTATED_ARGS`` 必须与 bootstrap 里实际改写的字段一致。

    这条是本文件的**自检**：如果以后有人在 bootstrap 的 ``if not is_main():`` 块里
    多抑制一个字段却没同步这里，上面的门控检查就会对那个新字段视而不见 —— 这条测试
    会先失败，把人拦在那一步。
    """
    tree = ast.parse(
        (REPO / "runtime/training/phases/bootstrap.py").read_text(encoding="utf-8")
    )
    mutated: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        # 匹配 `if not dist_env.is_main():`
        if "'is_main'" not in test_src or not isinstance(node.test, ast.UnaryOp):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for tgt in stmt.targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "args"
                ):
                    mutated.add(tgt.attr)

    assert mutated, "没在 bootstrap 里找到 `if not is_main():` 的 args 改写块"
    assert mutated == RANK_MUTATED_ARGS, (
        f"bootstrap 实际改写 args.{sorted(mutated)}，本文件登记的是 "
        f"{sorted(RANK_MUTATED_ARGS)}。请同步 RANK_MUTATED_ARGS —— 漏登记会让门控"
        f"检查对新字段视而不见。"
    )


def test_bootstrap_snapshots_before_mutating() -> None:
    """rank 不变副本必须在 rank 相关改写**之前**存好。

    顺序反了的话副本里存的就是被置 0 之后的值，非 rank 0 上依然是 0，等于没修。
    """
    src = (REPO / "runtime/training/phases/bootstrap.py").read_text(encoding="utf-8")
    snapshot = src.index("ctx.sample_steps_all_ranks =")
    mutation = src.index("args.sample_steps = 0")
    assert snapshot < mutation, (
        "ctx.sample_steps_all_ranks 的赋值出现在 args.sample_steps=0 之后，"
        "副本会存到被置 0 的值"
    )


def test_snapshot_is_not_rank_gated() -> None:
    """存副本那两行本身不能被 ``is_main()`` 包住。

    若只有 rank 0 存副本，其余 rank 的副本是 dataclass 默认值 0，门控又变回各 rank
    不一致 —— 换个位置犯同一个错。
    """
    tree = ast.parse(
        (REPO / "runtime/training/phases/bootstrap.py").read_text(encoding="utf-8")
    )
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Attribute)
            and node.attr in {"sample_steps_all_ranks", "sample_every_all_ranks"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "ctx"
        ):
            continue
        cur: ast.AST = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.If):
                assert "'is_main'" not in ast.dump(cur.test), (
                    f"ctx.{node.attr} 的赋值被 is_main() 门控住了 —— "
                    f"其余 rank 会拿到 dataclass 默认值 0"
                )
            elif isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
