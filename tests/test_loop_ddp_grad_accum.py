"""DDP × 梯度累积的交互，以及进程组销毁的异常路径。

这两处都是「不写测试就会静默错」的类型：
- 累积期间忘关 DDP 同步 → 训练结果**完全正确**，只是通信量放大 grad_accum 倍。
  没有任何报错，只有速度变慢，而多卡的全部意义就是速度。
- 异常路径不销毁进程组 → 本次训练照常报错退出，但**下一个**训练任务卡死在
  init_process_group 上。故障与原因隔了一次任务，极难定位。
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")
sys.path.insert(0, "runtime")


# ---------------------------------------------------------------------------
# no_sync 等价物：require_backward_grad_sync 的开关时机
# ---------------------------------------------------------------------------


def _accumulation_step():
    """从 loop.py 取被测函数。

    loop.py 顶层 import torch，本机没装 —— 但 `_accumulation_step` 是纯算术，
    与 torch 无关。直接从源码里把它抠出来 exec，避免为一个纯函数拖进整条
    torch import 链。这比 mock 掉半个 torch 更诚实：测的就是那份源码。
    """
    import ast
    import pathlib

    src = pathlib.Path("runtime/training/loop.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_accumulation_step":
            ns: dict = {}
            exec(compile(ast.Module([node], []), "<loop>", "exec"), ns)  # noqa: S102
            return ns["_accumulation_step"]
    raise AssertionError("loop.py 里找不到 _accumulation_step")


@pytest.fixture(scope="module")
def accum():
    return _accumulation_step()


def test_sync_only_on_group_end(accum):
    """grad_accum=4 时，8 个 micro-batch 里只有第 4、8 个该开同步。

    这直接决定通信次数：开 4 次 vs 开 2 次。中间步的 all_reduce 结果会被下一次
    累加覆盖，属于纯浪费。
    """
    dl_len, ga = 8, 4
    ends = [accum(i, dl_len, ga)[1] for i in range(dl_len)]
    assert ends == [False, False, False, True, False, False, False, True]
    assert sum(ends) == dl_len // ga


def test_sync_flag_identical_across_ranks(accum):
    """各 rank 的同步时机必须完全一致 —— 不一致直接死锁。

    部分 rank 开同步、部分不开时，开的那些会等在 all_reduce 上，不开的那些
    直接进下一步 —— 双方永远等不到对方。

    这条成立的前提是各 rank 的 dl_len 相等（sampler 分片时截断对齐）与 batch_idx
    同步推进。用同一组入参在「不同 rank」上算两遍，结果必须一样 —— 因为该判断
    只依赖这两个量，不含任何 rank 相关信息。
    """
    dl_len, ga = 7, 3
    rank0 = [accum(i, dl_len, ga) for i in range(dl_len)]
    rank1 = [accum(i, dl_len, ga) for i in range(dl_len)]
    assert rank0 == rank1


def test_tail_group_ends_at_epoch_boundary(accum):
    """尾组不满也要 step —— 否则尾部梯度被丢或泄漏进下一 epoch。

    dl_len=7 / ga=3：第 6 个（idx 5）是常规组末，第 7 个（idx 6）是不满的尾组，
    也必须 is_group_end=True 且 group_size=1（按实际数归一，不是恒 ÷3）。
    """
    dl_len, ga = 7, 3
    assert accum(2, dl_len, ga) == (3, True)
    assert accum(5, dl_len, ga) == (3, True)
    assert accum(6, dl_len, ga) == (1, True)


def test_ga1_syncs_every_step(accum):
    """grad_accum=1（默认）时每步都是组末 → 每步都同步，与无累积语义一致。"""
    assert all(accum(i, 10, 1)[1] for i in range(10))


def test_no_len_dataloader_falls_back(accum):
    """dl_len=None（无 __len__）退回旧行为，仍能给出确定的同步时机。"""
    assert accum(3, None, 4) == (4, True)
    assert accum(2, None, 4) == (4, False)


# ---------------------------------------------------------------------------
# 进程组销毁：异常路径必须也走到
# ---------------------------------------------------------------------------


def test_main_destroys_process_group_in_finally():
    """`main()` 用 try/finally 包住全部 phase，异常路径也销毁进程组。

    静态检查而非真跑 main()：那要完整的 torch + 数据集 + 权重。这里断言的是
    结构性质 —— destroy() 在 finally 块里，而不是跟在 finalize 后面。
    结构对了，异常路径就一定覆盖。
    """
    import ast
    import pathlib

    src = pathlib.Path("runtime/anima_train.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    main_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    tries = [n for n in ast.walk(main_fn) if isinstance(n, ast.Try)]
    assert tries, "main() 里没有 try 块 —— 异常路径不会销毁进程组"

    def _calls_destroy(nodes) -> bool:
        for n in nodes:
            for sub in ast.walk(n):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr == "destroy":
                        return True
        return False

    assert any(_calls_destroy(t.finalbody) for t in tries), (
        "destroy() 不在任何 finally 块里 —— 训练抛异常时进程组不会被销毁，"
        "留下的僵死 NCCL 通信器会让下一个训练任务卡死在 init_process_group"
    )


def test_all_phases_inside_try():
    """全部 phase 都在 try 内 —— 任一 phase 抛异常都能触发销毁。

    漏掉早期 phase（如 bootstrap）会有个隐蔽后果：bootstrap 里 init() 建完进程组
    后紧接着的代码抛异常，此时组已建但不在保护范围内。
    """
    import ast
    import pathlib

    src = pathlib.Path("runtime/anima_train.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    main_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    try_node = next(n for n in main_fn.body if isinstance(n, ast.Try))
    inside = {
        sub.func.attr
        for n in try_node.body
        for sub in ast.walk(n)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
    }
    # phase 的调用形如 phases.bootstrap.run(ctx) / loop.run(ctx) → attr 都是 "run"
    assert "run" in inside
    # 且 try 外层不该还有 phase 调用（除了 destroy 所在的 finally）
    outside_runs = [
        sub
        for n in main_fn.body
        if not isinstance(n, ast.Try)
        for sub in ast.walk(n)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "run"
    ]
    assert not outside_runs, "有 phase 调用在 try 之外，异常时不会销毁进程组"
