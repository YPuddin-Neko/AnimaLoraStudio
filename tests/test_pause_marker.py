"""多卡暂停的文件通道。

背景（真机实测）：torchrun 下进程树是 ``supervisor → elastic agent → N × worker``，
supervisor 只持有 agent 的 pid。发 SIGINT 给 agent，它接住后**转发 SIGTERM** 给
worker 并自己抛 SignalException —— worker 的 ``handle_interrupt`` 只注册了
SIGINT/SIGBREAK，收不到 SIGTERM，于是 pause snapshot 不会被写出来、任务无法续训。
日志里能看到 agent 的 "Sending process ... closing signal SIGTERM"。

文件标记绕开信号语义，且让所有 rank 在同一个循环位置看到请求 —— 信号做不到这种
同步：各 rank 收到的时刻不同，一个已 ``sys.exit`` 而另一个还等在 all_reduce 上会
挂住整组。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, ".")

from utils import distributed as d  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("LORA_TASK_DIR", raising=False)
    yield


def test_no_marker_path_without_task_dir():
    """裸 CLI 训练（无 LORA_TASK_DIR）→ 路径为 None，轮询短路。

    CLI 用 Ctrl+C，信号那条路在单进程下工作正常，不需要文件通道。返回 None 让
    每步的轮询变成一次 dict 查找，开销可忽略。
    """
    assert d.pause_marker_path() is None
    assert d.pause_requested() is False


def test_marker_path_derived_from_task_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    p = d.pause_marker_path()
    assert p is not None
    assert p.parent == tmp_path
    assert p.name == d.PAUSE_MARKER_NAME


def test_pause_requested_reflects_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    assert d.pause_requested() is False
    (tmp_path / d.PAUSE_MARKER_NAME).write_text("1", encoding="utf-8")
    assert d.pause_requested() is True


def test_clear_removes_marker(monkeypatch, tmp_path):
    """rank 0 退出前必须删掉标记。

    不删的话下次 resume 会立刻又看到标记、立刻又暂停 —— 表现为「一点继续就又停了」，
    而且用户会以为是 resume 坏了。
    """
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    marker = tmp_path / d.PAUSE_MARKER_NAME
    marker.write_text("1", encoding="utf-8")
    d.clear_pause_marker()
    assert not marker.exists()
    assert d.pause_requested() is False


def test_clear_is_idempotent(monkeypatch, tmp_path):
    """标记不存在时删除不该抛 —— 非 rank 0 可能先删过，或用户手工删了。"""
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    d.clear_pause_marker()
    d.clear_pause_marker()


def test_clear_without_task_dir_is_noop():
    d.clear_pause_marker()


def test_pause_requested_survives_unreadable_dir(monkeypatch, tmp_path):
    """目录被删/权限问题时按「没有请求」处理，不能抛。

    这是每个训练步都跑的热路径，抛异常会把一次 IO 抖动升级成训练崩溃。
    """
    missing = tmp_path / "gone"
    monkeypatch.setenv("LORA_TASK_DIR", str(missing))
    assert d.pause_requested() is False


def test_marker_name_is_shared_constant():
    """supervisor 与训练侧必须用同一个常量，不能各写一份字符串。

    两边写死不同的文件名 = 暂停永远不生效，而且没有任何报错（supervisor 以为发出去
    了、训练侧永远看不到）。这条钉住 supervisor 是 import 而非复制。
    """
    import pathlib

    src = pathlib.Path("studio/supervisor/core.py").read_text(encoding="utf-8")
    assert "from utils.distributed import PAUSE_MARKER_NAME" in src
    assert f'"{d.PAUSE_MARKER_NAME}"' not in src, (
        "supervisor 里出现了写死的标记文件名 —— 应该只用 import 来的常量"
    )


# ---------------------------------------------------------------------------
# 训练循环的响应位置
# ---------------------------------------------------------------------------


def _interrupt_sites() -> list[tuple[int, str]]:
    """[(行号, 包裹条件)]，取 loop.py 里每处真实的 handle_interrupt 调用。

    用 AST 而非正则：条件写成多行（黑格式化后很常见）时 ``if ... pause_requested()``
    的单行正则就匹配不到了，会静默落到**另一处**闸上去检查，等于测了错的东西。
    """
    import ast
    import pathlib
    import textwrap

    src = textwrap.dedent(
        pathlib.Path("runtime/training/loop.py").read_text(encoding="utf-8")
    )
    sites: list[tuple[int, str]] = []

    def walk(node, conds):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                t = ast.unparse(child.test)
                for s in child.body:
                    walk_stmt(s, [*conds, t])
                for s in child.orelse:
                    walk_stmt(s, conds)
            else:
                walk(child, conds)

    def walk_stmt(stmt, conds):
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

    walk(ast.parse(src), [])
    return sites


def test_loop_checks_pause_only_at_group_end():
    """batch 循环内的轮询必须在**组末步**才响应，不能在梯度累积中间退出。

    mid-accumulation 退出会留下悬挂的 partial backward 梯度（与 ADR 0006
    Addendum 1 放弃 mid-epoch save 的理由同源）。等到组末再响应，最多多跑
    grad_accum-1 个 micro-batch。

    注意有**两处**暂停闸，要求不同：
      - batch 循环内（本测试）：必须带 _is_group_end_pre
      - epoch 末、采样之前：batch 循环已退出，不存在 mid-accumulation，
        所以**不该**要求组末判定（见 test_pause_multigpu_signal.py）
    """
    sites = _interrupt_sites()
    assert sites, "loop.py 里找不到 handle_interrupt 调用"

    group_end_sites = [(ln, c) for ln, c in sites if "_is_group_end_pre" in c]
    assert group_end_sites, (
        "找不到带 _is_group_end_pre 的暂停闸 —— batch 循环内的轮询若不绑组末判定，"
        f"mid-accumulation 退出会留悬挂梯度。实际各处条件：{sites}"
    )
    for lineno, cond in sites:
        assert "not ctx.interrupted" in cond, (
            f"loop.py:{lineno} 的暂停闸没挡已 interrupted 状态 —— handle_interrupt "
            f"二次触发会走强退分支 exit(1)，把正常暂停变成失败退出。\n条件：{cond}"
        )


# ---------------------------------------------------------------------------
# 启动期检查点
# ---------------------------------------------------------------------------


def test_exit_if_pause_requested_noop_without_marker(monkeypatch, tmp_path):
    """没有标记时什么都不做 —— 每个 phase 之间都会调它，不能有副作用。"""
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    d.exit_if_pause_requested()  # 不抛、不退出即通过


def test_exit_if_pause_requested_exits_zero(monkeypatch, tmp_path):
    """见到标记 → SystemExit(0)。

    退出码必须是 0 而不是非零：非零会让 supervisor 的失败摘要把它当训练崩溃处理，
    而这是用户主动要求的停止。cancel_pending 已由 supervisor 置上，收尾走 canceled。
    """
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    (tmp_path / d.PAUSE_MARKER_NAME).write_text("1", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        d.exit_if_pause_requested()
    assert ei.value.code == 0


def test_exit_if_pause_requested_clears_marker_and_emits(monkeypatch, tmp_path):
    """rank 0 退出前删标记并把原因写进 task log。

    删标记是为了下次 resume 不会立刻又停（与训练循环那条同源）。emit 的消息要说清
    「无 epoch 末备份 → 标记为已取消」，否则用户看到任务变 canceled 会以为出错了。
    """
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    marker = tmp_path / d.PAUSE_MARKER_NAME
    marker.write_text("1", encoding="utf-8")
    seen: list[str] = []
    with pytest.raises(SystemExit):
        d.exit_if_pause_requested(seen.append)
    assert not marker.exists(), "rank 0 退出前必须删标记"
    assert seen and "取消" in seen[0], f"emit 的消息没说明后果: {seen}"


def test_exit_if_pause_requested_non_main_does_not_clear(monkeypatch, tmp_path):
    """非 rank 0 不删标记 —— 竞态理由同训练循环那条。

    若非 rank 0 先删，rank 0 还没走到检查点就看不见标记、继续往下跑，而其他 rank
    已经退出 → rank 0 卡在下一个集合操作上。
    """
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    marker = tmp_path / d.PAUSE_MARKER_NAME
    marker.write_text("1", encoding="utf-8")
    with pytest.raises(SystemExit):
        d.exit_if_pause_requested()
    assert marker.exists(), "非 rank 0 不该删标记"


def test_startup_checkpoints_between_every_phase():
    """每个长 phase 之后都要有检查点。

    真机上停止请求落在「加载文本编码器」那段（bootstrap 到进训练循环之间），当时
    一个检查点都没有 —— 用户只能取消，而取消在 torchrun 下会抛 SignalException 加
    30 行 traceback，看着像崩溃。

    断言 phase 调用与检查点在 main() 里交替出现，而不只是「存在至少一个」。
    """
    import pathlib
    import re

    src = pathlib.Path("runtime/anima_train.py").read_text(encoding="utf-8")
    body = src[src.find("def main("):]
    # 抓 phase 调用与检查点，按出现顺序
    seq = [
        ("phase", m.group(1)) if m.group(1) else ("check", "")
        for m in re.finditer(
            r"phases\.(\w+)\.(?:run|finish)\(ctx\)|exit_if_pause_requested\(", body,
        )
    ]
    kinds = [k for k, _ in seq]
    # 前四个 phase（bootstrap/models/dataset/text_cache）后面各要跟一个检查点
    covered = {
        name for (k, name), (nk, _) in zip(seq, seq[1:]) if k == "phase" and nk == "check"
    }
    for required in ("bootstrap", "models", "dataset", "text_cache"):
        assert required in covered, (
            f"{required} phase 之后没有停止检查点 —— 该 phase 在真机上是分钟级，"
            f"这段时间用户无法干净地停止训练"
        )
    assert kinds.count("check") >= 4


def test_baseline_sampling_is_rank0_gated_with_barriers():
    """resume phase 的 step-0 基线采样必须 rank0 门控 + 前后 barrier。

    这处**曾经漏了**（loop.py 的 step / epoch 采样都加了，唯独 resume.py 没有）。
    真机后果：rank 0 还在基线采样、rank 1 已经冲进训练前向的第一个 attention，两者
    显存峰值叠在同一时间窗口 → rank 1 OOM。次要后果是各 rank 并发写同一批
    step_0_baseline_*.png，必然写坏。

    按 AST 结构断言而非文本 —— 要确认 barrier 在 `if` 里侧、`is_main()` 外侧
    （集合操作必须所有 rank 都执行）。
    """
    import ast
    import pathlib

    src = pathlib.Path("runtime/training/phases/resume.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    def _has_call(node, name: str) -> bool:
        """节点子树里有没有调用 ``name``。

        同时认 ``ast.Name``（裸函数名，如 ``run_sample(...)``）与 ``ast.Attribute``
        （带模块前缀，如 ``dist_env.barrier()``）—— 只认后者会漏掉 run_sample，
        本测试最初就是这么写错的，导致它在真实的漏 barrier 代码上也「通过」。
        """
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == name:
                return True
            if isinstance(f, ast.Name) and f.id == name:
                return True
        return False

    # 找包住 run_sample 的那个 if（基线采样块）
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _has_call(node, "run_sample"):
            target = node
            break
    assert target is not None, "resume.py 里找不到基线采样的 if 块"

    body_src = [ast.unparse(s) for s in target.body]
    assert any("barrier()" in s for s in body_src), (
        "基线采样块里没有 barrier —— rank 0 采样时其余 rank 会径直进训练前向，"
        "显存峰值重叠导致 OOM"
    )
    # barrier 至少两次（前后各一）
    assert sum(s.count("barrier()") for s in body_src) >= 2, (
        "barrier 少于两次 —— 采样前后都要挡：前者让 rank 0 在其余 rank 释放激活后"
        "才开始吃显存，后者让其余 rank 等它结束"
    )
    # run_sample 必须在 is_main() 门控内
    guarded = False
    for stmt in target.body:
        if isinstance(stmt, ast.If) and _has_call(stmt.test, "is_main"):
            if _has_call(stmt, "run_sample"):
                guarded = True
    assert guarded, "run_sample 不在 is_main() 门控内 —— 各 rank 会并发写同一批文件"


def test_loop_clears_marker_on_rank0_only():
    """只有 rank 0 删标记。

    非 rank 0 也删的话会出现竞态：rank 1 先删、rank 0 还没轮询到，于是 rank 0
    这一步看不见标记、继续训练，而 rank 1 已经退出 —— rank 0 卡死在下一次
    all_reduce 上。
    """
    import pathlib

    src = pathlib.Path("runtime/training/loop.py").read_text(encoding="utf-8")
    idx = src.find("clear_pause_marker")
    assert idx > 0, "loop.py 没调 clear_pause_marker"
    window = src[max(0, idx - 300):idx]
    assert "is_main()" in window, "clear_pause_marker 没被 is_main() 门控"
