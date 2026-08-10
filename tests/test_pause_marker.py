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


def test_loop_checks_pause_only_at_group_end():
    """轮询必须在**组末步**才响应，不能在梯度累积中间退出。

    mid-accumulation 退出会留下悬挂的 partial backward 梯度（与 ADR 0006
    Addendum 1 放弃 mid-epoch save 的理由同源）。等到组末再响应，最多多跑
    grad_accum-1 个 micro-batch。

    静态检查：条件里必须同时含 _is_group_end_pre 与 pause_requested。
    """
    import pathlib
    import re

    src = pathlib.Path("runtime/training/loop.py").read_text(encoding="utf-8")
    m = re.search(r"if [^\n]*pause_requested\(\)[^\n]*:", src)
    assert m, "loop.py 里没有 pause_requested() 的轮询"
    cond = m.group(0)
    assert "_is_group_end_pre" in cond, (
        "暂停轮询没和组末判定绑定 —— mid-accumulation 退出会留悬挂梯度"
    )
    assert "not ctx.interrupted" in cond, (
        "没挡已 interrupted 状态 —— handle_interrupt 二次触发会走强退分支 exit(1)，"
        "把正常暂停变成失败退出"
    )


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
